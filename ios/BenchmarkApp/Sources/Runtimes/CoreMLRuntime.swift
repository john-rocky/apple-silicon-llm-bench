import Foundation
import CoreML
#if canImport(CoreMLLLM)
import CoreMLLLM
#endif
#if canImport(Tokenizers)
import Tokenizers
#endif

/// CoreML LLM adapter using `john-rocky/CoreML-LLM` (`CoreMLLLM` Swift
/// package). This is the loader the `mlboydaisuke/*-coreml` bundles are
/// published for, and it is the only Swift path today that runs Gemma 4
/// on iPhone via CoreML (swift-transformers' `LanguageModel.loadCompiled`
/// requires a single stateful `.mlpackage` which Gemma 4 does not ship as).
///
/// Two engine paths, picked at load time:
///
/// 1. `CoreMLLLM.load(model:)` — the generic library entry point that
///    auto-detects `chunk1.mlpackage` / monolithic / Gemma4 stateful
///    layouts. Used for Gemma 4 E2B/E4B, LFM2, Qwen 2.5.
///
/// 2. `Qwen35MLKVGenerator` — direct instantiation for Qwen 3.5 0.8B / 2B.
///    Qwen 3.5 ships a `chunk_a..d` MLKV layout that the generic
///    `load(from:)` path does not detect (CoreML-LLM v1.9.0 added the
///    public API but routing is intentionally opt-in because the MLKV
///    generator has a different prompt/decode contract than the streaming
///    `CoreMLLLM.stream` API).
///
/// Auto-downloads the model bundle into `Documents/Models/<folderName>/`
/// on first use, then ANE-compiles the chunks (slow on first run, ~1–2 min
/// per the upstream README).
///
/// Requires iOS 18+ / Swift 6.
@available(iOS 18.0, macOS 15.0, *)
public final class CoreMLRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .coreMLLLM
    #if canImport(CoreMLLLM)
    public let isAvailable: Bool = true
    #else
    public let isAvailable: Bool = false
    #endif
    public let supportedModels: [ModelInfo] = ModelCatalog.coreML

    nonisolated(unsafe) private var _loadedModelId: String?
    public var loadedModelId: String? { _loadedModelId }

    #if canImport(CoreMLLLM)
    private enum Engine {
        case general(CoreMLLLM)
        case qwen35(Qwen35MLKVGenerator, any Tokenizer)
    }
    nonisolated(unsafe) private var engine: Engine?
    #endif

    public init() {}

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }

        #if canImport(CoreMLLLM)
        guard let info = Self.downloaderInfo(for: model.id) else {
            throw LLMRuntimeError.loadFailed(
                "Model id \(model.id) is not registered in CoreMLLLM.ModelDownloader.ModelInfo.defaults."
            )
        }

        do {
            switch model.id {
            case "coreml-llm/qwen3.5-0.8b", "coreml-llm/qwen3.5-2b":
                try await loadQwen35(model: model, info: info, progress: progress)
            default:
                let loaded = try await CoreMLLLM.load(model: info, computeUnits: .cpuAndNeuralEngine) { status in
                    progress(0.5)
                    _ = status
                }
                self.engine = .general(loaded)
                self._loadedModelId = model.id
                progress(1)
            }
        } catch {
            throw LLMRuntimeError.loadFailed(error.localizedDescription)
        }
        #else
        throw LLMRuntimeError.unsupported("CoreMLLLM SPM product not present.")
        #endif
    }

    #if canImport(CoreMLLLM)
    private func loadQwen35(
        model: ModelInfo,
        info: ModelDownloader.ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        // Resolve the model folder. localModelURL returns the chunks
        // subdir (`qwen3_5_0_8b_decode_chunks_mlkv/`); the generator
        // wants the parent folder.
        let downloader = ModelDownloader.shared
        let modelURL: URL
        if let existing = downloader.localModelURL(for: info) {
            modelURL = existing
        } else {
            progress(0.1)
            modelURL = try await downloader.download(info)
        }
        let folder = modelURL.deletingLastPathComponent()

        let is08B = model.id == "coreml-llm/qwen3.5-0.8b"
        let cfg: Qwen35MLKVGenerator.Config = is08B ? .default0_8B : .default2B
        let gen = Qwen35MLKVGenerator(cfg: cfg)
        gen.setModelFolder(folder)
        progress(0.5)
        try await gen.load()

        let tokId = is08B ? "Qwen/Qwen3.5-0.8B" : "Qwen/Qwen3.5-2B"
        let tok = try await AutoTokenizer.from(pretrained: tokId)

        self.engine = .qwen35(gen, tok)
        self._loadedModelId = model.id
        progress(1)
    }
    #endif

    public func unloadModel() async {
        #if canImport(CoreMLLLM)
        engine = nil
        #endif
        _loadedModelId = nil
    }

    public func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGenerate(prompt: prompt, parameters: parameters, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGenerate(
        prompt: String,
        parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async throws {
        #if canImport(CoreMLLLM)
        guard let engine else { throw LLMRuntimeError.modelNotLoaded }

        let prefillStart = CFAbsoluteTimeGetCurrent()

        switch engine {
        case .general(let llm):
            var firstTokenAt: CFAbsoluteTime?
            var tokenCount = 0
            let stream = try await llm.stream(prompt, maxTokens: parameters.maxTokens)
            for await piece in stream {
                try Task.checkCancellation()
                if firstTokenAt == nil { firstTokenAt = CFAbsoluteTimeGetCurrent() }
                if !piece.isEmpty {
                    continuation.yield(.chunk(piece))
                    tokenCount += 1
                }
            }
            let end = CFAbsoluteTimeGetCurrent()
            let prefillTime = (firstTokenAt ?? end) - prefillStart
            let generateTime = max(end - (firstTokenAt ?? prefillStart), 0.001)
            continuation.yield(.info(GenerationInfo(
                promptTokenCount: 0,
                generationTokenCount: tokenCount,
                promptTime: prefillTime,
                generateTime: generateTime,
                stopReason: tokenCount >= parameters.maxTokens ? .length : .stop
            )))

        case .qwen35(let gen, let tok):
            let chatMessages: [Message] = [["role": "user", "content": prompt]]
            let inputIds: [Int] = (try? tok.applyChatTemplate(messages: chatMessages))
                ?? tok.encode(text: prompt)
            let inputIdsInt32 = inputIds.map { Int32($0) }
            var eosSet: Set<Int32> = [248044, 248045, 248046]
            if let eid = tok.eosTokenId { eosSet.insert(Int32(eid)) }

            // Mutable state captured by the onToken closure. The
            // generator drives onToken serially from its decode loop
            // (no concurrent calls), so unchecked mutation here is safe.
            final class StreamState: @unchecked Sendable {
                var firstTokenAt: CFAbsoluteTime?
                var tokenCount: Int = 0
                var accumIds: [Int] = []
                var emittedText: String = ""
            }
            let state = StreamState()

            _ = try await gen.generate(
                inputIds: inputIdsInt32,
                maxNewTokens: parameters.maxTokens,
                temperature: 0.0,
                topK: 40,
                topP: 1.0,
                repetitionPenalty: 1.1,
                eosTokenIds: eosSet,
                onToken: { tokenId in
                    if state.firstTokenAt == nil {
                        state.firstTokenAt = CFAbsoluteTimeGetCurrent()
                    }
                    state.tokenCount += 1
                    if eosSet.contains(tokenId) { return }
                    state.accumIds.append(Int(tokenId))
                    let current = tok.decode(tokens: state.accumIds)
                    if current.count > state.emittedText.count {
                        let delta = String(current[state.emittedText.endIndex...])
                        state.emittedText = current
                        continuation.yield(.chunk(delta))
                    }
                }
            )

            let end = CFAbsoluteTimeGetCurrent()
            let prefillTime = (state.firstTokenAt ?? end) - prefillStart
            let generateTime = max(end - (state.firstTokenAt ?? prefillStart), 0.001)
            continuation.yield(.info(GenerationInfo(
                promptTokenCount: inputIds.count,
                generationTokenCount: state.tokenCount,
                promptTime: prefillTime,
                generateTime: generateTime,
                stopReason: state.tokenCount >= parameters.maxTokens ? .length : .stop
            )))
        }

        continuation.finish()
        #else
        throw LLMRuntimeError.unsupported("CoreMLLLM SPM product not present.")
        #endif
    }

    #if canImport(CoreMLLLM)
    /// Map our `ModelInfo.id` strings to the `CoreMLLLM.ModelDownloader.ModelInfo`
    /// registered defaults.
    private static func downloaderInfo(for id: String) -> ModelDownloader.ModelInfo? {
        switch id {
        case "coreml-llm/gemma4-e2b":         return .gemma4e2b
        case "coreml-llm/gemma4-e4b":         return .gemma4e4b
        case "coreml-llm/qwen3.5-0.8b":       return .qwen35_08b
        case "coreml-llm/qwen3.5-2b":         return .qwen35_2b
        case "coreml-llm/lfm2.5-350m":        return .lfm2_5_350m
        case "coreml-llm/qwen2.5-0.5b":       return .qwen25_05b
        default:                              return nil
        }
    }
    #endif
}
