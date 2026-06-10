import Foundation
#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
#endif
#if canImport(Tokenizers)
import Tokenizers
#endif

/// Apple **Core AI** adapter — the Core ML successor announced at WWDC 2026
/// (iOS / macOS 27). Loads a `.aimodel` LLM bundle produced by the official
/// `coreai.llm.export` pipeline and runs it through the official
/// `coreai-models` Swift runtime (`CoreAILM`), faithful to Apple's intended
/// on-device usage.
///
/// We deliberately use the low-level `EngineFactory` / `InferenceEngine` path —
/// the same one Apple's own `llm-benchmark` CLI tool uses
/// (`swift/Sources/Tools/benchmark/BenchmarkMain.swift`) — rather than the
/// high-level `LanguageModelSession`: it yields a raw token stream, so we get
/// true per-token timing (TTFT, inter-token latency) instead of a single
/// aggregate.
///
/// **Two compute paths, two bundles.** On iPhone the compute unit is decided by
/// the *export shape*, not just a runtime flag: the static iOS export
/// (`--platform iOS`) is detected as a chunked-static model → the `static-shape`
/// **ANE** engine; the dynamic export → the `coreai-pipelined` **GPU** engine.
/// So `…-ane` and `…-gpu` are two separate AOT-compiled bundles, distinguished
/// in the result rows with no schema change.
///
/// **iOS needs AOT compilation.** An exported `.aimodel` ships MLIR IR which
/// iOS cannot JIT; it must be compiled with `xcrun coreai-build compile
/// --platform iOS` to a `.aimodelc`, then `metadata.json assets.main` points at
/// the device-arch compiled file. See `methodology/coreai-ios.md` /
/// `scripts/bench_coreai_iphone.sh`.
///
/// The compiled bundles are **side-loaded** under `Documents/CoreAIModels/<name>/`
/// (large; not published to HF).
///
/// Requires iOS 27 / macOS 27 — the `coreai-models` Swift package floor. When
/// that package is not linked into the build (`canImport` false), this file
/// compiles to an unavailable stub so the rest of the app is unaffected.
public final class CoreAIRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .coreAI
    #if canImport(CoreAILanguageModels)
    public let isAvailable: Bool = true
    #else
    public let isAvailable: Bool = false
    #endif
    public let supportedModels: [ModelInfo] = ModelCatalog.coreAI

    nonisolated(unsafe) private var _loadedModelId: String?
    public var loadedModelId: String? { _loadedModelId }

    #if canImport(CoreAILanguageModels)
    nonisolated(unsafe) private var engine: (any InferenceEngine)?
    nonisolated(unsafe) private var tokenizer: (any Tokenizer)?
    nonisolated(unsafe) private var eosTokenIds: Set<Int32> = []
    #endif

    public init() {}

    // MARK: - Model id → bundle + compute variant

    /// Map a catalog id to its AOT-compiled bundle folder + the engine variant.
    /// The ANE bundle is the static iOS export compiled `--preferred-compute
    /// neural-engine` (structure → `static-shape`); the GPU bundle is the
    /// dynamic export compiled `--preferred-compute gpu` (structure →
    /// `coreai-pipelined`). The forced variant matches the bundle's structure;
    /// passing `nil` would auto-resolve to the same engine.
    private static func bundleSpec(for id: String) -> (folder: String, variant: String?)? {
        switch id {
        case "core-ai/qwen3-0.6b-ane": return ("qwen3_0_6b_ane", "static-shape")
        case "core-ai/qwen3-0.6b-gpu": return ("qwen3_0_6b_gpu", "coreai-pipelined")
        default:                       return nil
        }
    }

    /// Resolve a side-loaded `.aimodel` bundle folder on device. We look in
    /// `Documents/CoreAIModels/<folder>/` first (push it there with
    /// `xcrun devicectl device copy to …` or Finder file sharing), then fall
    /// back to an embedded app-bundle resource.
    private static func resolveBundleURL(folder: String) -> URL? {
        let fm = FileManager.default
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            let u = docs.appendingPathComponent("CoreAIModels/\(folder)", isDirectory: true)
            if fm.fileExists(atPath: u.appendingPathComponent("metadata.json").path) { return u }
        }
        if let res = Bundle.main.url(forResource: folder, withExtension: nil),
           fm.fileExists(atPath: res.appendingPathComponent("metadata.json").path) {
            return res
        }
        return nil
    }

    // MARK: - Load

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }
        #if canImport(CoreAILanguageModels)
        guard let spec = Self.bundleSpec(for: model.id) else {
            throw LLMRuntimeError.loadFailed("Unknown Core AI model id \(model.id).")
        }
        guard let bundleURL = Self.resolveBundleURL(folder: spec.folder) else {
            throw LLMRuntimeError.loadFailed(
                "Core AI bundle '\(spec.folder)' not found. Side-load the exported "
                + "folder into Documents/CoreAIModels/\(spec.folder)/ "
                + "(it must contain metadata.json, the .aimodel, and tokenizer/)."
            )
        }

        var step = "start"
        do {
            progress(0.15)
            // Mirror Apple's llm-benchmark tool: build a ModelConfig from the
            // LanguageBundle and hand it to EngineFactory.
            step = "LanguageBundle(\(bundleURL.lastPathComponent))"
            let bundle = try LanguageBundle(at: bundleURL)
            step = "requireModelURL"
            let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
            step = "ModelConfig"
            let engineConfig = ModelConfig(
                name: bundle.name,
                tokenizer: bundle.tokenizer,
                vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength,
                serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main"
            )
            let configData = try JSONEncoder().encode(engineConfig)
            progress(0.35)

            step = "EngineFactory(variant=\(spec.variant ?? "auto"), model=\(modelURL.lastPathComponent))"
            let options = EngineOptions(variant: spec.variant, kvCacheStrategy: .auto)
            let engine = try await EngineFactory.createEngine(
                config: configData,
                modelURL: modelURL,
                options: options
            )
            progress(0.7)

            step = "loadTokenizer"
            let tok = try await bundle.loadTokenizer()
            var eos: Set<Int32> = []
            if let e = tok.eosTokenId { eos.insert(Int32(e)) }

            // Trigger kernel compilation up front so it folds into load time.
            step = "warmup"
            try? await engine.warmup(queryLength: 8, sampling: SamplingConfiguration(temperature: 0))

            self.engine = engine
            self.tokenizer = tok
            self.eosTokenIds = eos
            self._loadedModelId = model.id
            progress(1)
        } catch let e as LLMRuntimeError {
            throw e
        } catch {
            throw LLMRuntimeError.loadFailed("[\(step)] \(error)")
        }
        #else
        throw LLMRuntimeError.unsupported("Core AI runtime not present in this build (requires the coreai-models Swift package, iOS/macOS 27).")
        #endif
    }

    public func unloadModel() async {
        #if canImport(CoreAILanguageModels)
        engine = nil
        tokenizer = nil
        eosTokenIds = []
        #endif
        _loadedModelId = nil
    }

    // MARK: - Generate

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
        #if canImport(CoreAILanguageModels)
        guard let engine, let tokenizer else { throw LLMRuntimeError.modelNotLoaded }

        // Tokenize with the model's chat template (greedy, deterministic — the
        // same sampling Apple's benchmark tool uses: temperature 0).
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let promptIds: [Int] = (try? tokenizer.applyChatTemplate(messages: messages))
            ?? tokenizer.encode(text: prompt)
        let inputIds = promptIds.map { Int32($0) }

        try await engine.reset()
        let sampling = SamplingConfiguration(temperature: 0)
        let options = InferenceOptions(maxTokens: parameters.maxTokens, includeLogits: false)

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var firstTokenAt: CFAbsoluteTime?
        var genCount = 0
        var accumIds: [Int] = []
        var emitted = ""

        let stream = try engine.generate(
            with: inputIds,
            samplingConfiguration: sampling,
            inferenceOptions: options
        )
        for try await out in stream {
            try Task.checkCancellation()
            if firstTokenAt == nil { firstTokenAt = CFAbsoluteTimeGetCurrent() }
            let tid = out.tokenId
            genCount += 1
            if eosTokenIds.contains(tid) { break }
            accumIds.append(Int(tid))
            // Incremental decode → emit only the new text so the runner gets
            // real per-token timing for inter-token-latency percentiles.
            let current = tokenizer.decode(tokens: accumIds)
            if current.count > emitted.count {
                let delta = String(current[emitted.endIndex...])
                emitted = current
                continuation.yield(.chunk(delta))
            }
        }

        let end = CFAbsoluteTimeGetCurrent()
        let promptTime = (firstTokenAt ?? end) - prefillStart
        let generateTime = max(end - (firstTokenAt ?? prefillStart), 0.001)
        continuation.yield(.info(GenerationInfo(
            promptTokenCount: inputIds.count,
            generationTokenCount: genCount,
            promptTime: promptTime,
            generateTime: generateTime,
            stopReason: genCount >= parameters.maxTokens ? .length : .stop
        )))
        continuation.finish()
        #else
        throw LLMRuntimeError.unsupported("Core AI runtime not present in this build.")
        #endif
    }
}
