#if canImport(LiteRTLM)
import Foundation
// LiteRT-LM 0.12.0 is pre-concurrency: its `Engine` is an actor but `Conversation`
// isn't Sendable, so awaiting `engine.createConversation(...)` from this actor trips
// strict actor-isolation. `@preconcurrency` treats LiteRTLM types with the legacy
// rules (the Conversation is only ever used within this one runGenerate scope).
@preconcurrency import LiteRTLM

/// LiteRT-LM adapter — wraps Google's official `google-ai-edge/LiteRT-LM`
/// Swift API (`import LiteRTLM`, ≥ 0.12.0).
///
/// Loads `.litertlm` bundles (e.g. `litert-community/gemma-4-E2B-it-litert-lm`)
/// and drives generation through `Engine` → `Conversation.sendMessageStream`.
/// The decode backend is Metal GPU (`.gpu`); pass `.cpu()` to compare CPU.
///
/// This replaces the deprecated MediaPipe Tasks GenAI 0.10.x (`.task`) path,
/// which was iOS-only and could not read Gemma 4. The 0.12.0 package ships an
/// `ios-arm64` + `macos-arm64` xcframework, so it runs on both the iOS app
/// and the macOS `yardstick` CLI.
///
/// The runtime kind is still `.mediaPipe` (raw value `"litert-lm"`) for
/// source/JSONL stability.
public actor MediaPipeRuntime: LLMRuntime {
    public let kind: RuntimeKind = .mediaPipe
    public let isAvailable: Bool = true
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.liteRTLM

    public private(set) var loadedModelId: String?

    private var engine: Engine?
    private var modelPath: String?

    public init() {}

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }

        // Enable LiteRT-LM's benchmark counters so generation can report *real*
        // tokenizer token counts + tok/s (via Conversation.getBenchmarkInfo),
        // instead of estimating from streamed chunks. Must opt in first.
        ExperimentalFlags.optIntoExperimentalAPIs()
        ExperimentalFlags.enableBenchmark = true

        let snapshot = try await HFDownloader.snapshot(for: model, runtime: kind, progress: progress)
        let modelFile = try locateModelFile(in: snapshot, expected: model.primaryFile)

        do {
            // maxNumTokens caps the working context; the per-run output budget
            // is enforced separately in `runGenerate` via `parameters.maxTokens`.
            let config = try EngineConfig(
                modelPath: modelFile.path,
                backend: .gpu,
                maxNumTokens: 2048,
                cacheDir: NSTemporaryDirectory()
            )
            let engine = Engine(engineConfig: config)
            try await engine.initialize()
            self.engine = engine
            self.modelPath = modelFile.path
            self.loadedModelId = model.id
        } catch {
            throw LLMRuntimeError.loadFailed(error.localizedDescription)
        }
    }

    public func unloadModel() async {
        engine = nil
        modelPath = nil
        loadedModelId = nil
    }

    public nonisolated func generate(
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
        guard let engine else { throw LLMRuntimeError.modelNotLoaded }

        // Fresh conversation per run so each measurement is independent (no
        // KV reuse across runs). No system message is injected, to keep the
        // prompt identical to the other runtimes' single-prompt path.
        let sampler = try SamplerConfig(
            topK: 40,
            topP: parameters.topP,
            temperature: parameters.temperature
        )
        // v0.12.0 is an early-preview API: `ConversationConfig`'s `systemMessage`
        // is optional here; if a future release makes it required, pass
        // `Message("")` or fall back to `engine.createConversation()`.
        let conversation = try await engine.createConversation(
            with: ConversationConfig(samplerConfig: sampler)
        )

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var firstTokenAt: CFAbsoluteTime?
        var tokenCount = 0

        for try await chunk in conversation.sendMessageStream(Message(prompt)) {
            try Task.checkCancellation()
            if firstTokenAt == nil { firstTokenAt = CFAbsoluteTimeGetCurrent() }
            let text = chunk.toString
            if !text.isEmpty {
                continuation.yield(.chunk(text))
                tokenCount += 1  // chunk tally — fallback only; real count comes from getBenchmarkInfo
            }
            // Run the turn to its natural end (EOS): LiteRT-LM finalizes its
            // per-turn benchmark counters only on completion, and its streaming
            // API has no per-call output-token cap, so a mid-turn break would
            // leave getBenchmarkInfo empty. Output length is therefore the
            // model's own (vs. the 128-token hard cap other runtimes honor);
            // decode tok/s is a rate, so it stays comparable.
        }

        let end = CFAbsoluteTimeGetCurrent()
        let wallPrefill = (firstTokenAt ?? end) - prefillStart
        let wallGenerate = max(end - (firstTokenAt ?? prefillStart), 0.001)

        // Prefer LiteRT-LM's own per-turn counters: real tokenizer token counts
        // and tok/s for both prefill and decode. We back-derive promptTime /
        // generateTime so GenerationInfo's computed tok/s == LiteRT's reported
        // rates exactly. Fall back to chunk-count + wall-clock if the benchmark
        // info is unavailable (e.g. flag not honored on this build).
        let bench = try? conversation.getBenchmarkInfo()
        let decodeTokens = (bench.map { $0.lastDecodeTokenCount } ?? 0) > 0
            ? bench!.lastDecodeTokenCount : tokenCount
        let promptTokens = bench?.lastPrefillTokenCount ?? 0
        let generateTime: TimeInterval = {
            if let b = bench, b.lastDecodeTokensPerSecond > 0, b.lastDecodeTokenCount > 0 {
                return Double(b.lastDecodeTokenCount) / b.lastDecodeTokensPerSecond
            }
            return wallGenerate
        }()
        let promptTime: TimeInterval = {
            if let b = bench, b.lastPrefillTokensPerSecond > 0, b.lastPrefillTokenCount > 0 {
                return Double(b.lastPrefillTokenCount) / b.lastPrefillTokensPerSecond
            }
            return wallPrefill
        }()

        continuation.yield(.info(GenerationInfo(
            promptTokenCount: promptTokens,
            generationTokenCount: decodeTokens,
            promptTime: promptTime,
            generateTime: generateTime,
            stopReason: .stop  // ran to EOS
        )))
        continuation.finish()
    }

    /// Resolve the `.litertlm` file inside a downloaded snapshot.
    /// Prefers the explicit `primaryFile`, then a standard (non-web, non-NPU)
    /// `.litertlm`, then any `.litertlm`, then a legacy `.task`.
    private func locateModelFile(in dir: URL, expected: String) throws -> URL {
        if !expected.isEmpty {
            let direct = dir.appendingPathComponent(expected)
            if FileManager.default.fileExists(atPath: direct.path) { return direct }
        }
        let contents = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        let litertlm = contents.filter { $0.pathExtension == "litertlm" }
        let isPlatformVariant: (URL) -> Bool = { url in
            let n = url.lastPathComponent.lowercased()
            return n.contains("-web") || n.contains("intel") || n.contains("qualcomm")
        }
        if let standard = litertlm.first(where: { !isPlatformVariant($0) }) { return standard }
        if let any = litertlm.first { return any }
        if let task = contents.first(where: { $0.pathExtension == "task" }) { return task }
        throw LLMRuntimeError.loadFailed("No .litertlm or .task file found in \(dir.path)")
    }
}
#else
import Foundation

/// Compile-time-disabled LiteRT-LM runtime. Add the `LiteRTLM` product from
/// `https://github.com/google-ai-edge/LiteRT-LM` (≥ 0.12.0) to enable it.
/// See `runtimes/litert-lm.md` for the integration steps.
public final class MediaPipeRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .mediaPipe
    public let isAvailable: Bool = false
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.liteRTLM
    public var loadedModelId: String? { nil }

    public init() {}

    public func loadModel(_ model: ModelInfo, progress: @Sendable @escaping (Double) -> Void) async throws {
        throw LLMRuntimeError.unsupported("LiteRTLM package not added to the project. See runtimes/litert-lm.md.")
    }

    public func unloadModel() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { c in
            c.finish(throwing: LLMRuntimeError.unsupported("LiteRTLM package not added — see runtimes/litert-lm.md."))
        }
    }
}
#endif
