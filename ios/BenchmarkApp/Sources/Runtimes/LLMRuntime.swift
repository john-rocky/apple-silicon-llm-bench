import Foundation

/// Adapter protocol that every benchmarked runtime conforms to.
///
/// One implementation per runtime (MLX, llama.cpp, CoreML-LLM, …).
/// The benchmark runner only sees this surface, never the underlying SDK.
public protocol LLMRuntime: AnyObject, Sendable {
    var kind: RuntimeKind { get }
    var loadedModelId: String? { get async }

    /// Whether this runtime can be invoked at runtime.
    /// `false` means the underlying framework is not present in the build
    /// (e.g. MediaPipe XCFramework not added yet).
    var isAvailable: Bool { get }

    /// Models this runtime is able to load. The UI's model picker filters by this.
    var supportedModels: [ModelInfo] { get }

    /// Download (if needed) and load weights into memory.
    /// `progress` reports 0.0–1.0 for the download phase only; load itself is synchronous from the caller's view.
    func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws

    /// Release model weights and any associated GPU/ANE state.
    func unloadModel() async

    /// Stream generation events for a single prompt. The stream finishes after a `.info` event or on cancellation.
    func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error>

    /// Optional hint, called just before `loadModel`: size the runtime's working context
    /// (e.g. KV pre-allocation) to this task's output budget, so it doesn't reserve a
    /// larger context than the task needs and inflate peak memory. Runtimes that allocate
    /// KV dynamically ignore it (default no-op).
    func prepareContext(maxOutputTokens: Int) async
}

public extension LLMRuntime {
    func prepareContext(maxOutputTokens: Int) async {}
}

public enum GenerationEvent: Sendable {
    case chunk(String)
    case info(GenerationInfo)
}

public struct GenerationInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let promptTime: TimeInterval
    public let generateTime: TimeInterval
    public let stopReason: StopReason

    public enum StopReason: String, Sendable {
        case stop
        case length
        case cancelled
        case error
    }

    public var promptTokensPerSecond: Double {
        promptTime > 0 ? Double(promptTokenCount) / promptTime : 0
    }

    public var tokensPerSecond: Double {
        generateTime > 0 ? Double(generationTokenCount) / generateTime : 0
    }
}

public enum LLMRuntimeError: LocalizedError {
    case unsupported(String)
    case modelNotLoaded
    case modelNotInCatalog(String)
    case loadFailed(String)
    case generationFailed(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let what): return "Unsupported: \(what)"
        case .modelNotLoaded: return "No model is loaded."
        case .modelNotInCatalog(let id): return "Model \(id) is not in this runtime's catalog."
        case .loadFailed(let why): return "Model load failed: \(why)"
        case .generationFailed(let why): return "Generation failed: \(why)"
        case .downloadFailed(let why): return "Model download failed: \(why)"
        }
    }
}
