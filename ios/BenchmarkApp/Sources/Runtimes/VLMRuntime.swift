import Foundation
import CoreVideo

/// Adapter protocol for a vision-language model (VLM) running a *live camera*
/// workload. The benchmark only ever sees this surface, never the underlying
/// SDK (MLX, CoreML, …).
///
/// This is deliberately separate from `LLMRuntime`: the text harness streams
/// tokens for one prompt, while the camera harness feeds a *frame + prompt* per
/// inference and runs hundreds of inferences back-to-back for 10 minutes. The
/// headline metric is sustained **frames/sec**, not decode tok/s.
///
/// The two paths we compare on the same iPhone:
///   • `.gpu` — MLX (Metal) running Qwen3-VL 2B on the GPU.
///   • `.ane` — CoreML running the model on the Apple Neural Engine.
public protocol VLMRuntime: AnyObject, Sendable {
    var kind: RuntimeKind { get }

    /// Which compute unit this runtime drives. This is the axis the
    /// "GPU throttles / ANE holds" chart is split on.
    var placement: ComputePlacement { get }

    var loadedModelId: String? { get async }

    /// `false` when the underlying framework / converted model is not present
    /// in this build (the UI greys the runtime out and explains why).
    var isAvailable: Bool { get }

    var supportedModels: [ModelInfo] { get }

    func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws

    func unloadModel() async

    /// Run a single image + prompt inference, streaming the decoded caption as
    /// `.chunk` events and finishing with one `.info` (token counts + timing).
    /// The runner times the gap to the first `.chunk` as that inference's TTFT.
    func describe(
        pixelBuffer: CVPixelBuffer,
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error>

    /// Static estimate of the fraction (0–100) of model ops the Core ML runtime
    /// will schedule on the Neural Engine, computed from `MLComputePlan`.
    /// `nil` for GPU runtimes (and on OSes without the API). See `ANEResidency`.
    func aneResidencyPercent() async -> Double?
}

public enum ComputePlacement: String, Sendable, Codable {
    case gpu
    case ane

    public var label: String {
        switch self {
        case .gpu: return "GPU"
        case .ane: return "ANE"
        }
    }
}

public extension VLMRuntime {
    func aneResidencyPercent() async -> Double? { nil }
}

/// Catalog of vision-language models, grouped by the runtime that loads them.
///
/// Both paths target the same logical model — **Qwen3-VL 2B Instruct** — so the
/// comparison is GPU-vs-ANE on identical weights, not model-vs-model.
public enum VLMModelCatalog {
    /// MLX-format weights loaded by `MLXVLMRuntime` (GPU / Metal). These exist
    /// on `huggingface.co/mlx-community`, converted with mlx-vlm; the Swift
    /// `MLXVLM` library registers the `qwen3_vl` architecture so they load
    /// natively (no Python bridge).
    public static let mlx: [ModelInfo] = [
        ModelInfo(
            id: "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            displayName: "Qwen3-VL 2B Instruct (MLX, 4-bit)",
            quantization: "Q4",
            parameterCountB: 2.0,
            onDiskSizeMB: 1780,
            hfRepoId: "mlx-community/Qwen3-VL-2B-Instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-VL-2B-Instruct-8bit",
            displayName: "Qwen3-VL 2B Instruct (MLX, 8-bit)",
            quantization: "Q8",
            parameterCountB: 2.0,
            onDiskSizeMB: 2400,
            hfRepoId: "mlx-community/Qwen3-VL-2B-Instruct-8bit"
        ),
    ]

    /// CoreML bundle loaded by `CoreMLVLMRuntime` (ANE). This is the published
    /// `mlboydaisuke/qwen3-vl-2b-coreml` model — a SigLIP vision encoder +
    /// chunked INT8 decoder with DeepStack injection, converted by
    /// `john-rocky/CoreML-LLM` (`conversion/build_qwen3_vl_2b_*.py`). The
    /// download is driven by `CoreMLLLM.ModelDownloader.qwen3vl_2b`, so
    /// `hfRepoId` here is for display only.
    public static let coreML: [ModelInfo] = [
        ModelInfo(
            id: "coreml-vlm/qwen3-vl-2b",
            displayName: "Qwen3-VL 2B Instruct (CoreML, ANE)",
            quantization: "INT8 + fp16 vision",
            parameterCountB: 2.0,
            onDiskSizeMB: 4700,
            hfRepoId: "mlboydaisuke/qwen3-vl-2b-coreml"
        ),
    ]
}
