import Foundation

/// A model packaged for a specific runtime.
///
/// Different runtimes need different artifact formats — GGUF for llama.cpp,
/// MLX safetensors for MLX, `.task`/`.litertlm` for MediaPipe, `.pte` for
/// ExecuTorch, multi-`.mlmodelc` bundle for ANEMLL, single `.mlpackage` for
/// CoreML LLM. Each adapter publishes its own `supportedModels` list.
public struct ModelInfo: Codable, Sendable, Hashable {
    public let id: String
    public let displayName: String
    public let quantization: String
    public let parameterCountB: Double?
    public let onDiskSizeMB: Double?

    /// HuggingFace repo (`namespace/name`) the artifact lives in.
    public let hfRepoId: String
    /// Glob patterns to match when downloading. `["*"]` = whole snapshot.
    public let hfFilePatterns: [String]
    /// Path inside the downloaded snapshot, relative to the snapshot root.
    /// Empty string = the snapshot root itself.
    public let primaryFile: String

    public init(
        id: String,
        displayName: String,
        quantization: String,
        parameterCountB: Double? = nil,
        onDiskSizeMB: Double? = nil,
        hfRepoId: String,
        hfFilePatterns: [String] = ["*"],
        primaryFile: String = ""
    ) {
        self.id = id
        self.displayName = displayName
        self.quantization = quantization
        self.parameterCountB = parameterCountB
        self.onDiskSizeMB = onDiskSizeMB
        self.hfRepoId = hfRepoId
        self.hfFilePatterns = hfFilePatterns
        self.primaryFile = primaryFile
    }
}

/// Aggregate view of all models the app can run, grouped by runtime.
public enum ModelCatalog {
    /// Models the MLX Swift adapter can load. Curated 2026-05 — all
    /// entries verified present on `huggingface.co/mlx-community`.
    /// Sizes are approximate (verified via HF API or by download).
    public static let mlx: [ModelInfo] = [
        // --- Tiny — fits any device, including iPhone with headroom ---
        ModelInfo(
            id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            displayName: "Qwen 3.5 0.8B (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.8,
            onDiskSizeMB: 500,
            hfRepoId: "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-e2b-it-4bit",
            displayName: "Gemma 4 E2B (4-bit)",
            quantization: "Q4",
            parameterCountB: 2.0,
            onDiskSizeMB: 1330,
            hfRepoId: "mlx-community/gemma-4-e2b-it-4bit"
        ),

        // --- Small — fits 16 GB Mac and iPhone Pro models ---
        ModelInfo(
            id: "mlx-community/Qwen3.5-2B-MLX-4bit",
            displayName: "Qwen 3.5 2B (4-bit)",
            quantization: "Q4",
            parameterCountB: 2.0,
            onDiskSizeMB: 1500,
            hfRepoId: "mlx-community/Qwen3.5-2B-MLX-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-e4b-it-4bit",
            displayName: "Gemma 4 E4B (4-bit)",
            quantization: "Q4",
            parameterCountB: 4.0,
            onDiskSizeMB: 3000,
            hfRepoId: "mlx-community/gemma-4-e4b-it-4bit"
        ),

        // --- Medium — Mac-class (M-series with ≥16 GB) ---
        ModelInfo(
            id: "mlx-community/Qwen3.5-9B-MLX-4bit",
            displayName: "Qwen 3.5 9B (4-bit)",
            quantization: "Q4",
            parameterCountB: 9.0,
            onDiskSizeMB: 5500,
            hfRepoId: "mlx-community/Qwen3.5-9B-MLX-4bit"
        ),

        // --- Large / MoE — workstation-class Mac ---
        ModelInfo(
            id: "mlx-community/gemma-4-26b-a4b-it-4bit",
            displayName: "Gemma 4 26B-A4B (4-bit, MoE)",
            quantization: "Q4",
            parameterCountB: 26.0,
            onDiskSizeMB: 14_500,
            hfRepoId: "mlx-community/gemma-4-26b-a4b-it-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3.5-27B-4bit",
            displayName: "Qwen 3.5 27B (4-bit)",
            quantization: "Q4",
            parameterCountB: 27.0,
            onDiskSizeMB: 15_500,
            hfRepoId: "mlx-community/Qwen3.5-27B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3.5-35B-A3B-4bit",
            displayName: "Qwen 3.5 35B-A3B (4-bit, MoE)",
            quantization: "Q4",
            parameterCountB: 35.0,
            onDiskSizeMB: 19_500,
            hfRepoId: "mlx-community/Qwen3.5-35B-A3B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-31b-it-4bit",
            displayName: "Gemma 4 31B (4-bit)",
            quantization: "Q4",
            parameterCountB: 31.0,
            onDiskSizeMB: 17_500,
            hfRepoId: "mlx-community/gemma-4-31b-it-4bit"
        ),
    ]

    /// Models the llama.cpp adapter can load.
    /// One `.gguf` file per entry, downloaded from the listed HF repo.
    public static let llamaCpp: [ModelInfo] = [
        ModelInfo(
            id: "unsloth/gemma-4-E2B-it-GGUF/Q4_K_M",
            displayName: "Gemma 4 E2B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 2.0,
            onDiskSizeMB: 1700,
            hfRepoId: "unsloth/gemma-4-E2B-it-GGUF",
            hfFilePatterns: ["gemma-4-E2B-it-Q4_K_M.gguf"],
            primaryFile: "gemma-4-E2B-it-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/gemma-4-E4B-it-GGUF/Q4_K_M",
            displayName: "Gemma 4 E4B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 4.0,
            onDiskSizeMB: 3300,
            hfRepoId: "unsloth/gemma-4-E4B-it-GGUF",
            hfFilePatterns: ["gemma-4-E4B-it-Q4_K_M.gguf"],
            primaryFile: "gemma-4-E4B-it-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "bartowski/Llama-3.2-1B-Instruct-GGUF/Q4_K_M",
            displayName: "Llama 3.2 1B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 1.0,
            onDiskSizeMB: 800,
            hfRepoId: "bartowski/Llama-3.2-1B-Instruct-GGUF",
            hfFilePatterns: ["Llama-3.2-1B-Instruct-Q4_K_M.gguf"],
            primaryFile: "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF/Q4_K_M",
            displayName: "Qwen 2.5 0.5B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 0.5,
            onDiskSizeMB: 380,
            hfRepoId: "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            hfFilePatterns: ["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"],
            primaryFile: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "bartowski/Qwen_Qwen3.5-0.8B-GGUF/Q4_K_M",
            displayName: "Qwen 3.5 0.8B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 0.8,
            onDiskSizeMB: 560,
            hfRepoId: "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
            hfFilePatterns: ["Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"],
            primaryFile: "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3.5-2B-GGUF/Q4_K_M",
            displayName: "Qwen 3.5 2B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 2.0,
            onDiskSizeMB: 1300,
            hfRepoId: "unsloth/Qwen3.5-2B-GGUF",
            hfFilePatterns: ["Qwen3.5-2B-Q4_K_M.gguf"],
            primaryFile: "Qwen3.5-2B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3.5-9B-GGUF/Q4_K_M",
            displayName: "Qwen 3.5 9B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 9.0,
            onDiskSizeMB: 5500,
            hfRepoId: "unsloth/Qwen3.5-9B-GGUF",
            hfFilePatterns: ["Qwen3.5-9B-Q4_K_M.gguf"],
            primaryFile: "Qwen3.5-9B-Q4_K_M.gguf"
        ),
    ]

    /// Models the MediaPipe / LiteRT-LM adapter can load.
    /// Note: Gemma 4 ships only as `.litertlm` which the deprecated
    /// MediaPipeTasksGenAI 0.10.x runtime cannot read; Gemma 3n is the
    /// newest model that loads.
    public static let mediaPipe: [ModelInfo] = [
        ModelInfo(
            id: "litert-community/Gemma3-1B-IT/task",
            displayName: "Gemma 3 1B IT (.task)",
            quantization: "INT4",
            parameterCountB: 1.0,
            onDiskSizeMB: 555,
            hfRepoId: "litert-community/Gemma3-1B-IT",
            hfFilePatterns: ["*.task"],
            primaryFile: "gemma3-1b-it-int4.task"
        ),
        ModelInfo(
            id: "google/gemma-3n-E2B-it-litert-preview/task",
            displayName: "Gemma 3n E2B (.task, preview)",
            quantization: "INT4",
            parameterCountB: 2.0,
            onDiskSizeMB: 1500,
            hfRepoId: "google/gemma-3n-E2B-it-litert-preview",
            hfFilePatterns: ["*.task"],
            primaryFile: "gemma-3n-E2B-it-int4.task"
        ),
    ]

    /// Models the ExecuTorch adapter can load.
    /// Each repo ships a `.pte` plus a `tokenizer.model` (sentencepiece).
    /// No official Gemma 4 .pte exists yet (May 2026).
    public static let executorch: [ModelInfo] = [
        ModelInfo(
            id: "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
            displayName: "Llama 3.2 1B SpinQuant INT4 (.pte)",
            quantization: "INT4 (SpinQuant)",
            parameterCountB: 1.0,
            onDiskSizeMB: 1140,
            hfRepoId: "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
            hfFilePatterns: ["*.pte", "tokenizer.model"],
            primaryFile: "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
        ),
    ]

    /// Models the ANEMLL adapter can load.
    /// Each repo ships a multi-file bundle (embeddings + FFN chunks + lm_head + meta.yaml + tokenizer files).
    public static let anemll: [ModelInfo] = [
        ModelInfo(
            id: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5",
            displayName: "Gemma 3 1B IT (ANEMLL ANE)",
            quantization: "Q4 (ANE-tuned)",
            parameterCountB: 1.0,
            onDiskSizeMB: 1400,
            hfRepoId: "anemll/anemll-google-gemma-3-1b-it-ctx4096_0.3.5"
        ),
        ModelInfo(
            id: "anemll/anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5",
            displayName: "Llama 3.2 1B Instruct (ANEMLL ANE)",
            quantization: "Q4 (ANE-tuned)",
            parameterCountB: 1.0,
            onDiskSizeMB: 1500,
            hfRepoId: "anemll/anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5"
        ),
        ModelInfo(
            id: "anemll/anemll-google-gemma-3-270m-it-ctx4096_0.3.5",
            displayName: "Gemma 3 270M IT (ANEMLL ANE)",
            quantization: "Q4 (ANE-tuned)",
            parameterCountB: 0.27,
            onDiskSizeMB: 400,
            hfRepoId: "anemll/anemll-google-gemma-3-270m-it-ctx4096_0.3.5"
        ),
    ]

    /// Models the CoreML LLM adapter (john-rocky/CoreML-LLM) can load.
    /// Each id maps to a registered `CoreMLLLM.ModelDownloader.ModelInfo`
    /// that downloads and ANE-compiles a chunked `.mlmodelc` bundle from
    /// the `mlboydaisuke/*-coreml` HF namespace on first use.
    public static let coreML: [ModelInfo] = [
        ModelInfo(
            id: "coreml-llm/gemma4-e2b",
            displayName: "Gemma 4 E2B (CoreML, ANE)",
            quantization: "INT4 palettized",
            parameterCountB: 2.0,
            onDiskSizeMB: 5400,
            hfRepoId: "mlboydaisuke/gemma-4-E2B-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/gemma4-e4b",
            displayName: "Gemma 4 E4B (CoreML, ANE)",
            quantization: "INT4 palettized",
            parameterCountB: 4.0,
            onDiskSizeMB: 5500,
            hfRepoId: "mlboydaisuke/gemma-4-E4B-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/qwen3.5-0.8b",
            displayName: "Qwen 3.5 0.8B (CoreML, ANE)",
            quantization: "INT8",
            parameterCountB: 0.8,
            onDiskSizeMB: 1200,
            hfRepoId: "mlboydaisuke/qwen3.5-0.8B-CoreML"
        ),
        ModelInfo(
            id: "coreml-llm/qwen3.5-2b",
            displayName: "Qwen 3.5 2B (CoreML, ANE)",
            quantization: "INT8",
            parameterCountB: 2.0,
            onDiskSizeMB: 2800,
            hfRepoId: "mlboydaisuke/qwen3.5-2B-CoreML"
        ),
        ModelInfo(
            id: "coreml-llm/lfm2.5-350m",
            displayName: "LFM 2.5 350M (CoreML, ANE)",
            quantization: "INT8",
            parameterCountB: 0.35,
            onDiskSizeMB: 810,
            hfRepoId: "mlboydaisuke/lfm2.5-350m-coreml"
        ),
        ModelInfo(
            id: "coreml-llm/qwen2.5-0.5b",
            displayName: "Qwen 2.5 0.5B (CoreML, text)",
            quantization: "FP16",
            parameterCountB: 0.5,
            onDiskSizeMB: 309,
            hfRepoId: "mlboydaisuke/qwen2.5-0.5b-coreml"
        ),
    ]

    /// Default model picked when the app first launches.
    public static let defaultModel: ModelInfo = mlx[0]
}
