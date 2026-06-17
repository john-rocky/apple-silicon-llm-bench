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
            id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            displayName: "Qwen 2.5 0.5B (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.5,
            onDiskSizeMB: 300,
            hfRepoId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-0.6B-4bit",
            displayName: "Qwen3-0.6B (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.6,
            onDiskSizeMB: 350,
            hfRepoId: "mlx-community/Qwen3-0.6B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-4B-4bit",
            displayName: "Qwen3-4B (4-bit)",
            quantization: "Q4",
            parameterCountB: 4.0,
            onDiskSizeMB: 2300,
            hfRepoId: "mlx-community/Qwen3-4B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-8B-4bit",
            displayName: "Qwen3-8B (4-bit)",
            quantization: "Q4",
            parameterCountB: 8.0,
            onDiskSizeMB: 4500,
            hfRepoId: "mlx-community/Qwen3-8B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-14B-4bit",
            displayName: "Qwen3-14B (4-bit)",
            quantization: "Q4",
            parameterCountB: 14.0,
            onDiskSizeMB: 8000,
            hfRepoId: "mlx-community/Qwen3-14B-4bit"
        ),
        ModelInfo(
            id: "mlx-community/gemma-4-12b-it-4bit",
            displayName: "Gemma 4 12B (4-bit)",
            quantization: "Q4",
            parameterCountB: 12.0,
            onDiskSizeMB: 7000,
            hfRepoId: "mlx-community/gemma-4-12b-it-4bit"
        ),
        // Comparators for Lu's models. mlx LFM2-350M is v2.0 (no 2.5 on mlx-community yet) —
        // slight version skew vs our LFM2.5 litert; disclosed.
        ModelInfo(
            id: "mlx-community/LFM2-350M-4bit",
            displayName: "LFM2-350M (4-bit)",
            quantization: "Q4",
            parameterCountB: 0.35,
            onDiskSizeMB: 200,
            hfRepoId: "mlx-community/LFM2-350M-4bit"
        ),
        ModelInfo(
            id: "mlx-community/MiniCPM5-1B-4bit",
            displayName: "MiniCPM5-1B (4-bit)",
            quantization: "Q4",
            parameterCountB: 1.0,
            onDiskSizeMB: 600,
            hfRepoId: "mlx-community/MiniCPM5-1B-4bit"
        ),
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
        ModelInfo(
            id: "unsloth/Qwen3-4B-GGUF/Q4_K_M",
            displayName: "Qwen3-4B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 4.0,
            onDiskSizeMB: 2500,
            hfRepoId: "unsloth/Qwen3-4B-GGUF",
            hfFilePatterns: ["Qwen3-4B-Q4_K_M.gguf"],
            primaryFile: "Qwen3-4B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3-8B-GGUF/Q4_K_M",
            displayName: "Qwen3-8B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 8.0,
            onDiskSizeMB: 4900,
            hfRepoId: "unsloth/Qwen3-8B-GGUF",
            hfFilePatterns: ["Qwen3-8B-Q4_K_M.gguf"],
            primaryFile: "Qwen3-8B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/Qwen3-14B-GGUF/Q4_K_M",
            displayName: "Qwen3-14B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 14.0,
            onDiskSizeMB: 9000,
            hfRepoId: "unsloth/Qwen3-14B-GGUF",
            hfFilePatterns: ["Qwen3-14B-Q4_K_M.gguf"],
            primaryFile: "Qwen3-14B-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "unsloth/gemma-4-12B-it-GGUF/Q4_K_M",
            displayName: "Gemma 4 12B Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 12.0,
            onDiskSizeMB: 7000,
            hfRepoId: "unsloth/gemma-4-12B-it-GGUF",
            hfFilePatterns: ["gemma-4-12b-it-Q4_K_M.gguf"],
            primaryFile: "gemma-4-12b-it-Q4_K_M.gguf"
        ),
        ModelInfo(
            id: "LiquidAI/LFM2.5-350M-GGUF/Q4_K_M",
            displayName: "LFM2.5-350M Q4_K_M (GGUF)",
            quantization: "Q4_K_M",
            parameterCountB: 0.35,
            onDiskSizeMB: 230,
            hfRepoId: "LiquidAI/LFM2.5-350M-GGUF",
            hfFilePatterns: ["LFM2.5-350M-Q4_K_M.gguf"],
            primaryFile: "LFM2.5-350M-Q4_K_M.gguf"
        ),
    ]

    /// Models the LiteRT-LM adapter can load.
    ///
    /// Wires `google-ai-edge/LiteRT-LM` ≥ 0.13 (the official Swift API,
    /// `import LiteRTLM`), which reads Google's `.litertlm` bundles. The
    /// catalog is **not** Gemma-only — `litert-community` ships Qwen3
    /// (0.6B/4B), LFM/Liquid, and others in `.litertlm` alongside Gemma; we
    /// target Qwen3-0.6B here so it lines up with the existing Qwen3-0.6B
    /// rows on MLX / CoreML / Core AI. This path supersedes the old MediaPipe
    /// 0.10.x (`.task`) path, which could not read Gemma 4 at all.
    ///
    /// Sizes are the standard (non-web, non-NPU) Metal-GPU variant. Gemma 4
    /// `.litertlm` is QAT INT4 on the decoder with the embedding table kept
    /// in higher precision and memory-mapped (E2B ≈ 0.79 GB decoder + 1.12 GB
    /// mmap'd embeddings ≈ 2.59 GB on disk). Context window 32k. Qwen3-0.6B is
    /// the mixed blockwise-INT4 artifact (gs32 weights, INT8 embeddings).
    public static let liteRTLM: [ModelInfo] = [
        ModelInfo(
            id: "litert-community/gemma-4-E2B-it-litert-lm",
            displayName: "Gemma 4 E2B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 2.0,
            onDiskSizeMB: 2650,
            hfRepoId: "litert-community/gemma-4-E2B-it-litert-lm",
            hfFilePatterns: ["gemma-4-E2B-it.litertlm"],
            primaryFile: "gemma-4-E2B-it.litertlm"
        ),
        // Qwen3-0.6B — the model Lu's team is optimising; the 4-bit `.litertlm`
        // lines up with the existing Qwen3-0.6B rows on MLX / CoreML / Core AI.
        // Fallback if it won't load on GPU: the standard dynamic-INT8
        // `Qwen3-0.6B.litertlm` (614 MB) in the same repo.
        ModelInfo(
            id: "litert-community/Qwen3-0.6B",
            displayName: "Qwen3 0.6B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 0.6,
            onDiskSizeMB: 498,
            hfRepoId: "litert-community/Qwen3-0.6B",
            hfFilePatterns: ["qwen3_0_6b_mixed_int4.litertlm"],
            primaryFile: "qwen3_0_6b_mixed_int4.litertlm"
        ),
        // Lu's focus models (Liquid/LFM2 + MiniCPM) — NOT on litert-community, so these are
        // OUR own .litertlm conversions, side-loaded (no HF download; hfRepoId is a local marker).
        // See MODEL_AVAILABILITY.md / MODEL_MATRIX.md.
        ModelInfo(
            id: "litert-local/lfm2.5-350m",
            displayName: "LFM2.5-350M (.litertlm, local)",
            quantization: "INT4 (ekv1024)",
            parameterCountB: 0.35,
            onDiskSizeMB: 178,
            hfRepoId: "litert-local/LFM2.5-350M",
            hfFilePatterns: ["LFM2.5-350M_int4_ekv1024.litertlm"],
            primaryFile: "LFM2.5-350M_int4_ekv1024.litertlm"
        ),
        ModelInfo(
            id: "litert-local/minicpm5-1b",
            displayName: "MiniCPM5-1B (.litertlm, local)",
            quantization: "INT4 (ekv1024)",
            parameterCountB: 1.0,
            onDiskSizeMB: 532,
            hfRepoId: "litert-local/MiniCPM5-1B",
            hfFilePatterns: ["MiniCPM5-1B_int4_ekv1024.litertlm"],
            primaryFile: "MiniCPM5-1B_int4_ekv1024.litertlm"
        ),
        // Qwen3 4B / 8B — same mixed-INT4 .litertlm line as 0.6B, for a size-scaling
        // curve (0.6B → 4B → 8B). 8B (~4.4 GB) is desktop/Mac-tier; on phones it can
        // exceed the per-app memory ceiling (gemma-3n-style jetsam), so it stays Mac-only.
        ModelInfo(
            id: "litert-community/Qwen3-4B",
            displayName: "Qwen3 4B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 4.0,
            onDiskSizeMB: 2300,
            hfRepoId: "litert-community/Qwen3-4B",
            hfFilePatterns: ["qwen3_4b_mixed_int4.litertlm"],
            primaryFile: "qwen3_4b_mixed_int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/Qwen3-8B",
            displayName: "Qwen3 8B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 8.0,
            onDiskSizeMB: 4400,
            hfRepoId: "litert-community/Qwen3-8B",
            hfFilePatterns: ["qwen3_8b_mixed_int4.litertlm"],
            primaryFile: "qwen3_8b_mixed_int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/Qwen3-14B",
            displayName: "Qwen3 14B (.litertlm)",
            quantization: "INT4 (mixed, blockwise gs32)",
            parameterCountB: 14.0,
            onDiskSizeMB: 8000,
            hfRepoId: "litert-community/Qwen3-14B",
            hfFilePatterns: ["qwen3_14b_mixed_int4.litertlm"],
            primaryFile: "qwen3_14b_mixed_int4.litertlm"
        ),
        ModelInfo(
            id: "litert-community/gemma-4-12B-it-litert-lm",
            displayName: "Gemma 4 12B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 12.0,
            onDiskSizeMB: 7000,
            hfRepoId: "litert-community/gemma-4-12B-it-litert-lm",
            hfFilePatterns: ["gemma-4-12B-it.litertlm"],
            primaryFile: "gemma-4-12B-it.litertlm"
        ),
        ModelInfo(
            id: "litert-community/gemma-4-E4B-it-litert-lm",
            displayName: "Gemma 4 E4B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 4.0,
            onDiskSizeMB: 3750,
            hfRepoId: "litert-community/gemma-4-E4B-it-litert-lm",
            hfFilePatterns: ["gemma-4-E4B-it.litertlm"],
            primaryFile: "gemma-4-E4B-it.litertlm"
        ),
        ModelInfo(
            id: "google/gemma-3n-E2B-it-litert-lm",
            displayName: "Gemma 3n E2B (.litertlm)",
            quantization: "INT4 (QAT)",
            parameterCountB: 2.0,
            onDiskSizeMB: 3100,
            hfRepoId: "google/gemma-3n-E2B-it-litert-lm",
            hfFilePatterns: ["*.litertlm"],
            primaryFile: ""
        ),
    ]

    /// Back-compat alias. The adapter and earlier call sites refer to this
    /// runtime's catalog as `mediaPipe`; LiteRT-LM is the current name.
    public static let mediaPipe: [ModelInfo] = liteRTLM

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
        ModelInfo(
            id: "coreml-llm/qwen3-0.6b",
            displayName: "Qwen3-0.6B (CoreML, ANE)",
            quantization: "INT8 palettized",
            parameterCountB: 0.6,
            onDiskSizeMB: 900,
            hfRepoId: "mlboydaisuke/qwen3-0.6b-coreml"
        ),
    ]

    /// Models the Apple Foundation Models adapter can run.
    ///
    /// Apple FM is a single, pre-installed on-device model — no HF download,
    /// no model picking. The catalog still carries one entry so the harness
    /// can attach a stable id / display name to the runs. Size / quant are
    /// best-guess (Apple has not published a parameter count for the GA
    /// model; community estimates put it at ~3B params with 2-bit / 4-bit
    /// adapters).
    public static let appleFM: [ModelInfo] = [
        ModelInfo(
            id: "apple-fm/default",
            displayName: "Apple Foundation Model (default, on-device)",
            quantization: "Apple-quant (~2-4 bit, adapters)",
            parameterCountB: 3.0,
            onDiskSizeMB: nil,
            hfRepoId: ""
        ),
    ]

    /// Models the Apple **Core AI** adapter can run — the same Qwen3-0.6B
    /// `.aimodel` bundle exported by the official `coreai.llm.export` pipeline,
    /// exposed twice so the harness can benchmark both compute paths:
    /// the Neural Engine (`static-shape`) and the GPU (`coreai-pipelined`).
    ///
    /// The bundle is side-loaded into `Documents/CoreAIModels/qwen3_0_6b_ios/`
    /// (the `.aimodel` is ~434 MB and not published to HF), so `hfRepoId` is
    /// empty. Requires iOS 27 + the `coreai-models` Swift package.
    public static let coreAI: [ModelInfo] = [
        ModelInfo(
            id: "core-ai/qwen3-0.6b-ane",
            displayName: "Qwen3-0.6B (Core AI, ANE)",
            quantization: "mixed 4/8-bit palettized",
            parameterCountB: 0.6,
            onDiskSizeMB: 434,
            hfRepoId: ""
        ),
        ModelInfo(
            id: "core-ai/qwen3-0.6b-gpu",
            displayName: "Qwen3-0.6B (Core AI, GPU)",
            quantization: "INT4 (dynamic)",
            parameterCountB: 0.6,
            onDiskSizeMB: 327,
            hfRepoId: ""
        ),
    ]

    /// Default model picked when the app first launches.
    public static let defaultModel: ModelInfo = mlx[0]
}
