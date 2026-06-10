import Foundation
import CoreVideo
import CoreImage
import CoreML
#if canImport(CoreMLLLM)
import CoreMLLLM
#endif
#if canImport(Tokenizers)
import Tokenizers
#endif

/// CoreML **vision-language** adapter — the Apple Neural Engine path of the
/// camera benchmark.
///
/// Drives `john-rocky/CoreML-LLM`'s real Qwen3-VL 2B pipeline (public API since
/// v1.9.0), the converted ANE model the `mlboydaisuke/qwen3-vl-2b-coreml`
/// bundle ships:
///
///   • `Qwen3VL2BVisionEncoder` — `CGImage` → `Qwen3VL2BVisionFeatures`
///     (196×2048 pooled tokens + 3 DeepStack tensors, SigLIP + merger, ANE).
///     Preprocessing (448×448 resize, normalize, patchify) is internal.
///   • `Qwen3VL2BGenerator` — chunked INT8 decoder on `.cpuAndNeuralEngine`,
///     consuming the vision features + spliced `<|image_pad|>` (151655) tokens
///     with DeepStack injection.
///
/// Per frame: encode the camera image once, then decode a short caption. The
/// vision encoder is the conv-heavy per-frame ANE workload; its residency is
/// reported via `MLComputePlan` (see `ANEResidency`).
///
/// Requires iOS 18+. Falls back to a clear error if the `CoreMLLLM` product or
/// the converted bundle isn't present.
@available(iOS 18.0, macOS 15.0, *)
public final class CoreMLVLMRuntime: VLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .coreMLLLM
    public let placement: ComputePlacement = .ane
    #if canImport(CoreMLLLM)
    public let isAvailable: Bool = true
    #else
    public let isAvailable: Bool = false
    #endif
    public let supportedModels: [ModelInfo] = VLMModelCatalog.coreML

    /// Qwen3-VL special tokens used to build the vision chat template.
    private enum Tok {
        static let imEnd: Int32 = 151645      // <|im_end|>
        static let endOfText: Int32 = 151643  // <|endoftext|>
        static let visionStart = "<|vision_start|>"
        static let visionEnd = "<|vision_end|>"
        static let imagePad: Int32 = 151655   // <|image_pad|>
    }

    nonisolated(unsafe) private var _loadedModelId: String?
    public var loadedModelId: String? { _loadedModelId }

    #if canImport(CoreMLLLM)
    nonisolated(unsafe) private var generator: Qwen3VL2BGenerator?
    nonisolated(unsafe) private var encoder: Qwen3VL2BVisionEncoder?
    #endif
    nonisolated(unsafe) private var visionEncoderURL: URL?
    #if canImport(Tokenizers)
    nonisolated(unsafe) private var tokenizer: (any Tokenizer)?
    #endif

    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    public init() {}

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }

        #if canImport(CoreMLLLM)
        // 1. Resolve / download the CoreML-LLM bundle (text+vision, ANE).
        let info = ModelDownloader.ModelInfo.qwen3vl_2b
        let downloader = ModelDownloader.shared
        let chunksURL: URL
        if let existing = downloader.localModelURL(for: info) {
            chunksURL = existing
            progress(0.6)
        } else {
            progress(0.1)
            _ = try await downloader.download(info)
            guard let resolved = downloader.localModelURL(for: info) else {
                throw LLMRuntimeError.loadFailed(
                    "Downloaded \(info.id) but its chunk bundle wasn't found on disk.")
            }
            chunksURL = resolved
        }
        // localModelURL returns the chunks subdir; the generator wants the parent
        // model folder (which also holds qwen3_vl_2b_vision/).
        let modelFolder = chunksURL.deletingLastPathComponent()

        // 2. Load the decoder.
        let gen = Qwen3VL2BGenerator(cfg: .default)
        gen.modelFolderOverride = modelFolder
        gen.setComputeUnits(.cpuAndNeuralEngine)
        progress(0.75)
        try gen.load()
        self.generator = gen

        // 3. Load the vision encoder, if the bundle ships it.
        if let visionURL = Qwen3VL2BVisionEncoder.resolveModel(folder: modelFolder) {
            let enc = Qwen3VL2BVisionEncoder(cfg: .default)
            try enc.load(modelURL: visionURL)
            self.encoder = enc
            self.visionEncoderURL = visionURL
        } else {
            throw LLMRuntimeError.loadFailed(
                "Vision encoder (qwen3_vl_2b_vision/vision.mlmodelc) missing from \(model.hfRepoId).")
        }
        progress(0.9)

        #if canImport(Tokenizers)
        self.tokenizer = try? await AutoTokenizer.from(pretrained: "Qwen/Qwen3-VL-2B-Instruct")
        #endif

        self._loadedModelId = model.id
        progress(1)
        #else
        throw LLMRuntimeError.unsupported("CoreMLLLM SPM product not present.")
        #endif
    }

    public func unloadModel() async {
        #if canImport(CoreMLLLM)
        generator = nil
        encoder = nil
        #endif
        visionEncoderURL = nil
        #if canImport(Tokenizers)
        tokenizer = nil
        #endif
        _loadedModelId = nil
    }

    public func describe(
        pixelBuffer: CVPixelBuffer,
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        // Snapshot the frame to a CGImage so the camera can recycle the buffer.
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let cgImage = ciContext.createCGImage(ci, from: ci.extent)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runDescribe(
                        cgImage: cgImage, prompt: prompt,
                        parameters: parameters, continuation: continuation
                    )
                } catch is CancellationError {
                    continuation.yield(.info(GenerationInfo(
                        promptTokenCount: 0, generationTokenCount: 0,
                        promptTime: 0, generateTime: 0, stopReason: .cancelled)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runDescribe(
        cgImage: CGImage?,
        prompt: String,
        parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async throws {
        #if canImport(CoreMLLLM)
        guard let generator, let encoder else { throw LLMRuntimeError.modelNotLoaded }
        guard let cgImage else { throw LLMRuntimeError.generationFailed("frame → CGImage failed") }

        let prefillStart = CFAbsoluteTimeGetCurrent()

        // 1. Vision encode on the ANE.
        let features = try await encoder.encode(cgImage)

        // 2. Build the Qwen3-VL vision chat template with `features.count`
        //    image-pad tokens (the generator maps each to a vision row).
        let inputIds = try buildVisionInputIds(prompt: prompt, imageTokens: features.count)

        // 3. Greedy decode, streaming the caption.
        final class StreamState: @unchecked Sendable {
            var firstTokenAt: CFAbsoluteTime?
            var accumIds: [Int] = []
            var emitted = ""
            var count = 0
        }
        let state = StreamState()
        var eos: Set<Int32> = [Tok.imEnd, Tok.endOfText]
        #if canImport(Tokenizers)
        if let eid = tokenizer?.eosTokenId { eos.insert(Int32(eid)) }
        #endif

        _ = try await generator.generate(
            inputIds: inputIds,
            maxNewTokens: parameters.maxTokens,
            eosTokenIds: eos,
            visionFeatures: features,
            imagePadTokenId: Tok.imagePad,
            onToken: { tokenId in
                if state.firstTokenAt == nil { state.firstTokenAt = CFAbsoluteTimeGetCurrent() }
                state.count += 1
                if eos.contains(tokenId) { return }
                state.accumIds.append(Int(tokenId))
                let text = self.decode(state.accumIds)
                if text.count > state.emitted.count {
                    let delta = String(text[state.emitted.endIndex...])
                    state.emitted = text
                    continuation.yield(.chunk(delta))
                }
            }
        )

        let end = CFAbsoluteTimeGetCurrent()
        let promptTime = (state.firstTokenAt ?? end) - prefillStart
        let generateTime = max(end - (state.firstTokenAt ?? prefillStart), 0.001)
        continuation.yield(.info(GenerationInfo(
            promptTokenCount: inputIds.count,
            generationTokenCount: state.count,
            promptTime: promptTime,
            generateTime: generateTime,
            stopReason: state.count >= parameters.maxTokens ? .length : .stop)))
        continuation.finish()
        #else
        throw LLMRuntimeError.unsupported("CoreMLLLM SPM product not present.")
        #endif
    }

    public func aneResidencyPercent() async -> Double? {
        guard #available(iOS 17.4, macOS 14.4, *), let url = visionEncoderURL else { return nil }
        // Vision encoder = the per-frame ANE workload.
        return await ANEResidency.percent(ofCompiledModelAt: url, computeUnits: .cpuAndNeuralEngine)
    }

    // MARK: - Helpers

    /// Build `<|im_start|>user\n<|vision_start|> [image_pad×N] <|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n`.
    /// Encodes the wrapper text with special-token handling, then splices N
    /// `<|image_pad|>` ids so the generator has exactly one per vision row.
    private func buildVisionInputIds(prompt: String, imageTokens: Int) throws -> [Int32] {
        #if canImport(Tokenizers)
        guard let tok = tokenizer else {
            throw LLMRuntimeError.generationFailed("tokenizer unavailable for CoreML VLM")
        }
        let head = "<|im_start|>user\n\(Tok.visionStart)"
        let tail = "\(Tok.visionEnd)\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        var ids = tok.encode(text: head).map { Int32($0) }
        ids += Array(repeating: Tok.imagePad, count: max(1, imageTokens))
        ids += tok.encode(text: tail).map { Int32($0) }
        return ids
        #else
        throw LLMRuntimeError.generationFailed("tokenizer unavailable for CoreML VLM")
        #endif
    }

    private func decode(_ ids: [Int]) -> String {
        #if canImport(Tokenizers)
        if let tok = tokenizer { return tok.decode(tokens: ids) }
        #endif
        return ""
    }
}
