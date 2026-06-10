import Foundation
import CoreVideo
import CoreImage
import MLXLLM
import MLXVLM
import MLXLMCommon
import HuggingFace
import Tokenizers

/// MLX Swift **vision-language** adapter — the GPU path of the camera benchmark.
///
/// Loads a VLM (Qwen3-VL 2B) from the Hugging Face Hub and runs image+text
/// inference on the Metal GPU. `import MLXVLM` links the VLM model registry
/// (`qwen3_vl` → the native Swift `Qwen3VL` class), so the same
/// `loadModelContainer` / `ModelContainer.generate` path the text adapter uses
/// also builds and drives the VLM — we only add the camera frame to `UserInput`.
public actor MLXVLMRuntime: VLMRuntime {
    public let kind: RuntimeKind = .mlxSwift
    public let placement: ComputePlacement = .gpu
    public let isAvailable: Bool = true
    public nonisolated let supportedModels: [ModelInfo] = VLMModelCatalog.mlx

    public private(set) var loadedModelId: String?
    private var container: ModelContainer?

    public init() {}

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        let configuration = ModelConfiguration(id: model.id)
        print("[MLXVLM] load starting: \(model.id)")
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            // Same bridge-based loader as the text MLX adapter; with MLXVLM
            // linked, the model-type registry resolves `qwen3_vl` to the VLM
            // container. (If a build pins an MLX version that splits the
            // factories, swap this for `VLMModelFactory.shared.loadContainer`.)
            let container = try await loadModelContainer(
                from: HubDownloaderBridge(client: HubClient.default),
                using: HFTokenizerLoaderBridge(),
                configuration: configuration,
                progressHandler: { p in
                    let pct = Int(p.fractionCompleted * 100)
                    print(String(format: "[MLXVLM] download progress: %d%% (%lld/%lld bytes)",
                                  pct, p.completedUnitCount, p.totalUnitCount))
                    progress(p.fractionCompleted)
                }
            )
            self.container = container
            self.loadedModelId = model.id
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "[MLXVLM] load complete in %.2fs", dt))
        } catch {
            print("[MLXVLM] load FAILED: \(error.localizedDescription)")
            throw LLMRuntimeError.loadFailed(error.localizedDescription)
        }
    }

    public func unloadModel() async {
        container = nil
        loadedModelId = nil
    }

    public nonisolated func describe(
        pixelBuffer: CVPixelBuffer,
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        // Snapshot the frame into a CIImage up front so the camera can recycle
        // the pixel buffer immediately.
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let container = await self.container else {
                        throw LLMRuntimeError.modelNotLoaded
                    }

                    let mlxParameters = GenerateParameters(
                        maxTokens: parameters.maxTokens,
                        temperature: parameters.temperature,
                        topP: parameters.topP
                    )

                    let input = UserInput(
                        prompt: prompt,
                        images: [.ciImage(ciImage)]
                    )
                    let lmInput = try await container.prepare(input: input)
                    let stream = try await container.generate(
                        input: lmInput,
                        parameters: mlxParameters
                    )

                    for await event in stream {
                        try Task.checkCancellation()
                        switch event {
                        case .chunk(let text):
                            continuation.yield(.chunk(text))
                        case .info(let info):
                            continuation.yield(.info(GenerationInfo(
                                promptTokenCount: info.promptTokenCount,
                                generationTokenCount: info.generationTokenCount,
                                promptTime: info.promptTime,
                                generateTime: info.generateTime,
                                stopReason: Self.translate(info.stopReason)
                            )))
                        case .toolCall:
                            break
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.info(GenerationInfo(
                        promptTokenCount: 0,
                        generationTokenCount: 0,
                        promptTime: 0,
                        generateTime: 0,
                        stopReason: .cancelled
                    )))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func aneResidencyPercent() async -> Double? { nil }   // GPU path

    private static func translate(_ reason: GenerateStopReason) -> GenerationInfo.StopReason {
        switch reason {
        case .stop: return .stop
        case .length: return .length
        case .cancelled: return .cancelled
        @unknown default: return .stop
        }
    }
}
