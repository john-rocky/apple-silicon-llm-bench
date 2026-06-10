import Foundation

/// The live-camera VLM workload: run a short caption/count on the current frame,
/// back-to-back, for `durationSeconds`. The point is *sustained* operation —
/// 10 minutes — so thermal throttling has time to bite and the GPU-vs-ANE
/// divergence shows up.
public struct CameraVLMTask: Sendable {
    public let id = "camera-vlm"
    public let title = "Live camera (VLM)"
    public let summary = "Continuous camera captioning. Measures sustained FPS, TTFT, throttle, power, peak temp."

    /// Kept terse on purpose: a short instruction + a small token budget keeps
    /// each inference fast, so FPS reflects the model's frame throughput rather
    /// than how long a caption we asked for.
    public let prompt: String
    public let parameters: GenerationParameters

    /// Total wall-clock the session runs for. Headline = 10 minutes (600 s).
    public let durationSeconds: TimeInterval

    public init(
        prompt: String = "Briefly describe the scene and count the people. One sentence.",
        maxTokens: Int = 48,
        durationSeconds: TimeInterval = 600
    ) {
        self.prompt = prompt
        self.parameters = GenerationParameters(maxTokens: maxTokens, temperature: 0.0, topP: 1.0)
        self.durationSeconds = durationSeconds
    }

    /// Short preset for the 20–30 s demo clip.
    public static let clip = CameraVLMTask(durationSeconds: 30)
    /// Headline preset: 10-minute sustained session.
    public static let sustained = CameraVLMTask(durationSeconds: 600)
}
