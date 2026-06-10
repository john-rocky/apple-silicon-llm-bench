import Foundation

public struct SustainedGenerationTask: BenchmarkTask {
    public let id = "sustained-generation"
    public let title = "Sustained generation"
    public let summary = "512-token output. Measures thermal stability and decode degradation."

    public let prompt = "Write a detailed explanation of how local LLM inference works on mobile devices."

    public let parameters = GenerationParameters(maxTokens: 512, temperature: 0.7, topP: 0.9)

    public init() {}
}

/// Sustained-load task for energy / battery-efficiency measurement.
///
/// iOS only exposes battery level in 1% steps, so a short reply is far too
/// small to register a delta. This task asks the runner to keep generating for
/// `sustainSeconds` (re-prompting whenever a runtime hits EOS or its per-call
/// cap), which drains a measurable 3–5% on an iPhone 17 Pro in ~10 minutes —
/// enough for a stable joules-per-token estimate.
///
/// Drive it headless with, e.g.:
///
///     … -- --yardstick-autorun --runtime litert-lm \
///       --model-id "litert-community/gemma-4-E2B-it-litert-lm" \
///       --task energy --sustain-seconds 600
public struct EnergyTask: BenchmarkTask {
    public let id = "energy"
    public let title = "Energy (sustained)"
    public let summary = "Generates continuously for a fixed window so a 1%-resolution battery delta builds up. Reports J/token, average watts, and tokens/Wh."

    public let prompt = """
    Write a long, detailed technical essay on the history and future of computing. \
    Cover hardware (transistors, microprocessors, GPUs, neural accelerators), software \
    (operating systems, programming languages, compilers), networking, and the rise of \
    on-device machine learning. Use concrete examples and keep going in depth — do not \
    summarize or stop early.
    """

    public let parameters: GenerationParameters
    public let sustainSeconds: TimeInterval?

    /// - Parameters:
    ///   - sustainSeconds: total active-decode window to sustain (default 10 min).
    ///   - maxTokens: per-call output cap; the runner re-prompts when a call ends
    ///     before the window closes, so this only bounds prefill overhead, not
    ///     the total token count.
    public init(sustainSeconds: TimeInterval = 600, maxTokens: Int = 2048) {
        self.sustainSeconds = sustainSeconds
        self.parameters = GenerationParameters(maxTokens: maxTokens, temperature: 0.7, topP: 0.9)
    }
}
