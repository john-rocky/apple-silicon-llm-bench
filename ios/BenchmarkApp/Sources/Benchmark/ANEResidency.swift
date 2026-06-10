import Foundation
import CoreML

/// Computes a static "ANE residency %" for a compiled Core ML model: the
/// fraction of program operations the framework *predicts* it will run on the
/// Apple Neural Engine, as opposed to the GPU or CPU.
///
/// This uses `MLComputePlan` (iOS 17.4+ / macOS 14.4+), which walks the model's
/// `MLProgram` structure and, per operation, reports the *preferred* compute
/// device. We label each op ANE / GPU / CPU by the concrete device class of its
/// `preferred` device and report the ANE share.
///
/// Caveats, stated plainly because they matter for the writeup:
///  • This is a **static prediction**, not a runtime trace. The framework can
///    still fall back at execution time. For the ground-truth per-subsystem
///    power (ANE/GPU/CPU mW) you still need an Instruments Core ML + Power
///    trace — see `methodology/vlm-camera-ios.md`.
///  • A VLM is several sub-models (vision encoder, projector, decoder). Call
///    this per `.mlmodelc` and weight by op count, or report the vision
///    encoder's residency (the per-frame ANE workload) separately.
///  • Returns `nil` when the API is unavailable or the model is a plain
///    NeuralNetwork (not an `mlprogram`), where per-op device usage isn't
///    exposed the same way.
public enum ANEResidency {
    /// Fraction (0–100) of `MLProgram` operations whose preferred compute device
    /// is the Neural Engine. `nil` if it can't be determined.
    @available(iOS 17.4, macOS 14.4, *)
    public static func percent(
        ofCompiledModelAt url: URL,
        computeUnits: MLComputeUnits = .all
    ) async -> Double? {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        guard let plan = try? await MLComputePlan.load(contentsOf: url, configuration: config) else {
            return nil
        }
        guard case let .program(program) = plan.modelStructure else {
            // NeuralNetwork / pipeline models don't expose per-op device usage.
            return nil
        }

        var total = 0
        var onANE = 0
        for (_, function) in program.functions {
            for operation in function.block.operations {
                guard let usage = plan.deviceUsage(for: operation) else { continue }
                total += 1
                // MLComputeDevice is an enum: .cpu / .gpu / .neuralEngine.
                if case .neuralEngine = usage.preferred {
                    onANE += 1
                }
            }
        }
        guard total > 0 else { return nil }
        return Double(onANE) / Double(total) * 100.0
    }
}
