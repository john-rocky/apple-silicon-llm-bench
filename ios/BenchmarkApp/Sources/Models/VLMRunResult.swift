import Foundation

/// Result of one camera-VLM session. Kept separate from `BenchmarkResult`
/// because the headline is **frames/sec**, not decode tok/s, and the payload
/// carries the FPS-and-temperature time series the throttle chart is drawn
/// from. Serialized one-object-per-file under `Documents/results/`, same as the
/// text harness, so the export → `results/raw/*.jsonl` pipeline is unchanged.
public struct VLMRunResult: Codable, Sendable, Identifiable {
    public let id: UUID
    public let timestamp: Date
    public let device: DeviceSnapshot
    public let runtime: String          // "mlx-swift" | "coreml-llm"
    public let placement: String        // "gpu" | "ane"
    public let model: ModelInfo
    public let task: String             // "camera-vlm"
    public let prompt: String
    public let frameSource: String      // "live-camera" | "looping-asset"
    public let parameters: GenerationParameters
    public let metrics: VLMMetrics
    public let lastCaption: String

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        device: DeviceSnapshot,
        runtime: String,
        placement: String,
        model: ModelInfo,
        task: String,
        prompt: String,
        frameSource: String,
        parameters: GenerationParameters,
        metrics: VLMMetrics,
        lastCaption: String
    ) {
        self.id = id
        self.timestamp = timestamp
        self.device = device
        self.runtime = runtime
        self.placement = placement
        self.model = model
        self.task = task
        self.prompt = prompt
        self.frameSource = frameSource
        self.parameters = parameters
        self.metrics = metrics
        self.lastCaption = lastCaption
    }
}

public struct VLMMetrics: Codable, Sendable {
    public let coldRun: Bool
    public let loadTimeSeconds: Double?
    public let sessionDurationSeconds: Double

    // Throughput
    public let inferenceCount: Int
    public let totalTokensGenerated: Int
    /// Median inferences/sec over the steady-state window (the back 80% of the
    /// session, after warm-up). This is the headline number.
    public let sustainedFPS: Double
    public let startFPS: Double          // first ~10% window
    public let endFPS: Double            // last ~10% window
    public let peakFPS: Double
    /// 1 − end/start, in percent. Positive = throttled down by the end.
    public let fpsDropPercent: Double

    // Latency
    public let ttftMedianMS: Double
    public let ttftP95MS: Double

    // Compute placement
    public let aneResidencyPercent: Double?

    // Thermal
    public let initialThermalState: String
    public let peakThermalState: String
    public let finalThermalState: String
    /// Peak skin/SoC temperature in °C if a sensor log was attached out of band
    /// (iOS gives apps only the 4-level thermalState, so this is usually nil).
    public let peakTemperatureC: Double?

    // Energy (battery-delta, whole-system — reuses the text harness method)
    public let energyJoules: Double?
    public let batteryDeltaPercent: Float
    public let averagePackagePowerW: Double?
    public let energyMeasurementWindowSeconds: Double?
    public let energySource: String?

    // Time series — one sample per second of the session.
    public let fpsOverTime: [Double]
    /// Thermal level per second: 0 nominal, 1 fair, 2 serious, 3 critical.
    public let thermalLevelOverTime: [Int]

    // Memory
    public let memoryPeakMB: Double

    public init(
        coldRun: Bool,
        loadTimeSeconds: Double?,
        sessionDurationSeconds: Double,
        inferenceCount: Int,
        totalTokensGenerated: Int,
        sustainedFPS: Double,
        startFPS: Double,
        endFPS: Double,
        peakFPS: Double,
        fpsDropPercent: Double,
        ttftMedianMS: Double,
        ttftP95MS: Double,
        aneResidencyPercent: Double?,
        initialThermalState: String,
        peakThermalState: String,
        finalThermalState: String,
        peakTemperatureC: Double?,
        energyJoules: Double?,
        batteryDeltaPercent: Float,
        averagePackagePowerW: Double?,
        energyMeasurementWindowSeconds: Double?,
        energySource: String?,
        fpsOverTime: [Double],
        thermalLevelOverTime: [Int],
        memoryPeakMB: Double
    ) {
        self.coldRun = coldRun
        self.loadTimeSeconds = loadTimeSeconds
        self.sessionDurationSeconds = sessionDurationSeconds
        self.inferenceCount = inferenceCount
        self.totalTokensGenerated = totalTokensGenerated
        self.sustainedFPS = sustainedFPS
        self.startFPS = startFPS
        self.endFPS = endFPS
        self.peakFPS = peakFPS
        self.fpsDropPercent = fpsDropPercent
        self.ttftMedianMS = ttftMedianMS
        self.ttftP95MS = ttftP95MS
        self.aneResidencyPercent = aneResidencyPercent
        self.initialThermalState = initialThermalState
        self.peakThermalState = peakThermalState
        self.finalThermalState = finalThermalState
        self.peakTemperatureC = peakTemperatureC
        self.energyJoules = energyJoules
        self.batteryDeltaPercent = batteryDeltaPercent
        self.averagePackagePowerW = averagePackagePowerW
        self.energyMeasurementWindowSeconds = energyMeasurementWindowSeconds
        self.energySource = energySource
        self.fpsOverTime = fpsOverTime
        self.thermalLevelOverTime = thermalLevelOverTime
        self.memoryPeakMB = memoryPeakMB
    }
}

/// Persists a `VLMRunResult` to `Documents/results/` as one JSON file, matching
/// the text harness' on-disk convention so `UIFileSharingEnabled` export and
/// `scripts/import_ios_export.py` pick it up unchanged.
public enum VLMResultStore {
    public static func save(_ result: VLMRunResult) throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("results", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let stamp = ISO8601DateFormatter().string(from: result.timestamp)
            .replacingOccurrences(of: ":", with: "-")
        let safeModel = result.model.id.replacingOccurrences(of: "/", with: "_")
        let name = "\(result.runtime)-\(result.placement)-\(safeModel)-camera-vlm-\(stamp).json"

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        let url = dir.appendingPathComponent(name)
        try encoder.encode(result).write(to: url)
        return url
    }
}
