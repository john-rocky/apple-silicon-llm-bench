import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Drives one camera-VLM session: load the model, then run back-to-back
/// frame→caption inferences for `task.durationSeconds`, sampling FPS, TTFT,
/// thermal state, memory and battery the whole time. Produces a `VLMRunResult`.
///
/// Mirrors `BenchmarkRunner`'s sampler/energy approach so the two harnesses
/// report power and thermals the same way.
public actor CameraVLMRunner {
    public struct Snapshot: Sendable {
        public let elapsed: TimeInterval
        public let remaining: TimeInterval
        public let phase: Phase
        public let currentFPS: Double
        public let inferenceCount: Int
        public let tokens: Int
        public let lastCaption: String
        public let thermalState: String
        public let aneResidencyPercent: Double?
    }

    public enum Phase: Sendable, Equatable {
        case idle
        case loadingModel(progress: Double)
        case running
        case finalizing
        case done
        case failed(String)
    }

    public struct Configuration: Sendable {
        public var runtime: any VLMRuntime
        public var model: ModelInfo
        public var task: CameraVLMTask
        public var frameProvider: CameraFrameProvider
        public var coldRun: Bool

        public init(runtime: any VLMRuntime, model: ModelInfo, task: CameraVLMTask,
                    frameProvider: CameraFrameProvider, coldRun: Bool) {
            self.runtime = runtime
            self.model = model
            self.task = task
            self.frameProvider = frameProvider
            self.coldRun = coldRun
        }
    }

    private var snapshotContinuation: AsyncStream<Snapshot>.Continuation?
    private var cancelled = false

    public init() {}

    public func snapshots() -> AsyncStream<Snapshot> {
        AsyncStream { self.snapshotContinuation = $0 }
    }

    public func cancel() { cancelled = true }

    public func run(_ config: Configuration) async throws -> VLMRunResult {
        cancelled = false
        var device = DeviceSnapshot.capture()
        let memorySampler = MemorySampler()
        let thermalSampler = ThermalSampler()
        let energyMonitor = EnergyMonitor()
        await thermalSampler.start()

        // 1. Load model if needed.
        var loadTime: Double?
        let currentLoaded = await config.runtime.loadedModelId
        if currentLoaded != config.model.id {
            emit(.loadingModel(progress: 0), fps: 0, count: 0, tokens: 0, caption: "")
            let loadStart = CFAbsoluteTimeGetCurrent()
            let scope = await MainActor.run { DownloadActivityScope() }
            defer { Task { @MainActor in scope.end() } }
            try await config.runtime.loadModel(config.model) { fraction in
                Task { await self.emit(.loadingModel(progress: fraction), fps: 0, count: 0, tokens: 0, caption: "") }
            }
            loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        }
        let aneResidency = await config.runtime.aneResidencyPercent()

        // 2. Start the camera and samplers.
        try await config.frameProvider.start()
        await memorySampler.start()
        await energyMonitor.start()
        // Let auto-exposure settle and the first frame land.
        try? await waitForFirstFrame(config.frameProvider)

        // 3. The session loop.
        let t0 = CFAbsoluteTimeGetCurrent()
        let duration = config.task.durationSeconds
        var completions: [Double] = []     // seconds since t0, one per finished inference
        var ttfts: [Double] = []           // ms
        var totalTokens = 0
        var lastCaption = ""
        var thermalSeries: [Int] = []
        var lastThermalSecond = -1

        emit(.running, fps: 0, count: 0, tokens: 0, caption: "")

        while !cancelled {
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            if elapsed >= duration { break }

            // Sample thermal once per wall-clock second (independent of FPS).
            let sec = Int(elapsed)
            if sec > lastThermalSecond {
                lastThermalSecond = sec
                thermalSeries.append(ProcessInfo.processInfo.thermalState.rawValue)
            }

            guard let frame = config.frameProvider.latestFrame() else {
                try? await Task.sleep(nanoseconds: 10_000_000)
                continue
            }

            let inferStart = CFAbsoluteTimeGetCurrent()
            var firstChunkAt: CFAbsoluteTime?
            var caption = ""
            var reportedTokens = 0

            let stream = config.runtime.describe(
                pixelBuffer: frame, prompt: config.task.prompt, parameters: config.task.parameters
            )
            do {
                for try await event in stream {
                    switch event {
                    case .chunk(let text):
                        if firstChunkAt == nil { firstChunkAt = CFAbsoluteTimeGetCurrent() }
                        if caption.count < 400 { caption.append(text) }
                    case .info(let info):
                        reportedTokens = info.generationTokenCount
                    }
                }
            } catch is CancellationError {
                break
            } catch {
                // One bad frame shouldn't kill a 10-minute run; log and continue.
                print("[CameraVLM] inference error: \(error.localizedDescription)")
                continue
            }

            let now = CFAbsoluteTimeGetCurrent()
            completions.append(now - t0)
            if let f = firstChunkAt { ttfts.append((f - inferStart) * 1000) }
            totalTokens += reportedTokens
            if !caption.isEmpty { lastCaption = caption }

            let fps = instantaneousFPS(completions, now: now - t0, window: 5)
            emit(.running, fps: fps, count: completions.count, tokens: totalTokens,
                 caption: lastCaption, ane: aneResidency, elapsed: now - t0, duration: duration)
        }

        // 4. Tear down + collect.
        emit(.finalizing, fps: 0, count: completions.count, tokens: totalTokens, caption: lastCaption)
        config.frameProvider.stop()
        await memorySampler.stop()
        await thermalSampler.stop()
        let memoryPeakMB = await memorySampler.peakMB
        let energy = await energyMonitor.snapshot()
        let sessionDuration = CFAbsoluteTimeGetCurrent() - t0

        let endBattery = DeviceSnapshot.currentBattery()
        device.batteryState = endBattery.state
        device.batteryLevel = endBattery.level

        let metrics = buildMetrics(
            coldRun: config.coldRun, loadTime: loadTime, sessionDuration: sessionDuration,
            completions: completions, ttfts: ttfts, totalTokens: totalTokens,
            thermalSeries: thermalSeries, aneResidency: aneResidency,
            thermalSampler: thermalSampler, energy: energy, memoryPeakMB: memoryPeakMB
        )
        // buildMetrics needs awaited thermal states — do it here.
        let resolved = await resolveThermal(metrics, sampler: thermalSampler)

        emit(.done, fps: resolved.sustainedFPS, count: completions.count,
             tokens: totalTokens, caption: lastCaption)

        return VLMRunResult(
            device: device,
            runtime: config.runtime.kind.rawValue,
            placement: config.runtime.placement.rawValue,
            model: config.model,
            task: config.task.id,
            prompt: config.task.prompt,
            frameSource: frameSourceLabel(config.frameProvider.source),
            parameters: config.task.parameters,
            metrics: resolved,
            lastCaption: String(lastCaption.prefix(200))
        )
    }

    // MARK: - Metrics

    /// Builds metrics with placeholder thermal strings; `resolveThermal` fills
    /// them from the (actor-isolated) sampler.
    private func buildMetrics(
        coldRun: Bool, loadTime: Double?, sessionDuration: Double,
        completions: [Double], ttfts: [Double], totalTokens: Int,
        thermalSeries: [Int], aneResidency: Double?,
        thermalSampler: ThermalSampler,
        energy: (joules: Double?, batteryDeltaPercent: Float, durationSeconds: TimeInterval),
        memoryPeakMB: Double
    ) -> VLMMetrics {
        let seconds = max(1, Int(sessionDuration.rounded()))
        let fpsOverTime = rollingFPS(completions, totalSeconds: seconds, window: 5)

        // Steady-state window = back 80% of the run (skip warm-up + first load spike).
        let steady = Array(fpsOverTime.dropFirst(fpsOverTime.count / 5)).filter { $0 > 0 }
        let sustained = median(steady)
        let head = Array(fpsOverTime.prefix(max(1, fpsOverTime.count / 10))).filter { $0 > 0 }
        let tail = Array(fpsOverTime.suffix(max(1, fpsOverTime.count / 10))).filter { $0 > 0 }
        let startFPS = head.isEmpty ? 0 : head.reduce(0, +) / Double(head.count)
        let endFPS = tail.isEmpty ? 0 : tail.reduce(0, +) / Double(tail.count)
        let peakFPS = fpsOverTime.max() ?? 0
        let drop = startFPS > 0 ? (1 - endFPS / startFPS) * 100 : 0

        let avgPowerW: Double? = {
            guard let j = energy.joules, energy.durationSeconds > 0 else { return nil }
            return j / energy.durationSeconds
        }()

        return VLMMetrics(
            coldRun: coldRun,
            loadTimeSeconds: loadTime,
            sessionDurationSeconds: sessionDuration,
            inferenceCount: completions.count,
            totalTokensGenerated: totalTokens,
            sustainedFPS: sustained,
            startFPS: startFPS,
            endFPS: endFPS,
            peakFPS: peakFPS,
            fpsDropPercent: drop,
            ttftMedianMS: median(ttfts),
            ttftP95MS: percentile(ttfts, 0.95),
            aneResidencyPercent: aneResidency,
            initialThermalState: "nominal",   // filled by resolveThermal
            peakThermalState: "nominal",
            finalThermalState: "nominal",
            peakTemperatureC: nil,
            energyJoules: energy.joules,
            batteryDeltaPercent: energy.batteryDeltaPercent,
            averagePackagePowerW: avgPowerW,
            energyMeasurementWindowSeconds: energy.joules != nil ? energy.durationSeconds : nil,
            energySource: energy.joules != nil ? "battery-1pct" : nil,
            fpsOverTime: fpsOverTime,
            thermalLevelOverTime: thermalSeries,
            memoryPeakMB: memoryPeakMB
        )
    }

    private func resolveThermal(_ m: VLMMetrics, sampler: ThermalSampler) async -> VLMMetrics {
        let initial = ThermalMonitor.describe(await sampler.initialState)
        let peak = ThermalMonitor.describe(await sampler.peakState)
        let final = ThermalMonitor.describe(await sampler.finalState)
        return VLMMetrics(
            coldRun: m.coldRun, loadTimeSeconds: m.loadTimeSeconds,
            sessionDurationSeconds: m.sessionDurationSeconds,
            inferenceCount: m.inferenceCount, totalTokensGenerated: m.totalTokensGenerated,
            sustainedFPS: m.sustainedFPS, startFPS: m.startFPS, endFPS: m.endFPS,
            peakFPS: m.peakFPS, fpsDropPercent: m.fpsDropPercent,
            ttftMedianMS: m.ttftMedianMS, ttftP95MS: m.ttftP95MS,
            aneResidencyPercent: m.aneResidencyPercent,
            initialThermalState: initial, peakThermalState: peak, finalThermalState: final,
            peakTemperatureC: m.peakTemperatureC,
            energyJoules: m.energyJoules, batteryDeltaPercent: m.batteryDeltaPercent,
            averagePackagePowerW: m.averagePackagePowerW,
            energyMeasurementWindowSeconds: m.energyMeasurementWindowSeconds,
            energySource: m.energySource,
            fpsOverTime: m.fpsOverTime, thermalLevelOverTime: m.thermalLevelOverTime,
            memoryPeakMB: m.memoryPeakMB
        )
    }

    // MARK: - Helpers

    private func emit(_ phase: Phase, fps: Double, count: Int, tokens: Int, caption: String,
                      ane: Double? = nil, elapsed: TimeInterval = 0, duration: TimeInterval = 0) {
        snapshotContinuation?.yield(Snapshot(
            elapsed: elapsed,
            remaining: max(0, duration - elapsed),
            phase: phase,
            currentFPS: fps,
            inferenceCount: count,
            tokens: tokens,
            lastCaption: caption,
            thermalState: ThermalMonitor.describe(ProcessInfo.processInfo.thermalState),
            aneResidencyPercent: ane
        ))
    }

    private func waitForFirstFrame(_ provider: CameraFrameProvider, timeout: TimeInterval = 5) async throws {
        let deadline = CFAbsoluteTimeGetCurrent() + timeout
        while provider.latestFrame() == nil {
            if CFAbsoluteTimeGetCurrent() > deadline { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    /// FPS over the last `window` seconds ending at `now`.
    private func instantaneousFPS(_ completions: [Double], now: Double, window: Double) -> Double {
        let lower = now - window
        let n = completions.filter { $0 >= lower && $0 <= now }.count
        let span = min(window, now)
        return span > 0 ? Double(n) / span : 0
    }

    /// One sample per second: trailing-`window`-second FPS at each second mark.
    private func rollingFPS(_ completions: [Double], totalSeconds: Int, window: Double) -> [Double] {
        guard !completions.isEmpty else { return Array(repeating: 0, count: totalSeconds) }
        var out: [Double] = []
        out.reserveCapacity(totalSeconds)
        for s in 1...totalSeconds {
            let t = Double(s)
            let lower = t - window
            let n = completions.filter { $0 >= lower && $0 <= t }.count
            let span = min(window, t)
            out.append(span > 0 ? Double(n) / span : 0)
        }
        return out
    }

    private func frameSourceLabel(_ source: CameraFrameProvider.Source) -> String {
        switch source {
        case .liveCamera: return "live-camera"
        case .loopingAsset: return "looping-asset"
        }
    }

    private func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        let mid = s.count / 2
        return s.count % 2 == 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid]
    }

    private func percentile(_ xs: [Double], _ p: Double) -> Double {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        let rank = max(1, Int((p * Double(s.count)).rounded(.up)))
        return s[min(rank - 1, s.count - 1)]
    }
}
