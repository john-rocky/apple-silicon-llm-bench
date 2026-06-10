import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Battery-delta-based energy estimation.
///
/// iOS does not expose powermetrics-style energy counters to third-party apps.
/// What we *can* read is `UIDevice.current.batteryLevel`, which is reported in
/// 1% steps. We sample at the start and end of a run, take the delta, and
/// multiply by an estimated whole-pack energy capacity to get joules.
///
/// Limitations:
/// - 1% resolution means short runs (< ~30 s on a 0.6 B model) may show 0%
///   delta. Energy is reported as `nil` in that case.
/// - Battery capacity is per-device. We use a conservative pack capacity
///   table keyed off the hardware model identifier; unknown devices fall
///   back to a 12 Wh estimate.
/// - The number includes display + radios + everything else the OS is doing.
///   For comparability, run with brightness fixed and Airplane Mode on (the
///   pre-flight checklist surfaces this).
///
/// Despite the limitations the metric is useful: a runtime that drains 0.5%
/// for the same workload as another that drains 1.5% is meaningfully more
/// energy-efficient on the same device.
public actor EnergyMonitor {
    private(set) var startedAt: CFAbsoluteTime = 0
    private(set) var startBatteryLevel: Float = -1
    private(set) var startThermalState: ProcessInfo.ThermalState = .nominal

    public init() {}

    public func start() {
        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        startBatteryLevel = UIDevice.current.batteryLevel
        #else
        startBatteryLevel = -1
        #endif
        startedAt = CFAbsoluteTimeGetCurrent()
        startThermalState = ProcessInfo.processInfo.thermalState
    }

    /// Returns (joulesUsed, batteryDeltaPercent, durationSeconds).
    /// `joulesUsed` is `nil` if no measurable battery drop was observed.
    public func snapshot() -> (joules: Double?, batteryDeltaPercent: Float, durationSeconds: TimeInterval) {
        let duration = CFAbsoluteTimeGetCurrent() - startedAt
        #if canImport(UIKit)
        let endLevel = UIDevice.current.batteryLevel
        // batteryLevel is -1 in simulator and on devices that haven't reported yet.
        guard startBatteryLevel >= 0, endLevel >= 0 else {
            return (nil, 0, duration)
        }
        let delta = startBatteryLevel - endLevel
        guard delta > 0 else {
            return (nil, 0, duration)
        }
        let packWh = Self.estimatedBatteryWh()
        // 1 Wh = 3600 J. Energy used = packWh * delta_fraction * 3600.
        let joules = Double(packWh) * Double(delta) * 3600
        return (joules, delta * 100, duration)
        #else
        return (nil, 0, duration)
        #endif
    }

    /// Estimate battery capacity in watt-hours for the current device.
    ///
    /// Numbers are pulled from public battery datasheets / iFixit teardowns
    /// for each device model. New device identifiers fall back to 12 Wh,
    /// which is roughly the iPhone-non-Pro median.
    private static func estimatedBatteryWh() -> Double {
        let model = hardwareModelIdentifier()
        switch model {
        // iPhone 17 family. Apple publishes mAh, not Wh; we convert at the
        // ~3.88 V nominal implied by the iPhone 17 Pro Max teardown
        // (5,112 mAh ↔ 19.99 Wh). The Pro ships in two pack sizes — the
        // global/physical-SIM unit (3,988 mAh ≈ 15.5 Wh) and the US eSIM-only
        // unit (4,252 mAh ≈ 16.5 Wh). The measured device is the eSIM-only
        // unit; switch to 15.5 for a physical-SIM Pro.
        case "iPhone18,1": return 16.5  // iPhone 17 Pro (US eSIM-only, 4,252 mAh)
        case "iPhone18,2": return 20.0  // iPhone 17 Pro Max (5,112 mAh, teardown)
        case "iPhone18,3": return 14.3  // iPhone 17 (3,692 mAh)
        // iPhone 16 family
        case "iPhone17,1": return 13.0  // iPhone 16 Pro
        case "iPhone17,2": return 17.0  // iPhone 16 Pro Max
        case "iPhone17,3": return 13.0  // iPhone 16
        case "iPhone17,4": return 14.0  // iPhone 16 Plus
        // iPhone 15 family
        case "iPhone16,1": return 12.7  // iPhone 15 Pro
        case "iPhone16,2": return 16.7  // iPhone 15 Pro Max
        case "iPhone15,4": return 12.4  // iPhone 15
        case "iPhone15,5": return 14.0  // iPhone 15 Plus
        // iPhone 14 family
        case "iPhone15,2": return 12.4  // iPhone 14 Pro
        case "iPhone15,3": return 16.7  // iPhone 14 Pro Max
        case "iPhone14,7": return 12.7  // iPhone 14
        case "iPhone14,8": return 16.7  // iPhone 14 Plus
        default: return 12.0
        }
    }

    private static func hardwareModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        return mirror.children.reduce(into: "") { partial, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            partial.append(Character(UnicodeScalar(UInt8(value))))
        }
    }
}
