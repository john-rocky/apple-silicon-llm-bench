import Foundation
import Darwin

/// Process-memory sampling via Mach `task_info`.
///
/// We report **`phys_footprint`** (`TASK_VM_INFO`) — the byte count iOS charges
/// the process and the exact value **jetsam** uses to decide what to kill. It
/// counts dirty + compressed + IOKit-attributed memory, so it tracks the real
/// shipping-app ceiling far better than `resident_size` (which omits compressed
/// pages and can under-report by hundreds of MB under memory pressure). On an
/// 8 GB device the line between "fits" and "jetsam" is precisely this number,
/// which is why it is the honest memory metric for a benchmark.
public enum MemoryMonitor {
    /// Current physical footprint in megabytes — the jetsam-relevant figure.
    /// Returns 0 on failure.
    public static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / (1024 * 1024)
    }

    /// Resident size (RSS) in megabytes. Kept for reference and back-compat with
    /// pre-`phys_footprint` runs; prefer `footprintMB()` — jetsam looks at
    /// `phys_footprint`, not `resident_size`. Returns 0 on failure.
    public static func residentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / (1024 * 1024)
    }
}

/// Records peak physical footprint across a sliding window.
public actor MemorySampler {
    private(set) var peakMB: Double = 0
    private var task: Task<Void, Never>?

    public init() {}

    public func start(intervalMS: Int = 100) {
        stop()
        peakMB = MemoryMonitor.footprintMB()
        task = Task { [weak self] in
            while !Task.isCancelled {
                let current = MemoryMonitor.footprintMB()
                await self?.bump(current)
                try? await Task.sleep(nanoseconds: UInt64(intervalMS) * 1_000_000)
            }
        }
    }

    public func stop() {
        task?.cancel()
        task = nil
    }

    private func bump(_ value: Double) {
        if value > peakMB { peakMB = value }
    }
}
