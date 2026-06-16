# Memory methodology

iOS will jetsam an app that uses too much memory. For LLM runtimes this is the dominant failure mode after thermal throttling.

## Sampling

We read **`phys_footprint`** via the Mach `task_info` `TASK_VM_INFO` call — the
byte count iOS charges the process and the exact value **jetsam** uses to decide
what to kill (dirty + compressed + IOKit-attributed memory):

```swift
var info = task_vm_info_data_t()
var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
let result = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
}
// info.phys_footprint  (bytes)
```

Sampled every 100 ms on a background queue during a run. Runs captured before
2026-06 used `mach_task_basic_info.resident_size` (RSS); `phys_footprint` is
typically **higher** because RSS omits compressed pages, so numbers across the
two eras are not byte-identical — re-measure a device to compare on one basis.
`MemoryMonitor.residentMB()` is still available for the RSS reference.

## Reported values

- `baseline_mb` — before the model is loaded
- `after_load_mb` — once the runtime reports model ready
- `peak_during_prefill_mb` — peak during prompt processing
- `peak_during_decode_mb` — peak during generation
- `after_generation_mb` — sampled 200 ms after generation completes
- `after_unload_mb` — only when the runtime exposes an unload API

The interesting deltas:

- `after_load_mb - baseline_mb` ≈ model + runtime overhead
- `peak_during_decode_mb - after_load_mb` ≈ KV cache + transient buffers
- `after_generation_mb - after_load_mb` ≈ steady-state cost of an idle loaded model

## `phys_footprint` vs `resident_size` vs `os_proc_available_memory()`

`resident_size` (RSS) omits compressed pages, so under memory pressure it
*under-reports* by hundreds of MB — it is **not** the number jetsam charges.
`os_proc_available_memory()` reports remaining headroom, but its semantics
shifted across iOS versions. `phys_footprint` is the stable, jetsam-relevant
figure and the one Instruments' "Memory" gauge shows, so that is what we report.

## Jetsam budget

The actual jetsam threshold depends on the device, foreground/background state, and what other apps are doing — Apple does not publish exact numbers. As a rule of thumb on a 6 GB iPhone:

- Foreground app, screen on: ~3 GB before jetsam risk
- Background app: ~200-500 MB

We do not enforce a budget in the benchmark. If a runtime gets jetsam'd, that is the result.

## Wired-memory ticket (MLX)

MLX Swift exposes `WiredMemoryTicket` for coordinating concurrent generations. We do **not** use it in the standalone benchmark, because the benchmark only runs one generation at a time. A separate "concurrent inference" task may be added later.
