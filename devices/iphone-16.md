# iPhone 16

Base iPhone 16 (non-Pro), A18 — the **prior-generation** base device, so iPhone 17 vs 16
isolates the **gen-over-gen** step on the volume tier (A19 vs A18).

| Field | Value |
|-------|-------|
| Chip | Apple A18 |
| Model identifier | `iPhone17,3` |
| RAM | 8 GB |
| Memory bandwidth | ~60 GB/s (LPDDR5X — estimate, confirm) |
| Battery | ~13.0 Wh — `EnergyMonitor.estimatedBatteryWh()` |
| Sustained cooling | No vapor chamber |
| iOS version tested | iOS 26 |

## Notes

- **Why it's interesting:** base A18 vs base A19 (the iPhone 17) is the clean **generational**
  delta on the volume tier — how much one year of silicon buys for on-device decode, and (via
  the bandwidth roofline in the [LiteRT-LM package](../docs/litert-lm/)) how much of it is memory.
- Same **8 GB / no-vapor-chamber** caveats as the iPhone 17 (see that page): LiteRT-LM fits
  comfortably; the GPU-heavy runtimes sit near the jetsam edge under sustained load.
- Energy Wh constant is already wired; run **unplugged** from `nominal` per
  [`../methodology/energy-ios.md`](../methodology/energy-ios.md).

## Results

See the runtime/model rows in [`../RESULTS.md`](../RESULTS.md) filtered to `iPhone 16`, and the
LiteRT-LM per-device package in [`../docs/litert-lm/`](../docs/litert-lm/).
