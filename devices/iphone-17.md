# iPhone 17

Base iPhone 17 (non-Pro) — the **volume** device: same A19 generation as the 17 Pro,
but the base GPU/thermal tier and 8 GB RAM, so it shows what the newest silicon does
*without* the Pro's headroom.

| Field | Value |
|-------|-------|
| Chip | Apple A19 |
| Model identifier | `iPhone18,3` |
| RAM | 8 GB |
| Memory bandwidth | ~68 GB/s (LPDDR5X 8533 MT/s — estimate, confirm) |
| Battery | ~14.3 Wh (3,692 mAh) — `EnergyMonitor.estimatedBatteryWh()` |
| Sustained cooling | No vapor chamber (Pro-only) |
| iOS version tested | iOS 26 |

## Notes

- **Why it's interesting:** same A19 generation as the 17 Pro, so **17 Pro vs 17** isolates
  the **Pro-vs-base tier** (GPU, bandwidth ~77 vs ~68 GB/s, vapor chamber) at a fixed generation.
- **8 GB RAM** is plenty for LiteRT-LM (Gemma E2B ≈ 0.6 GB) but tight for the GPU-heavy comparison
  runtimes (MLX ≈ 2.9 GB, llama.cpp ≈ 3.2 GB) under a 600 s sustained run — if one gets jetsam'd,
  that is itself a result (LiteRT-LM's footprint advantage matters most on 8 GB).
- **No vapor chamber** → expect the sustained-throttle curve to fall harder/sooner than the
  17 Pro's. That device-dependence is exactly the per-device question.
- Energy Wh constant is already wired; run **unplugged** from `nominal` per
  [`../methodology/energy-ios.md`](../methodology/energy-ios.md).

## Results

See the runtime/model rows in [`../RESULTS.md`](../RESULTS.md) filtered to `iPhone 17`, and the
LiteRT-LM per-device package in [`../docs/litert-lm/`](../docs/litert-lm/).
