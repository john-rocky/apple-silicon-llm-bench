# Energy methodology — iPhone (battery-delta)

How the iPhone battery-efficiency rows (J/token, average watts, tokens/Wh,
tokens-per-1%-battery) are produced. This is the on-device counterpart to the
Mac `powermetrics` flow in [`energy.md`](energy.md); the two use **different
instruments** and are not directly comparable in absolute joules — compare
runtimes *within a device*, never one device to another.

## The instrument

iOS exposes no powermetrics-style energy counter to third-party apps. The one
signal we can read is `UIDevice.current.batteryLevel`, reported in **1% steps**.
We sample it at the start and end of a run and convert the drop to joules:

```
joules        = (start_pct − end_pct) × pack_capacity_Wh × 3600
J_per_token   = joules / generated_tokens
avg_power_W   = joules / window_seconds            # = ΔWh × 3600 / Δt
tokens_per_Wh = generated_tokens / (joules / 3600)
tokens_per_1% = generated_tokens / battery_delta_percent
```

`pack_capacity_Wh` is a per-device constant in
[`EnergyMonitor.estimatedBatteryWh()`](../ios/BenchmarkApp/Sources/Benchmark/EnergyMonitor.swift).
For the measured **iPhone 17 Pro (`iPhone18,1`, US eSIM-only)** we use
**16.5 Wh** — the 4,252 mAh pack at the ~3.88 V nominal implied by the iPhone 17
Pro Max teardown (5,112 mAh ↔ 19.99 Wh). The global/**physical-SIM** Pro ships a
smaller 3,988 mAh ≈ **15.5 Wh** pack; on that variant every absolute joule here
scales by 15.5/16.5 (≈ −6%). The `batteryDeltaPercent` field is recorded in every
JSONL, so joules can be rescaled to either pack after the fact without
re-measuring.

## Why a dedicated `energy` task (not short-chat)

A 128-token short-chat reply finishes in seconds and burns far less than 1% of
the pack — below the battery sampler's resolution, so it reports no energy at
all. The **`energy` task** instead keeps the runtime generating (re-prompting
on EOS) for a fixed window — **600 s by default** — which drains a measurable
**3–5%** on an iPhone 17 Pro. That is enough signal for a stable per-token
estimate while staying short enough to run four runtimes back-to-back.

Drive it headless:

```
xcrun devicectl device process launch --device <udid> \
  com.iosllmbenchmark.benchmarkapp -- \
  --yardstick-autorun --runtime <kind> --model-id "<id>" \
  --task energy --sustain-seconds 600 --runs 1
```

Watch for `YARDSTICK_ENERGY … state=unplugged battery_delta_pct=… joules=…`.

## ⚠️ Run unplugged — the make-or-break

**USB power charges the phone, so the battery never drops and `energyJoules`
comes back `nil`.** The run must execute on battery. Two ways:

1. **Wireless `devicectl` (recommended for live driving).** Pair the phone over
   the network (it shows up as e.g. `DaisukenoiPhone.coredevice.local`), unplug
   USB, and drive with `devicectl --device <hostname>`. **Wi-Fi must stay on**,
   so this is *not* full Airplane Mode — leave Wi-Fi enabled and treat its idle
   draw as a small constant shared by every runtime. Cellular off.
2. **Launch-then-unplug (full Airplane Mode).** Launch the run over USB *without*
   `--console` (the command returns immediately), then pull the cable. The app
   keeps the screen awake (idle timer disabled) and runs to completion on
   battery, saving the JSONL to `Documents/results/`. Reconnect afterwards and
   `devicectl device copy from` to collect it.

Either way the runner refreshes `device.batteryState` to the **end-of-run**
value, and `energyJoules` is non-`nil` **only** if the level actually fell — so a
populated energy figure is itself proof the run was on battery. Still, verify
`device.batteryState == "unplugged"` in each JSONL and discard anything else.

## Pre-flight checklist (hold constant across the runtimes you compare)

- [ ] **Unplugged**, on battery (see above).
- [ ] **Low Power Mode OFF** — it throttles the CPU/GPU and would confound runtimes.
- [ ] **Brightness fixed** (e.g. 50%) and **Auto-Brightness OFF** — the display is
      part of the whole-system draw.
- [ ] **Start battery 80–95%** — the charge curve is non-linear near full and
      empty; the 80–95% band keeps Wh-per-% closest to the nominal constant.
- [ ] **Screen on** the whole run (the app disables auto-lock).
- [ ] **No other foreground apps**, notifications quiet, background app refresh off.
- [ ] **Record thermal + room temperature.** The JSONL carries
      `peakThermalState`; note ambient temp in the PR — a hot room throttles.
- [ ] Same model **file** and prompt as every other runtime in the comparison.

## What the number is and isn't

- **Whole-system, not chip-only.** It includes the display, Wi-Fi idle, and
  background OS work — everything the battery powered during the window. Good for
  ranking runtimes under identical conditions; not an isolated "inference-only"
  joule count. (The Mac `powermetrics` rows have the same whole-system caveat.)
- **±1% quantization dominates the error bar.** A 4% drop read at 1% resolution
  carries roughly ±0.5% absolute → **±12.5%** on joules. Prefer larger drops
  (longer `--sustain-seconds`) and, where time allows, repeat runs. This swamps
  the ~1% uncertainty in nominal pack voltage, so chasing a more precise Wh
  constant is not worth it.
- **Pack capacity drifts with battery health.** An aged pack has less than its
  rated Wh; the JSONL records the hardware id and battery level so contributors
  with degraded batteries can flag it.
- **Thermal throttling skews efficiency.** A throttled run does less work per
  joule; cross-check `peakThermalState`.

## The question this axis answers

On the **M4 Max**, CoreML/ANE had the *lowest instantaneous watts* yet the
*worst* J/token — its slow decode kept the package powered far longer, so speed
won the energy race; Apple FM was most efficient. **Does the same hold on the
phone, or does the ANE's low power finally win J/token when the GPU runtimes pay
their on-device throughput tax?** That is what these rows are for.
