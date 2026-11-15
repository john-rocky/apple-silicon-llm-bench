# Comprehensive on-device benchmark — runbook (prep'd 2026-06-24)

Goal: one pass that captures **speed + memory + energy** across **every comparable model × runtime** on
iPhone 17 Pro. Everything Mac-side is staged; this is the plug-in-and-go guide.

Driver: `scripts/comprehensive_bench.sh` · Matrix: `results/raw/2026-06-25-comprehensive/manifest.tsv`
(set `DEV=<devicectl id>` env if the identifier changed.)

## 0. Pre-flight (once)
- **App build:** must be the one with the memory entitlements (`increased-memory-limit` +
  `extended-virtual-addressing`). It's wired in `ios/BenchmarkApp/project.yml` + `BenchmarkApp.entitlements`;
  the capabilities are registered on the App ID. Rebuild via Xcode (capabilities need the GUI once) or, if the
  profile is cached, `xcodebuild ... -derivedDataPath ~/bench-dd build` then install. **Without these, ≳2 GB
  models falsely OOM** (this was the whole 3B saga — see `docs/litert-community-vs-mlx-coreai.md`).
- **Device space:** the side-load set is large (Qwen3-8B ANE 5.2 GB, etc.). Free space first; uninstall/reinstall
  the app to clear stale side-loads if needed. >1 GB pushes over WiFi can truncate — prefer USB for `stage`.

## 1. Stage the bundles (USB)
```
scripts/comprehensive_bench.sh stage
```
Side-loads all Core AI (`Documents/CoreAIModels/…`) and LiteRT-local (`Documents/models/litert-lm/…`) bundles
from the Mac. `download` rows (litert-community, mlx-community) pull on-device on first run — keep WiFi on.

## 2. Pass A — speed + memory + TTFT + ITL (plugged in OK, ~40 min)
```
scripts/comprehensive_bench.sh speed            # whole matrix
scripts/comprehensive_bench.sh speed Qwen3      # one family
```
Runs `short-chat` (128 tok, greedy) 3× cold per cell → `speed_mem.jsonl` with `decode_tps`, `ttft_ms`,
`peak_mb`. The app also records ITL p50/p95/p99, prefill tok/s, thermal, rolling decode-rate (in
`Documents/results/` — pull with `collect`). Memory is `phys_footprint` (jetsam-charged), sampled every 100 ms.

## 3. Pass B — energy (UNPLUGGED, battery-limited, do selectively)
Energy needs a real battery drain, so **USB must be unplugged** — over USB the device charges and
`energyJoules` comes back `nil`. Each run is the `energy` task: sustained 600 s / 2048-tok calls (override with
the 3rd/4th args). One run ≈ 10 min and drains a few %, so **don't run the whole matrix in one charge** — pick
the key comparison cells, recharge to ~90% between batches.

Per run:
```
# battery 80–95%, Low Power Mode OFF, brightness fixed + auto-brightness OFF, no other foreground apps
scripts/comprehensive_bench.sh energy core-ai/qwen3-1.7b-gpu core-ai 600 2048
#  → launches DETACHED; UNPLUG USB NOW; wait ~690 s; then:
scripts/comprehensive_bench.sh collect
```
`collect` pulls `Documents/results/` and prints the energy rows. **`energyJoules != nil` is the proof the
battery actually dropped** (the metric self-verifies). Key derived numbers: `energyJoulesPerToken`,
`averagePackagePowerW`, `batteryDeltaPercent`, plus `tokens/Wh` (compute from joules; iPhone 17 Pro pack = 16.5 Wh).

Suggested energy subset (the efficiency story = the next theme):
- ANE vs GPU on Core AI (same model): `core-ai/qwen3-1.7b-ane` vs `…-gpu` — **the money shot** (ANE should win J/token).
- Core AI vs MLX vs LiteRT on one mid model (Qwen3-1.7B) — efficiency across runtimes.
- A 3B (Llama-3.2-3B) MLX vs LiteRT — efficiency at the size that stresses memory.

## Comparable matrix (what's staged)
| Family | Sizes | Core AI (ANE/GPU) | MLX | LiteRT-LM |
|---|---|---|---|---|
| Qwen3 | 0.6 / 1.7 / 4 / 8B | ✓ all (side-load) | ✓ 4bit | ✓ (0.6/4/8B community, 1.7B local int4) |
| DeepSeek-R1 | 1.5B | ✓ | ✓ | ✓ |
| TinySwallow | 1.5B | ✓ | ✓ | ✓ |
| VibeThinker | 1.5B | ✓ (ANE 71.5/GPU 75.7 already) | ✗ no repo | ✓ |
| Llama-3.2 | 3B | ✗ no iOS class | ✓ | ✓ local (needs entitlements) |
| SmolLM3 | 3B | ✗ no wrapper | ✓ | ✓ local |
| Ministral | 3B | ✗ ministral3 arch | ✗ MLX-Swift can't load | ✓ local |
| OLMo-2 | 1B | ✗ no wrapper | ✗ no repo | ✓ local |
| Gemma4-E2B / E4B | 2B/4B-eff | (GPU IR exists, no catalog id) | ✓ | ✓ community |

## Gotchas (learned the hard way)
- **Memory entitlements are mandatory** for ≳2 GB models — the 3B "OOM" was a missing-entitlement artifact, not a
  runtime limit (Ministral-3B 0→3/3, Llama-3B 1/3→5/5 once added). Phi-4-mini int8 (3.6 GB) genuinely OOMs — but
  that's the *quant*; an int4 Phi (~2 GB) would load.
- **Mac crashes from running iOS bundles on the Mac** (litert-mac-verify on an iOS artifact). Exports/compiles are
  safe; verify on-device. Don't run iOS bundles on the Mac.
- **CoreAI compute unit is fixed at export** (ANE = iOS static palettized; GPU = macOS dynamic). ANE first-load
  compiles on-device (slow cold), warms after.
- **Cold vs warm:** every headless launch is a fresh process = cold. For warm, launch once manually, then drive.
- **`devicectl --console` inside a `while read` loop eats the loop's stdin** — the driver uses `</dev/null`.
- **Wireless option for energy:** pair over WiFi (`…coredevice.local`) so you can drive runs with USB unplugged.

## Mac iso protocol — make prefill iso too (NEW; iPhone is already iso)
The published Mac table measured decode iso (same 512-gen + greedy for all runtimes) but the **prefill prompt was
not iso** (Core AI used a 512-token synthetic prompt, MLX/LiteRT short prompts → prefill tok/s only directional).
Fix going forward: **all three runtimes use the SAME prompt-token-count AND the SAME decode-token-count**, greedy.

Standard: **512-token prefill + 512 decode, greedy, n≥3.** One shared prompt for everyone.
1. Build a fixed ~512-token prompt once (filler is fine — content doesn't change decode; only the token count must
   match). Save it as `prompt_512.txt` and reuse for MLX + LiteRT; Core AI's `-p 512` is its synthetic equivalent.
2. Commands (identical prefill 512 / decode 512):
   - **Core AI:** `llm-benchmark --model <bundle> -p 512 -g 512 -n 5`  (was `-g 1024`; use `-g 512` to match)
   - **MLX:** `mlx_lm.generate --model <repo> --prompt "$(cat prompt_512.txt)" --max-tokens 512 --temp 0`
     (or the `scripts/measure_energy.py` wrapper for the streamed decode rate)
   - **LiteRT-LM:** `litert-mac-verify <model> "$(cat prompt_512.txt)" --max-tokens 512 --backend gpu`
3. Report decode tok/s (iso as before) AND prefill tok/s (now iso — drop the "directional-only" caveat once
   re-measured). Update the report's Conditions table to "Mac prefill = 512-token shared prompt (all runtimes)".

Note: decode tok/s barely moves vs the old numbers (decode is prefill-insensitive); this mainly makes the
**prefill** column a fair apples-to-apples comparison and removes the only non-iso caveat in the study.
