# Runbook — one-shot full matrix (iPhone + Mac, all models, all tasks)

Driver: [`scripts/full_matrix.sh`](../../scripts/full_matrix.sh). Each step is an explicit phase;
nothing runs by itself. Phases are idempotent. Find the UDID with `xcrun devicectl list devices`.

## What the matrix covers
| | iPhone (devicectl) | Mac M4 Max (`yardstick` CLI) |
|---|---|---|
| Runtimes | litert-lm · mlx-swift · llama-cpp · coreml-llm | litert-lm · mlx-swift¹ |
| Qwen3 | 0.6B · 4B · (8B attempted²) | 0.6B · 4B · 8B |
| Gemma | 4-E2B (litert/mlx/llama/coreml) | 4-E2B (litert/mlx) |
| Lu's models³ | LFM2.5-350M · MiniCPM5-1B (our local `.litertlm` + mlx/gguf) | same (litert/mlx) |
| Tasks | short-chat · **long-context sweep** (2K/8K/32K) · sustained-generation · energy (unplugged) | short-chat · long-context sweep · sustained-generation |

**Continuous / long-context tasks** (the "does it hold up under load" axis):
- `long-context` / `long-context-8k` — same task at growing prompt lengths → prefill tok/s + TTFT
  scaling and **decode rate at KV depth** (is decode flat?). Nominal sizes are approximate; the real
  `promptTokenCount` is recorded and used by the report. (Early data: decode falls off sharply with
  context — e.g. mlx Qwen3-0.6B 390→219 tok/s from ~2.7K→10.7K tokens.)
- `long-context-32k` — a **0.6B-only ceiling probe**, NOT in the routine sweep: at ~32K the KV cache
  explodes (~105 GB even on a 0.6B model with mlx), so it OOMs on ≥4B. Run it by hand on small models.
- `sustained-generation` — a 512-token continuous decode → decode-degradation curve
  (`decodeRateRollingWindow`). This is a *finite* decode and does **not** hit the `sustainSeconds`
  hang; only the unplugged `energy` task (continuous re-prompting) does, and litert is skipped there.

¹ The SwiftPM CLI ships MLX + LiteRT-LM only. llama.cpp / CoreML on Mac use the **xcodebuild**
target — but it currently can't co-build with LiteRT-macOS (an `executorch`×`CLiteRTLM_mac`
`module.modulemap` collision). To get Mac llama/coreml, build the xcodebuild `yardstick` target
*without* the LiteRT-macOS dep, or resolve the collision first. (TODO, P2.)
² Qwen3-8B (~4.5 GB) may exceed iOS's per-app memory ceiling and get jetsam-killed — that result
is itself recorded (fairness rule 4), same as the gemma-3n CoreML case.
³ Lu's focus models, converted by us (`~/code/litertlm-convert/deliverables/`) since LiteRT publishes
neither on `litert-community`. `prefetch` side-loads them (no HF). Mac preview: MiniCPM5-1B litert runs
(239 tok/s); LFM2.5-350M litert loads-but-fails-to-invoke on 0.13.1 — the iPhone run retests it.

## Known, expected outcomes (not bugs in the harness — record them, rule 4)
- **litert energy/sustained hangs** on 0.13.x (`DEADLINE_EXCEEDED` in `callback_thread_pool`). The
  driver **skips** litert in `iphone-energy`. See [`LITERT_SUSTAINED_HANG.md`](LITERT_SUSTAINED_HANG.md).
- **litert on macOS is CPU-only** (GPU/OpenCL context fails). See [`MACOS_DESKTOP.md`](MACOS_DESKTOP.md).
- **CoreML gemma-3n** peaks ~3.3 GB → jetsam on repeated runs (iPhone). A memory-trimmed bundle is needed.

## Order of operations
```bash
UDID=<udid>                                   # from: xcrun devicectl list devices

# Mac side (do when the Mac is free — it shares GPU with other work):
./scripts/full_matrix.sh build                # SwiftPM yardstick CLI (Release)
./scripts/full_matrix.sh mac                  # litert+mlx × all models × {short-chat,long-context,sustained}

# iPhone side:
#   (0) ONE-TIME: open ios/BenchmarkApp in Xcode, Release, ⌘R  (the new Qwen3-4B/8B catalog rows
#       need a fresh install; CLI signing isn't available headless).
UDID=$UDID ./scripts/full_matrix.sh prefetch  # download all on Mac + side-load iPhone-side models
UDID=$UDID ./scripts/full_matrix.sh iphone    # plugged: short-chat + long-context, n=3 cold
#   then UNPLUG the phone:
UDID=$UDID ./scripts/full_matrix.sh iphone-energy   # 600 s battery-delta per model (litert skipped)
UDID=$UDID ./scripts/full_matrix.sh collect   # pull Documents/results -> /tmp/yardstick-collect

# Reduce:
#   rename collected JSONL -> results/raw/<device>-<rt>-<model>-<task>-runN.jsonl
#   (rt token: mlx-swift->mlx, llama.cpp->llama-cpp; model token: qwen3-0.6b / qwen3-4b / gemma-4-e2b …)
./scripts/full_matrix.sh report               # regenerate docs/litert-lm/ ; update MACOS_DESKTOP.md by hand
```

## Fairness invariants the driver holds (verify after collect)
- `coldRun: true` and `initialThermalState: nominal` on each short-chat/long-context run (re-run drift).
- 128-token output cap (`generatedTokenCount` ≈ 128) so every runtime emits the same count.
- `buildConfiguration: Release`, `phys_footprint` memory, n=3 median.
- Quant is each runtime's native 4-bit (disclosed per row); GPU vs ANE vs CPU is disclosed per row.

## Side-load paths (what `prefetch` writes, for manual fixes)
- llama.cpp → `Documents/models/llama.cpp/<repo with / → __>/<file.gguf>`
- litert-lm → `Documents/models/litert-lm/<repo with / → __>/<file.litertlm>`
- mlx-swift → `Library/Caches/huggingface/hub/models--<org>--<name>/{blobs,refs}` (HubClient rebuilds `snapshots/`)
- coreml-llm → `Documents/Models/<folder>/` (the prebuilt ANE bundle, e.g. `gemma4-e2b`)

On-device HF download stalls ~32 MB for large models, so side-load (Mac download → `devicectl copy to`)
is **required**, not optional.

## Extending the matrix
Add a line to the `QWEN_*` / `GEMMA` arrays at the top of `full_matrix.sh`
(`runtime|catalog-id|hf-repo|file-glob`) and the matching `ModelCatalog` entry, then re-`build`
(Mac) / ⌘R (iPhone). The full verified comparator set (litert/mlx/gguf per model + device fit) is in
[`MODEL_MATRIX.md`](MODEL_MATRIX.md); availability/conversion notes in
[`MODEL_AVAILABILITY.md`](MODEL_AVAILABILITY.md). Qwen3-14B and Gemma-12B are Mac-tier only
(phones jetsam) and already wired as `MAC_BIG`. Quality-parity and a long-context *sweep* (multiple lengths) are still TODO
(both need a Swift change + the one ⌘R).
