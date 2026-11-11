# Methods & reproduction — LiteRT-LM cross-runtime bench (Apple silicon)

> Everything needed to **re-run and cite** the [LiteRT-LM package](README.md). Pinned
> versions, exact model files, fixed device state, and the fairness rules in one place. Companion to
> the auto-generated [`README.md`](README.md) (numbers) and
> [`LITERT_SUSTAINED_HANG.md`](LITERT_SUSTAINED_HANG.md) (the one known failure). Part of the neutral
> [Apple Silicon LLM Benchmark](../../README.md) — one headless harness for every runtime.

## Near-one-command repro (iPhone, headless)

The phone build needs your signing once; everything after is scripted. From the repo root:

```bash
# 0. Fetch pinned runtimes (LiteRT-LM v0.13.1, llama.cpp b8999, CoreML-LLM, coreai-models …)
cd ios/BenchmarkApp && ./scripts/bootstrap.sh

# 1. Build + install the app ONCE in Release (your Apple Developer team), then ⌘R or:
open BenchmarkApp.xcodeproj    # Scheme ▸ Run ▸ Build Configuration: Release

# 2. Drive the whole short-chat matrix headless (3 cold launches per runtime/model)
UDID=$(xcrun devicectl list devices | awk '/iPhone/{print $NF; exit}')
UDID="$UDID" ./scripts/run_device_bench.sh            # runtime × model × 3 cold runs
UDID="$UDID" ./scripts/run_device_bench.sh collect    # copy JSONL off-device

# 3. Rename into results/raw/<device>-<runtime>-<model>-short-chat-runN.jsonl, then regenerate:
python3 scripts/litert_lm_report.py                   # rewrites docs/litert-lm/README.md
```

Energy / throttle (sustained) is a separate, **unplugged** protocol — see
[`methodology/energy-ios.md`](../../methodology/energy-ios.md) and the `--task energy` invocation
below. Mac / desktop tier (Qwen3 8B/14B that phones can't hold) runs with no signing via the
`yardstick` CLI: `swift run -c release yardstick run --task short-chat --runtime <rt> --model <id>
--output <jsonl>`.

## Pinned versions (fix these to reproduce a number exactly)

| Component | Pin | Where it's set |
|---|---|---|
| **LiteRT-LM** | **v0.13.1** | `bootstrap.sh` `LITERTLM_TAG=v0.13.1`; SPM `from: "0.13.0"`; vendored `version.bzl` `VERSION = "0.13.1"` |
| **llama.cpp** | release **`b8999`** (Metal xcframework) | `bootstrap.sh` `LLAMA_TAG=b8999` |
| **MLX** | `ml-explore/mlx-swift-lm` **`branch: "main"`** ⚠️ | `Package.swift` / `project.yml` — *unpinned; pin a commit for exact repro* |
| **swift-transformers** | `from: "1.0.0"` (Anemll patched to `1.0.0..<2.0.0`) | `Package.swift`, `bootstrap.sh` |
| **swift-huggingface** | `from: "0.8.1"` | `Package.swift` |
| **CoreML-LLM** | `john-rocky/CoreML-LLM` default branch ⚠️ | `bootstrap.sh` — *unpinned; pin a commit for exact repro* |
| **coreai-models** | `apple/coreai-models` **`0.1.0`** | `bootstrap.sh` `COREAI_TAG=0.1.0` |
| **OS / device** | iOS **27.0**, iPhone 17 Pro (`iPhone18,1`) | recorded per JSONL (`systemVersion`, `modelIdentifier`) |
| **Build config** | **Release** | fairness rule 7; `run_device_bench.sh` header |

> ⚠️ Two dependencies (`mlx-swift-lm`, `CoreML-LLM`) track a moving branch today. The reported
> numbers are honest for the resolved graph at capture time, but for a *bit-exact* re-run pin both to
> a commit. Recording the resolved `Package.resolved` alongside a result set is the clean fix and is
> on the to-do list.

## Exact model files (per runtime)

Each runtime uses its **native** format and quant — disclosed per row, never silently equated
(fairness rule 3). HF downloads currently resolve the **latest snapshot**; pin a `revision`/commit
per model for exact reproduction (tracked).

| Runtime | Qwen3-0.6B | Gemma-4-E2B |
|---|---|---|
| LiteRT-LM / GPU | `litert-community/Qwen3-0.6B` → `qwen3_0_6b_mixed_int4.litertlm` (mixed INT4, gs32) | `litert-community/gemma-4-E2B-it-litert-lm` → `gemma-4-E2B-it.litertlm` (INT4 QAT) |
| MLX-Swift / GPU | `mlx-community/Qwen3-0.6B-4bit` (Q4) | `mlx-community/gemma-4-e2b-it-4bit` (Q4) |
| llama.cpp / GPU | — | `unsloth/gemma-4-E2B-it-GGUF` → `Q4_K_M` |
| CoreML / ANE | `coreml-llm/qwen3-0.6b` (INT8 palettized) | `coreml-llm/gemma4-e2b` (INT8) |
| Core AI / GPU+ANE | `core-ai/qwen3-0.6b-{gpu,ane}` (exported `.aimodel`, AOT-compiled per GPU arch, side-loaded) | — |

## Fixed device state (hold constant across the runtimes you compare)

From [`methodology/energy-ios.md`](../../methodology/energy-ios.md) and
[`methodology/thermal.md`](../../methodology/thermal.md):

- **Unplugged**, on battery — for energy runs the level must actually fall, else `energyJoules` is
  `nil` (a populated figure is itself proof the run was on battery).
- **Low Power Mode OFF**; **Auto-Brightness OFF**, brightness fixed (e.g. 50%); **screen on** (the
  app disables auto-lock).
- **Start battery 80–95%** — keeps Wh-per-% near the nominal constant (the charge curve is non-linear
  near full/empty).
- **Airplane-ish:** cellular off. For wireless `devicectl` driving, **Wi-Fi stays on** (treated as a
  small constant idle draw shared by every runtime); for launch-then-unplug, full Airplane Mode.
- **Thermal start uniform:** begin every run at `.nominal` (`run_device_bench.sh` cools ~20 s between
  runtimes); record `peakThermalState` and **room temperature** in the PR — a hot room throttles.
- **No other foreground apps**, background app refresh off, notifications quiet.

## What each metric means (and its honest limits)

Definitions: [`methodology/measurement.md`](../../methodology/measurement.md). Key ones:

- **Decode tok/s** `= generated_tokens / generate_time` (prefill excluded) — the fair cross-runtime
  headline on a fixed device. **TTFT** includes prefill, in ms. **n=3 median**, cold.
- **Peak RAM** = **`phys_footprint`** via Mach `task_info(TASK_VM_INFO)` — the figure **jetsam**
  charges (dirty + compressed + IOKit), sampled every 100 ms. Higher than `resident_size`, which
  under-reports compressed pages; pre-2026-06 runs used RSS and are not byte-comparable
  ([`methodology/memory.md`](../../methodology/memory.md)).
- **Energy** = battery-delta: `joules = Δpct × pack_Wh × 3600`, `pack_Wh = 16.5` for the measured
  `iPhone18,1` (US eSIM). **±1% battery quantization dominates the error bar** (a 4% drop ⇒ ≈ ±12.5%
  on joules) — compare runtimes *within* a device, never one device's joules to another's. Whole-
  system draw (display + Wi-Fi idle + OS), not chip-only.

### Bandwidth roofline = an **estimate**, flagged as such

The "why it's fast" tables report `effective GB/s = decode_tok/s × weight-bytes/token` (quant-scaled
per row, INT8 ≈ 2× INT4 bytes) against the chip's peak bandwidth. **The peak is a public-teardown
estimate, not an Apple figure** (`iPhone18,1` → 76.8 GB/s from LPDDR5X 9600 MT/s × 64-bit / 8). The
per-token byte count is likewise a model-structure estimate (Qwen3-0.6B ≈ 0.35 GB, Gemma-4-E2B ≈
0.79 GB at INT4). So **absolute GB/s carries that estimate; the same-device ordering is robust.**
Pinning the true peak bandwidth + a measured on-device STREAM/memcpy roofline is an obvious next step
and is on the to-do list.

## Fairness rules (summary)

Full text: [`methodology/fairness-rules.md`](../../methodology/fairness-rules.md).

1. Same prompt + same `maxTokens` (LiteRT-LM output is **capped at the 128-token budget** so it
   decodes the *same token count* as every runtime). 2. Cold vs warm reported separately. 3. Quant /
size / format / backend explicit per row. 4. **Failed runs stay in the table with their reason**
(see the [sustained hang](LITERT_SUSTAINED_HANG.md)). 5. Integration difficulty disclosed, not folded
into speed. 6. Never average across device classes. 7. Release builds (Debug flagged). 8. No
cherry-picking — median of n≥3 for variance-sensitive tasks. 9. Disclose charging / low-power /
thermal state. 10. Prefer the official runtime SDK and note its version.

**Disclosed, not equalised** (can't be, in any honest cross-runtime comparison): quantisation is each
runtime's native format; GPU (LiteRT/MLX/llama) vs ANE (CoreML) is a real backend difference;
LiteRT-LM's INT8 embedding table + Metal buffers make its footprint structurally higher than a
dynamic-KV runtime like MLX. We show these, we don't hide them.

## Reproduce the model-availability snapshot

```bash
python3 - <<'PY'
from huggingface_hub import list_models, list_repo_files
print([m.id for m in list_models(author="litert-community", limit=500)])
print(list_repo_files("litert-community/Qwen3-0.6B"))   # confirm the .litertlm bundle
PY
```

See [`MODEL_AVAILABILITY.md`](MODEL_AVAILABILITY.md) for the current inventory and which
model families are benchmarkable today.
