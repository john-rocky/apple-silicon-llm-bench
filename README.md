# Yardstick

**Apple Silicon AI Benchmark — Mac + iPhone + iPad.**

A neutral, reproducible benchmark for running local LLMs (and, in time, ASR / TTS) on Apple Silicon. Compares **MLX Swift, llama.cpp, CoreML (swift-transformers), MediaPipe / LiteRT-LM, ExecuTorch, ANEMLL** — and Apple's own Foundation Models — under real device constraints, not just `tok/s` on a server.

> Originally `ios-llm-benchmark`. Renamed in May 2026 once the harness grew to cover Mac as a first-class target alongside iPhone / iPad.

---

## 📊 Latest numbers — Apple M4 Max, short-chat (128 tokens, decode tok/s, median)

> One device, three runtimes, multiple models. Decode tok/s is the primary headline number; the full table (prefill, TTFT, peak memory, per-run audit trail) lives in [`RESULTS.md`](RESULTS.md). Read the [Headline observations](RESULTS.md#headline-observations-read-this-after-the-tables) section before drawing conclusions — the runtime ranking is **model-size-dependent**.

### Cross-runtime — same logical model, different backends (decode tok/s, median)

| Logical model | Params | n | mlx-swift (Q4) | llama.cpp (Q4_K_M) | coreml-llm |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 0.5B | 0.5 B | 3 | 116.8 | **292.3** | 180.7 (FP16) |
| Qwen 3.5 0.8B | 0.8 B | 3 | 81.3 | **192.0** | _chunked layout — upstream blocker_ |
| Qwen 3.5 2B   | 2 B   | 3 | 79.9 | **132.9** | _not run_ |
| Gemma 4 E2B   | 2 B   | 3 | 56.6 | **119.6** | 32.9 (INT4 palettized) |
| Gemma 4 E4B   | 4 B   | 1 | **45.1** | 40.7 | _not run_ |

→ llama.cpp Metal wins decode on every cell at or below 2 B params (1.7×–2.5× over MLX-Swift). At 4 B params (Gemma 4 E4B) **MLX-Swift overtakes llama.cpp**. The ranking is not a property of the runtime; it's a property of `(runtime, model, device)`.

### Cross-runtime — peak memory (MB, median)

The decode-tok/s table above hides the memory side. Same models, looking at peak working-set instead:

| Logical model | Params | mlx-swift | llama.cpp | coreml-llm |
|---|---:|---:|---:|---:|
| Qwen 2.5 0.5B | 0.5 B | **413** | 543 | 959 |
| Qwen 3.5 0.8B | 0.8 B | **618** | 754 | — |
| Gemma 4 E2B   | 2 B   | 2834 | 3182 | **1055** |
| Gemma 4 E4B   | 4 B   | **4417** | 5093 | — |

→ **"CoreML/ANE wins memory" is only true at the larger end** of this range. At 0.5 B params, MLX-Swift's working set (413 MB) is less than half of CoreML's (959 MB). The crossover where ANE residency starts paying off sits between 0.5 B and 2 B params on this device.

### Per-runtime model scaling

<sub>**llama.cpp** (Q4_K_M GGUF, M4 Max, short-chat)</sub>

| Model | Params | n | TTFT (ms) | Decode tok/s | Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 0.5B | 0.5 B | 1 | 120 | 292.3 | 543 |
| Qwen 3.5 0.8B | 0.8 B | 3 | 26  | 192.0 | 754 |
| Llama 3.2 1B  | 1.0 B | 3 | 25  | **285.9** | 1022 |
| Qwen 3.5 2B   | 2 B   | 3 | 31  | 132.9 | 1445 |
| Gemma 4 E2B   | 2 B   | 3 | 43  | 119.6 | 3182 |
| Gemma 4 E4B   | 4 B   | 1 | 162 | 40.7  | 5093 |

<sub>**mlx-swift** (Q4 / MLX, M4 Max, short-chat)</sub>

| Model | Params | n | TTFT (ms) | Decode tok/s | Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 0.5B | 0.5 B | 3 | 32 | 116.8 | 413 |
| Qwen 3.5 0.8B | 0.8 B | 3 | 52 | 81.3 | 618 |
| Qwen 3.5 2B   | 2 B   | 3 | 50 | 79.9 | 1243 |
| Gemma 4 E2B   | 2 B   | 3 | 100 | 56.6 | 2834 |
| Gemma 4 E4B   | 4 B   | 1 | 114 | 45.1 | 4417 |

<sub>**coreml-llm** (CoreML / ANE, M4 Max, short-chat)</sub>

| Model | Params | n | TTFT (ms) | Decode tok/s | Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|
| LFM 2.5 350M | 0.35 B | 1 | 383 | 58.9 | **98** |
| Qwen 2.5 0.5B | 0.5 B | 3 | 171 | 180.7 | 959 |
| Gemma 4 E2B  | 2 B    | 3 | 616 | 32.9 | **1055** |

→ CoreML/ANE trades throughput for memory: ~3× less peak working set than MLX-Swift at the same model size, at ~half the decode tok/s. **Lowest** per-byte footprint of any backend on this device.

**[Full results — by model, by runtime, full per-run audit trail →](RESULTS.md)**

---

## 🙋 Contributing a row

This table is the repo. **The easiest possible contribution is one new row.** All three of these are equally valuable:

1. **A new device.** Run the existing models on your iPhone / iPad / Mac. Tooling in [`Yardstick_USER_RUNS.md`](../Yardstick_USER_RUNS.md). The "Devices wanted" list at the bottom of [`RESULTS.md`](RESULTS.md#devices-wanted) is the shortlist.
2. **A new model.** Drop the model id into the [`ModelCatalog`](ios/BenchmarkApp/Sources/Models/ModelCatalog.swift) for the runtime that can load it.
3. **A new runtime.** Wire it up in [`ios/BenchmarkApp/Sources/Runtimes/`](ios/BenchmarkApp/Sources/Runtimes/) following the `LLMRuntime` protocol; the harness will pick it up.

Workflow once you have the build set up:

```sh
# 1. Run 3 times to get a stable median:
for run in 1 2 3; do
  yardstick run --task short-chat \
                --runtime mlx-swift \
                --model <id-or-hf-repo> \
                --output results/raw/<device>-<runtime>-<model>-short-chat-run${run}.jsonl
done

# 2. Regenerate the tables — they're auto-built from JSONL:
python scripts/render_results.py

# 3. Commit the JSONLs + the updated RESULTS.md, open a PR.
```

CI runs `python scripts/render_results.py --check` on every PR — it fails if the JSONLs and the tables disagree, so the human-edited section of RESULTS.md cannot drift out of sync with the raw data.

Full step-by-step (build, model picker, device-specific gotchas) lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## What gets measured

Per `(runtime, model, device, build)` tuple:

- **Speed** — TTFT, prefill `tok/s`, decode `tok/s`, sustained-decode drift over 512+ tokens.
- **Memory** — baseline, peak during decode, after-generation.
- **Thermal** — initial / peak / final state across the run.
- **Energy** — joules per token where the battery-step API gives a useful signal.
- **Lifecycle** — survives background → foreground, cancellation latency, streaming.
- **Quality** *(roadmap)* — WER / CER for ASR, perplexity / MMLU for LLM, byte-identical comparison vs Python references.

Methodology lives under [`methodology/`](methodology/). The numbers we publish follow [`methodology/fairness-rules.md`](methodology/fairness-rules.md).

## Project shape

```
Yardstick/
├── Package.swift              SPM: YardstickKit library + `yardstick` Mac CLI
├── apple/
│   └── YardstickCLI/          Mac command-line runner
├── ios/
│   └── BenchmarkApp/          On-device iOS app (`.xcodeproj`)
├── runtimes/                  Per-runtime notes (adapters, gotchas, version pins)
├── devices/                   Per-device pages (chip, RAM, OS, build, signing)
├── methodology/               How we measure each axis fairly
├── models/                    Curated model catalog
├── prompts/                   Standardized prompts per task
└── results/
    ├── raw/                   JSONL dumps per run
    └── (summary tables generated into RESULTS.md)
```

## Running on Mac (CLI)

> **Current status (May 2026)**: SPM build is clean. Runtime is blocked by [`ml-explore/mlx-swift#349`](https://github.com/ml-explore/mlx-swift/issues/349) — the MLX Metal kernel bundle isn't emitted by `swift build` from a downstream package, so `swift run yardstick run …` exits with `Failed to load the default metallib`. The same workaround applies to `mlx-swift-examples/llm-tool` (its README says "Build the llm-tool scheme in Xcode"). A macOS app target that wraps the CLI through Xcode's Metal toolchain is queued as Phase 2.

When the Phase-2 macOS target lands, this is the intended shape:

```sh
$ yardstick list
$ yardstick run --task short-chat \
                --runtime mlx-swift \
                --model mlx-community/Qwen3-0.6B-4bit \
                --output results/raw/m4max-mlx-qwen3-0.6b.jsonl
```

For now, build verification only:

```sh
$ swift build       # Build complete!
```

## Running on iPhone (app)

```sh
cd ios/BenchmarkApp
./scripts/bootstrap.sh           # downloads llama.xcframework + Anemll source
open BenchmarkApp.xcodeproj      # set your Team in Signing & Capabilities
                                 # ⌘R on a connected iPhone
```

First launch downloads the chosen model (default: `mlx-community/gemma-4-e2b-it-4bit`, ~1.3 GB) into the app's Documents directory. Use the picker to swap.

| Runtime | Adapter | Wire-up |
|---|---|---|
| MLX Swift | `MLXRuntime.swift` | SPM (`mlx-swift-lm`) |
| llama.cpp | `LlamaCppRuntime.swift` | vendored `llama.xcframework` (`bootstrap.sh`) |
| CoreML (swift-transformers) | `CoreMLRuntime.swift` | SPM (`swift-transformers` `Models` + `Generation`) |
| MediaPipe / LiteRT-LM | `MediaPipeRuntime.swift` | `canImport`-gated; add `paescebu/SwiftTasksGenAI` via Xcode UI |
| ExecuTorch | `ExecuTorchRuntime.swift` | SPM (`pytorch/executorch` `swiftpm-*` branch) |
| ANEMLL | `AnemllRuntime.swift` | local SPM via vendored `Anemll/` (`bootstrap.sh`) |

Adapters whose framework isn't present at build time are gated with `#if canImport(...)` and fall back to a clear "not added" error rather than failing the build.

## Devices

Verified in-tree:

- [`devices/mac-m4-max.md`](devices/mac-m4-max.md) — Apple M4 Max (macOS 26)
- [`devices/macbook-air-m3.md`](devices/macbook-air-m3.md) — MacBook Air M3, 16 GB (macOS 26)
- [`devices/iphone-17-pro.md`](devices/iphone-17-pro.md) — iPhone 17 Pro (iOS 26)

**Community devices wanted.** If you have an Apple Silicon device not listed above, the fastest way to contribute a row to `RESULTS.md` is to:

1. Add a `devices/<your-device>.md` describing the hardware/OS/build.
2. Run the app or CLI per [`methodology/measurement.md`](methodology/measurement.md).
3. PR the resulting `results/raw/<device>-*.jsonl` and the updated `RESULTS.md` rows.

Devices we'd love numbers for:

- iPhone 15 Pro / 16 Pro / 17 Pro Max / 17 Air
- iPad Pro M2 / M4
- MacBook Pro M1 / M2 / M3 / M4 (Pro / Max)
- Mac Studio Ultra (M2 Ultra / M3 Ultra)
- Mac mini M2 / M4

## Backend status on Mac

| Backend | Build on Mac | Run on Mac | Notes |
|---|:---:|:---:|---|
| MLX Swift LM | ✅ | ✅ | Native SPM macOS. The Xcode-built tool target sidesteps mlx-swift#349. |
| llama.cpp | ✅ | ✅ | `macos-arm64_x86_64` slice in `Vendored/llama.xcframework`. CLI uses `LD_RUNPATH_SEARCH_PATHS` to resolve the framework at runtime. |
| CoreML (CoreMLLLM) | ✅ | ✅ (some models) | macOS 15+. Models with the single-top-level `.mlpackage` layout (e.g. LFM 2.5 350M) auto-download from HF and run; the chunked / multi-`.mlpackage` repos (e.g. `mlboydaisuke/qwen3.5-0.8B-CoreML`) need upstream `CoreMLLLM` work to load. |
| ExecuTorch | ✅ | ⏸ | Build path is clean; current ET-community models ship SentencePiece `tokenizer.model` but ET's `hf_tokenizer.cpp` expects HF-format `tokenizer.json`. Needs a model with HF tokenizer or an ET-side SentencePiece adapter. |
| ANEMLL | ✅ | ⏸ | Build path is clean; `swift-huggingface.HFDownloader` fails on `.mlmodelc/` directory-shaped HF repos. Needs upstream downloader work. |
| MediaPipe / LiteRT-LM | ⛔ | ⛔ | `paescebu/SwiftTasksGenAI 0.10.24` ships only `ios-arm64` slices — no `macos-arm64*`. Blocked upstream. |

## Roadmap

- **Phase 1** — repo rename, top-level SPM (`YardstickKit` + `yardstick` CLI), Mac CLI builds clean, README + device pages, methodology docs, iOS app intact.
- **Phase 2** — Mac CLI runs end-to-end (via Xcode-built target to sidestep mlx-swift #349), first M4 Max numbers committed to `RESULTS.md`.
- **Phase 2.5** — All 5 buildable backends (MLX, llama.cpp, CoreML, ExecuTorch, ANEMLL) wired into the Mac tool target; first cross-backend row (Gemma 4 E2B: MLX vs llama.cpp).
- **Phase 3** *(in progress)* — fill remaining adapter row gaps (downloader + model-format work, mostly upstream), MacBook Air M3 + iPhone 17 Pro numbers via `[Yardstick_USER_RUNS.md](../Yardstick_USER_RUNS.md)`.
- **Phase 4** — quality / accuracy tasks: WER + CER (reusing `swift-transformers` Whisper normalizer), perplexity, MMLU subset. ASR + TTS adapters (WhisperKit, Apple Speech, system TTS).
- **Phase 5** — public results dashboard, regeneration CI, comparison plots.

## License

MIT, see [`LICENSE`](LICENSE).
