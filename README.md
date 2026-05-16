# Yardstick

**Apple Silicon AI Benchmark — Mac + iPhone + iPad.**

A neutral, reproducible benchmark for running local LLMs (and, in time, ASR / TTS) on Apple Silicon. Compares **MLX Swift, llama.cpp, CoreML (swift-transformers), MediaPipe / LiteRT-LM, ExecuTorch, ANEMLL** — and Apple's own Foundation Models — under real device constraints, not just `tok/s` on a server.

> Originally `ios-llm-benchmark`. Renamed in May 2026 once the harness grew to cover Mac as a first-class target alongside iPhone / iPad.

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
