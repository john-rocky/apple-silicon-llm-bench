# Project details

> Repo: `apple-silicon-llm-bench` · CLI/brand: `yardstick`. Started life as `ios-llm-benchmark` — iPhone is still the headline target, now measured alongside iPad and Mac.

## Project shape

```
Yardstick/
├── Package.swift              SPM: YardstickKit library + `yardstick` Mac CLI
├── apple/
│   └── YardstickCLI/          Mac command-line runner
├── ios/
│   └── BenchmarkApp/          On-device iOS app (`.xcodeproj`)
├── android/                   adb-driven Android runs (LiteRT-LM, llama.cpp; Pixel 8a)
├── runtimes/                  Per-runtime notes (adapters, gotchas, versions)
├── devices/                   Per-device pages (chip, RAM, OS, build, signing)
├── methodology/               How we measure each axis fairly
├── matrices/                  Standing matrix files (./bench matrix|regress)
├── models/                    Curated model catalog
├── prompts/                   Standardized prompts per task (text/ = canonical bytes)
└── results/
    ├── raw/                   JSONL dumps per run
    ├── summary/               machine-readable accumulation (build_summary.py)
    ├── regression-reports/    release-over-release verdicts (regression_report.py)
    └── (tables generated into RESULTS.md + LEADERBOARD.md)
```

## Running on Mac: notes

> **Status (July 2026)**: the SPM CLI runs end-to-end — no Xcode target required. mlx-swift#349 (the MLX Metal bundle not being emitted by `swift build` from a downstream package) is resolved on the pinned **mlx-swift 0.31.3** with current Swift: `swift build` now emits `mlx-swift_Cmlx.bundle` (carrying `default.metallib`) next to the `yardstick` binary, so `swift run yardstick run …` loads MLX and runs. Build **Release** for real numbers — a Debug build adds large per-token host overhead and understates decode ([fairness-rules #7](../methodology/fairness-rules.md)).

> If a Release build ever fails with `unable to spawn … Metal.xctoolchain/usr/bin/metal (No such file or directory)`, the on-demand Metal toolchain mount changed (a reboot or Xcode update remounts it under a new path) and SwiftPM's cached build manifest still points at the old one. Clear the manifest and rebuild — `rm -rf .build/out/Intermediates.noindex/XCBuildData` — which re-derives the current toolchain path (a full `swift package clean` also works, but recompiles everything). Only re-download the toolchain (`xcodebuild -downloadComponent MetalToolchain`) if the asset itself is gone. Unrelated to the harness.

## Running on iPhone: adapters

| Runtime | Adapter | Wire-up |
|---|---|---|
| MLX Swift | `MLXRuntime.swift` | SPM (`mlx-swift-lm`) |
| llama.cpp | `LlamaCppRuntime.swift` | vendored `llama.xcframework` (`bootstrap.sh`) |
| CoreML (swift-transformers) | `CoreMLRuntime.swift` | SPM (`swift-transformers` `Models` + `Generation`) |
| LiteRT-LM | `MediaPipeRuntime.swift` | SPM (`google-ai-edge/LiteRT-LM` ≥ 0.13, product `LiteRTLM`); `#if canImport(LiteRTLM)`-gated |
| ExecuTorch | `ExecuTorchRuntime.swift` | SPM (`pytorch/executorch` `swiftpm-*` branch) |
| ANEMLL | `AnemllRuntime.swift` | local SPM via vendored `Anemll/` (`bootstrap.sh`) |
| Apple Foundation Models | `AppleFMRuntime.swift` | system framework, `#if canImport(FoundationModels)` (macOS 26 / iOS 26) |

Adapters whose framework isn't present at build time are gated with `#if canImport(...)` and fall back to a clear "not added" error rather than failing the build.

## Devices

Verified in-tree:

- [`devices/mac-m4-max.md`](../devices/mac-m4-max.md) — Apple M4 Max (macOS 26)
- [`devices/macbook-air-m3.md`](../devices/macbook-air-m3.md) — MacBook Air M3, 16 GB (macOS 26)
- [`devices/iphone-17-pro.md`](../devices/iphone-17-pro.md) — iPhone 17 Pro (iOS 26)

**Community devices wanted.** If you have an Apple Silicon device not listed above, the fastest way to contribute a row to `RESULTS.md` is to:

1. Add a `devices/<your-device>.md` describing the hardware/OS/build.
2. Run the app or CLI per [`methodology/measurement.md`](../methodology/measurement.md).
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
| MLX Swift LM | ✅ | ✅ | Native SPM macOS — `swift run yardstick` runs end-to-end. mlx-swift 0.31.3 emits the Metal bundle next to the CLI binary, so the old #349 Xcode-target workaround is no longer needed. |
| llama.cpp | ✅ | ✅ | `macos-arm64_x86_64` slice in `Vendored/llama.xcframework`. CLI uses `LD_RUNPATH_SEARCH_PATHS` to resolve the framework at runtime. |
| CoreML (CoreMLLLM) | ✅ | ✅ (some models) | macOS 15+. Models with the single-top-level `.mlpackage` layout (e.g. LFM 2.5 350M) auto-download from HF and run; the chunked / multi-`.mlpackage` repos (e.g. `mlboydaisuke/qwen3.5-0.8B-CoreML`) need upstream `CoreMLLLM` work to load. |
| ExecuTorch | ✅ | ⏸ | Build path is clean; current ET-community models ship SentencePiece `tokenizer.model` but ET's `hf_tokenizer.cpp` expects HF-format `tokenizer.json`. Needs a model with HF tokenizer or an ET-side SentencePiece adapter. |
| ANEMLL | ✅ | ⏸ | Build path is clean; `swift-huggingface.HFDownloader` fails on `.mlmodelc/` directory-shaped HF repos. Needs upstream downloader work. |
| LiteRT-LM | ✅ | ✅ | `google-ai-edge/LiteRT-LM` ships `ios-arm64` + `macos-arm64` slices, wired via SPM (product `LiteRTLM`, macOS 12+). Measured on an M4 Max on v0.16.0 ([all headline tables](headline-tables.md)). Watch the package's `-all_load` for duplicate-symbol clashes with the vendored `llama`/`executorch` static libs (fall back to scoped `-force_load`). |

## Operating this repo continuously

**The standing-benchmark harness now lives in its own repo:
[edge-llm-bench](https://github.com/john-rocky/edge-llm-bench)** — carved out
2026-08-17 with the v0.15.0 → v0.16.0 regression run as its founding
baseline. That repo is canonical for the harness going forward; this one
remains the measurement archive (published tables, full raw history, the
`./reproduce` registry). The harness copy here still works — runbooks in
[`docs/OPERATIONS.md`](OPERATIONS.md), entrypoint `./bench`, standings
in [`LEADERBOARD.md`](../LEADERBOARD.md).

## Roadmap

- **Phase 1** — repo rename, top-level SPM (`YardstickKit` + `yardstick` CLI), Mac CLI builds clean, README + device pages, methodology docs, iOS app intact.
- **Phase 2** — Mac CLI runs end-to-end via plain SPM (`swift run yardstick`; mlx-swift #349 resolved on 0.31.3 — no Xcode-target workaround), first M4 Max numbers committed to `RESULTS.md`.
- **Phase 2.5** — All 5 buildable backends (MLX, llama.cpp, CoreML, ExecuTorch, ANEMLL) wired into the Mac tool target; first cross-backend row (Gemma 4 E2B: MLX vs llama.cpp).
- **Phase 3** *(in progress)* — fill remaining adapter row gaps (downloader + model-format work, mostly upstream), MacBook Air M3 + iPhone 17 Pro numbers.
- **Phase 4** — quality / accuracy tasks: WER + CER (reusing `swift-transformers` Whisper normalizer), perplexity, MMLU subset. ASR + TTS adapters (WhisperKit, Apple Speech, system TTS). Scores are reported per task, not as one quality number: [quantcost](https://github.com/Ani-2003-HD/quantcost) found Qwen2.5-1.5B (llama.cpp) breaking on GSM8K at Q3 and on TriviaQA at Q2 while its JSON-schema conformance stayed at 100%.
- **Phase 5** — public results dashboard, regeneration CI, comparison plots.
