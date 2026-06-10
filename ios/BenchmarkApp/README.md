# BenchmarkApp

The iOS app that drives `ios-llm-benchmark` runs.

## What it does

- Picks a runtime, model, and task.
- Loads the model (downloading from HuggingFace on first run, caching to the app's Documents directory).
- Runs the selected task with timing, memory, and thermal sampling.
- Persists each result as JSON in `Documents/results/`.
- Lets you export each result via the iOS share sheet.

## Build

```bash
./scripts/bootstrap.sh          # fetch llama.xcframework + Anemll source
open BenchmarkApp.xcodeproj
```

In Xcode:

1. Select the `BenchmarkApp` target.
2. Set your Apple Developer **Team** under Signing & Capabilities.
3. Select your physical iPhone as the run destination. (Simulator builds work for compile validation only — never for performance numbers; see [`../../methodology/fairness-rules.md`](../../methodology/fairness-rules.md).)
4. ⌘R.

First build resolves SwiftPM dependencies (`mlx-swift-lm`, `swift-huggingface`, `swift-transformers`, `executorch`, `Anemll`, `CoreMLLLM`). Takes several minutes.

### What `bootstrap.sh` does

1. Downloads `llama.xcframework` (~168 MB) from llama.cpp release `b8999` into `Vendored/`.
2. Clones `https://github.com/Anemll/Anemll.git` into `Vendored/Anemll/` and patches its `swift-transformers` pin to a 1.x range (its `branch:main` pin conflicts with `CoreMLLLM`).
3. (If `xcodegen` is installed and `REGEN_XCODEPROJ=1` is set, regenerates `BenchmarkApp.xcodeproj` from `project.yml`. Most users skip this — the `.xcodeproj` is committed.)

The `Vendored/` directory is git-ignored and re-fetched on demand.

### LiteRT-LM

Wired via SPM — `xcodegen generate` picks it up from `project.yml`
(`google-ai-edge/LiteRT-LM` ≥ 0.12.0, product `LiteRTLM`, macOS 12 / iOS 15).
The adapter (`MediaPipeRuntime.swift`) is `#if canImport(LiteRTLM)`-gated, so it
lights up automatically once the package resolves; no Xcode-UI or CocoaPods step.

If the link fails with duplicate symbols, the package's `-all_load` is colliding
with the vendored `llama`/`executorch` static libs — swap it for a scoped
`-force_load <path-to-libLiteRTLM.a>` in `OTHER_LDFLAGS`.

## Architecture

```
Sources/
├── BenchmarkApp.swift        @main, AppSession (runtimes + history)
├── Info.plist
├── Assets.xcassets/
├── Models/                   data structures persisted in result JSON
├── Runtimes/
│   ├── LLMRuntime.swift      protocol every adapter conforms to
│   ├── MLXRuntime.swift      MLX Swift LM implementation
│   └── StubRuntime.swift     placeholder for not-yet-wired runtimes
├── Benchmark/
│   ├── BenchmarkRunner.swift orchestrates a single run
│   ├── BenchmarkTask.swift   protocol + catalog
│   ├── ResultStore.swift     JSON persistence
│   ├── MemoryMonitor.swift   Mach task_info sampling
│   ├── ThermalMonitor.swift  ProcessInfo.thermalState sampling
│   └── Tasks/                concrete task definitions (A/B/C/D)
└── Views/
    ├── RootView.swift        TabView shell
    ├── RunView.swift         pick + run + live partial output
    ├── ResultDetailView.swift breakdown + share-sheet export
    ├── HistoryView.swift     list of saved runs
    └── AboutView.swift       device + pre-flight checklist
```

Adding a new runtime is a single new file under `Runtimes/`: implement `LLMRuntime`, then list its `RuntimeKind` case in the picker. The runner doesn't need to change.

## Adding a model

Edit `Models/ModelCatalog.swift` and append a `ModelInfo`. The MLX runtime only requires the HuggingFace `id` to be a valid `mlx-community/...` repo; everything else is metadata for the UI and result row.

## Where results go

- On-device: `Documents/results/<runtime>_<modelId>_<task>_<timestamp>.json`
- Exported: tap the share icon on a result detail screen → AirDrop / Files / Mail
- Repository: drop the JSON into `../../results/raw/` and open a PR

## Known gotchas

- **First run downloads ~400 MB** for the default `Qwen3-0.6B-4bit` model. Use Wi-Fi.
- **First generation is slower than later ones** because Metal compiles shaders on demand. The `coldRun` flag in the result captures this.
- **Backgrounding the app during a run** may pause generation depending on iOS version. We have not yet wired up a `BackgroundTaskAssertion` — generation may be cut off after ~30 s in background.
