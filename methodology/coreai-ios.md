# Core AI methodology — iPhone (the official Apple path)

How the **Apple Core AI** rows are produced. Core AI is the Core ML successor
announced at WWDC 2026 (iOS / macOS 27). This benchmark uses Apple's own export
pipeline and Swift runtime end-to-end.

## The model

**Qwen3-0.6B**, Apple's headline example in the
[`apple/coreai-models`](https://github.com/apple/coreai-models) catalog. Two
exports are used, because **on iPhone the compute unit is decided by the export
shape, not a runtime flag** (see below):

| Export | Command | Shape | Engine it lands on |
| --- | --- | --- | --- |
| **iOS / static** | `coreai.llm.export qwen3-0.6b --platform iOS` | fixed ctx 4096, mixed 4/8-bit | **ANE** (`static-shape`) |
| **dynamic** | `coreai.llm.export qwen3-0.6b --platform macOS` | dynamic ctx, INT4 | **GPU** (`coreai-pipelined`) |

## iOS requires AOT compilation (the load-time gotcha)

An exported `.aimodel` is **MLIR IR** (`main.mlirb`, `compilation.targets: []`).
macOS JIT-compiles it at load; **iOS cannot JIT** — loading the raw IR fails with
`NSPOSIXErrorDomain Code=2 "No such file or directory"`. The model must be
**ahead-of-time compiled** for the device:

```bash
xcrun coreai-build compile qwen3_0_6b_ios.aimodel \
    --platform iOS --preferred-compute neural-engine --output out/   # or: gpu
# → out/qwen3_0_6b_ios.<arch>.aimodelc  (one per GPU family; h18p = iPhone 17 Pro)
```

Then assemble a loadable bundle: copy the device-arch `.aimodelc` + the
`tokenizer/` next to a `metadata.json` whose `assets.main` points at the
compiled file (`qwen3_0_6b_ios.h18p.aimodelc`), per `models/README.md#compiled-models`.
`scripts/bench_coreai_iphone.sh` does export → compile → assemble → side-load.

## The runtime — Apple's `CoreAILM` Swift package

`CoreAIRuntime` ([`Sources/Runtimes/CoreAIRuntime.swift`](../ios/BenchmarkApp/Sources/Runtimes/CoreAIRuntime.swift))
drives the model through `EngineFactory` (the same low-level path as Apple's
`llm-benchmark` tool). The engine is **auto-detected from the model structure**:
`EngineFactory` maps a single dynamic `main` graph → the `coreai-pipelined` GPU
engine, and a static (chunked) graph → the `static-shape` ANE engine. So the two
catalog ids point at the two compiled bundles, and the runtime lets the structure
pick the engine:

| Catalog id | Bundle | Engine (auto) | Compute |
| --- | --- | --- | --- |
| `core-ai/qwen3-0.6b-ane` | static, ANE-compiled | `static-shape` | Neural Engine |
| `core-ai/qwen3-0.6b-gpu` | dynamic, GPU-compiled | `coreai-pipelined` | GPU |

Greedy decoding (`temperature: 0`), the bundle's embedded tokenizer + chat
template, measured by the same `BenchmarkRunner`/tasks/metrics as every other
runtime — so it is directly comparable to CoreML-LLM (ANE) and MLX (GPU).

## Results (iPhone 17 Pro, Qwen3-0.6B, short-chat)

| Engine | Compute | Decode tok/s | TTFT (warm) | Peak MB |
| --- | --- | ---: | ---: | ---: |
| Core AI GPU (`coreai-pipelined`) | GPU | **~180 warm** / 71 cold | ~26 ms | ~524 |
| MLX | GPU | ~115 | ~57 ms | 539 |
| Core AI ANE (`static-shape`) | ANE | ~50 | ~63 ms | ~1166 |
| CoreML-LLM | ANE | ~39 | ~548 ms | **~184** |

**Cold vs warm matters for Core AI GPU.** The pipelined engine pays a heavy
first-run cost (kernel compilation + filling its 3-deep pipeline) — ~71 tok/s on
the cold run, then **~180 tok/s** once warm, the fastest of the four. MLX is flat
cold-to-warm (~115). The static-shape ANE path is steady (~50) with the lowest
TTFT spread; CoreML-LLM is the memory champion (~184 MB, ~6× leaner than Core AI
ANE) at the cost of decode speed and a slower recurrent prefill.

## Mac scaling (matched) — the GPU lead shrinks

The iPhone GPU lead (Core AI 1.6× MLX at 0.6B) is a small-model effect. On M4 Max
with matched params (512-token prompt, 512 gen, greedy, 4-bit, warm) — Core AI via
Apple's `llm-benchmark`, MLX via `mlx_lm`:

| Model | Core AI GPU | MLX | lead |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1121 | 455 | 2.47× |
| Qwen3-8B | 94 | 90 | 1.05× |

Tiny models are dominated by per-token dispatch overlap (Core AI's strength); at a
realistic 8B both are memory-bandwidth-bound and converge to a near-tie. Chart:
`docs/charts/mac_coreai_scaling.png` (`scripts/coreai_mac_scaling_chart.py`).

## Requirements & reproduction

- macOS 26.4+ / Xcode 27 (+ `coreai-core`) to export & compile; iPhone on iOS 27 to run.
- The BenchmarkApp iOS target is bumped to **iOS 27** (the `coreai-models`
  package floor); the `coreai-models` Swift package is symlinked at
  `ios/BenchmarkApp/Vendored/coreai-models`. ExecuTorch is dropped from the iOS
  app while Core AI is linked (its `executorch.xcframework` and Core AI's
  `CXGrammar.xcframework` both emit `include/module.modulemap`).
- Full device run: [`scripts/bench_coreai_iphone.sh`](../scripts/bench_coreai_iphone.sh).
  Note: drive launches **without `--console`** (detached) from any non-interactive
  shell — `--console` fails to attach (CoreDeviceError 10002) when backgrounded.
