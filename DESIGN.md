# DESIGN — iOS On-device LLM Benchmark

This document is the long-form spec. The [README](README.md) is the marketing pitch; this is the contract.

## 1. Purpose

This project provides a practical benchmark suite for running local LLMs on iPhone and Apple platforms.

The goal is **not** to prove that one runtime is universally better than another. The goal is to answer:

> "If I want to ship an on-device LLM inside an iOS app, which runtime/model combination is actually usable?"

The benchmark focuses on real app constraints: decode speed, prefill speed, memory, app/package size, model loading time, thermal behavior, stability during long generation, ease of integration, and App Store / TestFlight practicality.

Audience: iOS developers, mobile AI engineers, researchers, and companies evaluating local LLM deployment.

## 2. Target runtimes

### Required initial targets

| Runtime          | Focus                                  |
|------------------|----------------------------------------|
| CoreML-LLM       | Apple Neural Engine behavior           |
| MLX Swift / MLX Swift LM | Apple GPU / unified memory     |
| llama.cpp        | GGUF / Metal backend, broad model coverage |
| LiteRT-LM / MediaPipe LLM | Google AI Edge mobile stack   |

### Optional future targets

- **ExecuTorch** (CoreML backend) — PyTorch mobile inference
- **ANEMLL** — another ANE-focused baseline
- **ONNX Runtime CoreML EP** — broader model deployment comparison

Each runtime gets a one-page summary in [`runtimes/`](runtimes/) covering integration cost, model formats it accepts, and known iOS gotchas.

## 3. Target devices

Initial priority: **iPhone 17 Pro, iPhone 16 Pro, iPhone 15 Pro**, plus an M-series Mac for comparison.

Every result must include:

- device name and chip
- iOS version
- available memory (if measurable)
- thermal state before the test
- battery / charging state
- low-power-mode status
- build configuration: Debug / Release / TestFlight / App-Store-equivalent Release

Simulator results may appear as "build validation only" — never as performance comparisons.

## 4. Target models

Initial candidates (small enough to fit in iPhone memory, broadly available):

- Gemma 4 E2B / Gemma 3 1B QAT
- Qwen 3 0.6B / 1.7B / 4B
- SmolLM2 / SmolLM3 (135M to 3B class)
- Llama 3.2 1B / 3B
- Phi 3.5 mini

Each model entry must record:

- original model name and parameter count
- source format and runtime format (`.mlpackage`, `.mlmodelc`, `.gguf`, MLX safetensors, LiteRT format)
- quantization (FP16, INT8, INT4, mixed)
- on-disk size: model + tokenizer + config
- supported context length

> **Important:** never compare different quantization levels as if they were the same model. A 4-bit GGUF and an FP16 CoreML model are different deployment profiles, not different runtime speeds.

## 5. Benchmark tasks

Four fixed tasks. Full prompts in [`prompts/`](prompts/).

### Task A — Short chat response

Measures normal assistant-style latency.

- Prompt: `Explain what on-device AI means in simple terms.`
- Output: 128 tokens
- Metrics: TTFT, decode tok/s, total time, peak memory, post-run thermal state

### Task B — Long-context prefill

Measures prompt ingestion / prefill performance.

- Input: fixed 2,000-token prompt (optionally 4,000 if the runtime supports it)
- Output: 64 tokens
- Metrics: prefill time, prefill tok/s, TTFT, decode tok/s after prefill, peak memory

### Task C — Sustained generation

Measures thermal stability and long-running behavior.

- Prompt: `Write a detailed explanation of how local LLM inference works on mobile devices.`
- Output: 512 tokens (optionally 1,024)
- Metrics: average decode tok/s, minimum decode tok/s, speed drop over time, thermal state changes, peak memory, completion / crash / hang

### Task D — App-style interaction loop

Simulates real app usage. This task often exposes runtimes that look fast on a single benchmark but fall over in real lifecycle conditions.

Flow:

1. Load model
2. Ask short question
3. Ask another question with prior history
4. Cancel generation midway
5. Start a new generation
6. Background the app
7. Foreground the app
8. Generate again

Metrics: model load time, session recovery, cancellation success, memory after repeated generations, stability after background/foreground, runtime errors.

## 6. Core metrics

### Performance

- model load time
- first-run warmup time (if a separate warmup pass is used)
- prefill tokens/sec
- decode tokens/sec
- time to first token
- total generation time
- tokens generated

### Memory

- baseline app memory
- memory after model load
- memory during prefill
- peak memory during decode
- memory after generation
- memory after model unload (if supported)

### Thermal

- thermal state before
- thermal state after
- thermal state during sustained generation
- decode-rate degradation curve once the device gets hot

### Binary and storage

- app binary size
- model file size
- tokenizer / config size
- total installed size
- whether the model can be downloaded after install vs. bundled

### Integration practicality (scored separately)

- Swift API quality
- documentation quality
- build / model-conversion ease
- streaming output support
- cancellation support
- background / foreground behavior
- App Store practicality
- license clarity

## 7. Result table format

The compact summary table (see README) is the canonical view. A more human-readable recommendations section follows beneath it. See [methodology/measurement.md](methodology/measurement.md) for column definitions.

## 8. Fairness rules

1. **Same prompt, same token budget.** All runtimes use the same prompts and output token limits where possible.
2. **Cold and warm runs reported separately.** Cold = app launched, model loaded, first generation. Warm = model already loaded.
3. **Quantization is explicit.** Every result row shows model size, quantization, runtime format, and backend.
4. **Failed runs stay in the table.** OOM, crash, or unsupported configuration is information.
5. **Don't hide integration difficulty.** A fast-but-unshippable runtime gets described that way.

Full text in [methodology/fairness-rules.md](methodology/fairness-rules.md).

## 9. Repository structure

See [docs/project.md](docs/project.md#project-shape).

## 10. First release scope (v0.1)

- Benchmark purpose and methodology
- Fixed prompts
- Result table format
- iOS app that runs Tasks A and C against MLX Swift on at least one iPhone
- At least one comparison runtime in design (llama.cpp wired as next priority)
- Honest TODOs for everything else

We do **not** wait for every runtime to be measured before publishing.

## 11. What makes this benchmark different

Most LLM benchmarks focus on server-side inference or academic quality metrics. This benchmark focuses on **mobile shipping reality**:

- Can it run on an actual iPhone?
- Can it survive repeated generations?
- Can it stream tokens into a real app UI?
- Can it be cancelled?
- Can it survive background/foreground transitions?
- Does it overheat?
- Is the model package size realistic?
- Is the runtime easy to integrate into Swift?
- Is it realistic for TestFlight or App Store deployment?

Positioned as: *a practical benchmark for engineers who want to ship local LLMs inside iOS apps.*

## 12. Non-goals

- Not a general LLM quality benchmark (use MMLU, GSM8K, MT-Bench, etc.)
- Does not rank model intelligence
- Does not claim one runtime is universally best
- Does not benchmark server inference
- Does not focus on Android in the initial version
- Does not provide legal advice about model licenses

## 13. Scoring system

A simple practical score can be useful as a tiebreaker, but it is **not** the headline result.

| Category               | Weight |
|------------------------|-------:|
| Decode speed           |    25% |
| Memory efficiency      |    20% |
| Thermal stability      |    15% |
| Model size practicality|    15% |
| Swift integration      |    15% |
| App lifecycle stability|    10% |

Raw numbers always come first. The score is a convenience summary only.

## 14. Beyond the original spec

A few extensions that go beyond the v0 design:

- **Energy per token** ✅ — implemented via `UIDevice.batteryLevel` delta + per-device battery-pack capacity lookup. See [methodology/energy.md](methodology/energy.md). Most meaningful on the 512-token sustained-generation task; a short chat run is too small to register a 1% battery step.
- **TTFT-after-idle** — TTFT measured 60 s after the previous generation, to expose runtimes that need re-warming.
- **Cancellation latency** — wall-clock between calling `cancel()` and the runtime actually releasing the GPU/ANE.
- **First-cold-launch latency** — time from app launch to "model ready", because for many apps this is the user-facing number that matters most.
- **Streaming visual smoothness** — sample inter-token wall-clock variance; jittery streams feel worse than a steady slower stream.
- **Repeat-N stability** — run Task A 20× in a row and report variance; runtimes whose tail latency degrades are hard to ship.
- **Watchdog instrumentation** — log if a generation thread blocks the main thread or trips the iOS 0x8badf00d watchdog.

These are tracked as TODOs and will be added once the v0 baseline exists.
