# RESULTS

Real on-device / on-Mac measurements. Each row corresponds to a `(runtime, model, device, build)` tuple. Raw JSON dumps live in [`results/raw/`](results/raw/); the tables here are generated summaries.

> **Coverage status (2026-05-16)**: The first Mac rows are in. iPhone 17 Pro and MacBook Air M3 are queued — see the [Devices we'd love numbers for](README.md#devices) list. PRs that add a `results/raw/<device>-*.jsonl` and a row here are warmly welcomed.

## Format

| Column          | Definition |
|-----------------|------------|
| Runtime         | mlx-swift / llama.cpp / coreml-llm / litert-lm / executorch / anemll |
| Model           | HF repo id or original name |
| Device          | e.g. `iPhone 17 Pro`, `Mac M4 Max` |
| OS              | iOS / macOS version |
| Build           | Debug / Release / TestFlight / App-Store-equivalent Release |
| Quant           | FP16 / Q8 / Q4 / mixed |
| Load            | seconds, cold model load on already-downloaded weights |
| TTFT            | milliseconds, time to first generated token |
| Prefill tok/s   | prompt-processing throughput |
| Decode tok/s    | generation throughput |
| Peak Mem        | MB, peak resident memory during generation |
| Thermal         | nominal / fair / serious / critical (peak state seen) |
| Notes           | failures, caveats, special configuration |

## Task A — Short chat (128 tokens)

Prompt: `"Explain what on-device AI means in simple terms."` (greedy decode, temperature 0.0, max-tokens 128).

| Runtime    | Model                                      | Device     | OS         | Build | Quant | Load (s) | TTFT (ms) | Prefill tok/s | Decode tok/s | Peak Mem (MB) | Thermal | Notes |
|------------|--------------------------------------------|------------|------------|-------|-------|---------:|----------:|--------------:|-------------:|--------------:|---------|-------|
| mlx-swift  | mlx-community/Qwen3.5-0.8B-MLX-4bit        | Mac M4 Max | macOS 26   | Debug | Q4    |     1.77 |       962 |          24.2 |         82.2 |           614 | nominal | First-token cost dominates on a 0.8B model. Cold load includes MLX kernel JIT. |
| mlx-swift  | mlx-community/gemma-4-e2b-it-4bit          | Mac M4 Max | macOS 26   | Debug | Q4    |     2.62 |       369 |          60.8 |         60.3 |          2846 | nominal | MLX GPU path. |
| llama.cpp  | unsloth/gemma-4-E2B-it-GGUF Q4_K_M         | Mac M4 Max | macOS 26   | Debug | Q4_K_M | 365.97 |        44 |        1249.9 |        118.9 |          3182 | nominal | **Same Gemma 4 E2B as MLX & CoreML rows** — Metal kernels on M4 Max are ~2× faster decode and ~20× faster prefill than MLX-Swift for this model. |
| coreml-llm | coreml-llm/gemma4-e2b                      | Mac M4 Max | macOS 26   | Debug | INT4 palettized | 854.49 |       642 |             — |         32.8 |          1067 | nominal | **Same Gemma 4 E2B as MLX & llama.cpp rows** — CoreMLLLM on `.cpuAndNeuralEngine` (ANE). **Peak memory 1067 MB** vs 2846 MB (MLX) / 3182 MB (llama.cpp) — ANE-resident weights cut working set ~3×. Decode is the slowest of the three (32.8 vs 60.3 vs 118.9 tok/s), so this backend trades throughput for memory + battery. Cold load includes a 5.4 GB download. Prefill metric is 0 because CoreMLLLM's stream API doesn't surface a separate prefill counter. |
| coreml-llm | coreml-llm/lfm2.5-350m                     | Mac M4 Max | macOS 26   | Debug | INT8  |     5.97 |       383 |             — |         58.9 |            98 | nominal | LFM 2.5 350M via CoreMLLLM on `.cpuAndNeuralEngine`. **Peak memory 98 MB** — ANE residency at its extreme. Generated 94 tokens before hitting EOS. |
| mlx-swift  | mlx-community/Qwen3.5-2B-MLX-4bit          | Mac M4 Max | macOS 26   | Debug | Q4    |   189.45 |        50 |             — |         78.4 |          1248 | nominal | Outlier load time (warm-cache repro pending); steady-state decode is fast. |
| mlx-swift  | mlx-community/gemma-4-e4b-it-4bit          | Mac M4 Max | macOS 26   | Debug | Q4    |  1129.69 |       114 |         239.7 |         45.2 |          4417 | nominal | Larger Gemma 4 (E4B) — decode drops to 45 tok/s vs 60 (E2B), prefill jumps to 239 tok/s vs 60 (likely warmer kernel cache by this run). Load includes 5.2 GB cold download (succeeded on a second-day retry after two stalled attempts at ~31 MB). |
| mlx-swift  | _any_                                      | MacBook Air M3 16 GB | macOS 26 | Debug | Q4 | _wanted_ | _wanted_ | _wanted_ | _wanted_ | _wanted_ | _wanted_ | See [`devices/macbook-air-m3.md`](devices/macbook-air-m3.md). |
| mlx-swift  | _any_                                      | iPhone 17 Pro | iOS 26   | Release | Q4 | _wanted_ | _wanted_ | _wanted_ | _wanted_ | _wanted_ | _wanted_ | Run via [`ios/BenchmarkApp`](ios/BenchmarkApp). |

Raw JSON for each Mac M4 Max row lives in [`results/raw/`](results/raw/) (`m4max-mlx-<model>-short-chat.jsonl`).

## Task B — Long-context prefill (2 K prompt, 64-token output)

_No rows yet. Add a Mac / iPhone row by running `yardstick run --task long-context …` (Mac) or the iOS app's "Long context" task and PR the JSONL + a row here._

## Task C — Sustained generation (512 tokens)

_No rows yet. Watches thermal drift over a longer run; particularly important on the MacBook Air M3 (fanless) and iPhone._

## Task D — App-lifecycle loop

_No rows yet. The iOS app's lifecycle task covers cancellation latency, background→foreground recovery, and repeated-generation stability. Not yet plumbed for the Mac CLI._

## Failed runs

Failure is signal — keep them visible.

| Runtime    | Model                                | Device     | Result          | Reason |
|------------|--------------------------------------|------------|-----------------|--------|
| coreml-llm | coreml-llm/qwen3.5-0.8b              | Mac M4 Max | Load failed     | `mlboydaisuke/qwen3.5-0.8B-CoreML` ships chunked / specialized `.mlpackage`s (decode_chunks_mlkv, decode_fp16_mseq128, prefill_stateful, …) rather than the single top-level `model.mlpackage` `CoreMLLLM` expects. Model-side repo layout to be reorganized, or `CoreMLLLM` to learn the chunked layout. |
| executorch | executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET | Mac M4 Max | Tokenizer parse error | `tokenizers:hf_tokenizer.cpp:75 — error parsing json file`. ET's HF tokenizer C++ implementation expects `tokenizer.json` (HF Tokenizers format); both the SpinQuant and the plain `Llama-3.2-1B-Instruct-ET` repos only ship `tokenizer.model` (SentencePiece binary). Needs an ET model with HF-format tokenizer, or a SentencePiece adapter on the ET side. |
| anemll     | anemll/anemll-google-gemma-3-270m-it-ctx4096_0.3.5 | Mac M4 Max | Download failed | `swift-huggingface`'s `HFDownloader` errored with `HTTPClientError error 1` — the repo packages model components as `.mlmodelc/` directories with internal files, and the downloader's path-traversal for nested directories returns a 4xx/5xx path. Either the downloader needs `.mlmodelc/` glob support, or ANEMLL needs its own download flow. |
| litert-lm  | (any)                                | Mac M4 Max | Not buildable   | `paescebu/SwiftTasksGenAI 0.10.24`'s vendored `MediaPipeTasksGenAI.xcframework` only ships `ios-arm64` + `ios-arm64_x86_64-simulator` slices — no `macos-arm64*` slice. Mac wiring blocked upstream. |

## How to add a row

1. Build the runner.
   - Mac CLI:
     ```sh
     cd ios/BenchmarkApp && xcodegen generate
     xcodebuild -project BenchmarkApp.xcodeproj -scheme yardstick \
                -destination "platform=macOS" -configuration Release build
     ```
   - iPhone app: see [`README.md`](README.md#running-on-iphone-app).
2. Run a task and capture the JSONL:
   ```sh
   ./yardstick run --task short-chat \
                   --runtime mlx-swift \
                   --model mlx-community/gemma-4-e2b-it-4bit \
                   --output results/raw/<device>-<runtime>-<model>-<task>.jsonl
   ```
3. PR the JSONL into [`results/raw/`](results/raw/) and add the corresponding row above. Devices should match a page in [`devices/`](devices/) — add one if your hardware isn't represented yet.
