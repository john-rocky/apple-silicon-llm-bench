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
| mlx-swift  | mlx-community/gemma-4-e2b-it-4bit          | Mac M4 Max | macOS 26   | Debug | Q4    |     2.62 |       369 |          60.8 |         60.3 |          2846 | nominal | Larger memory footprint than the 0.8B; decode slightly slower than Qwen 3.5 0.8B as expected. |
| mlx-swift  | mlx-community/Qwen3.5-2B-MLX-4bit          | Mac M4 Max | macOS 26   | Debug | Q4    |   189.45 |        50 |             — |         78.4 |          1248 | nominal | Outlier load time (warm-cache repro pending); steady-state decode is fast. |
| mlx-swift  | mlx-community/gemma-4-e4b-it-4bit          | Mac M4 Max | macOS 26   | Debug | Q4    |        — |         — |             — |            — |             — | —       | HF download timed out at ~3.6 GB on the first attempt; will retry. |
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
| mlx-swift  | mlx-community/gemma-4-e4b-it-4bit    | Mac M4 Max | Download failed | HuggingFace timeout at ~3.6 GB on first attempt. Retry after network recovery. |

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
