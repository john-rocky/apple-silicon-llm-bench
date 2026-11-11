# Apple Silicon LLM Benchmark

A neutral, reproducible benchmark for running local LLMs on Apple Silicon: iPhone, iPad and Mac. I built it to find out which runtime and model I could actually ship in an iOS app. It compares MLX Swift, llama.cpp, LiteRT-LM, Cactus, Core ML and Apple Core AI, the Core ML successor in iOS 27 / macOS 27.

This repo is now the archive of those measurements, with the published tables and the raw records. Since 2026-08-17 the benchmark has kept running in [edge-llm-bench](https://github.com/john-rocky/edge-llm-bench), and most new runs go there.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/john-rocky/apple-silicon-llm-bench/main/docs/images/gemma4-e2b-iphone-dark.svg">
  <img alt="Gemma 4 E2B on an iPhone 17 Pro, one row per build, grouped by capture session. Session 7/20, Cactus, CQ4 uncalibrated: 50.6 tok/s cold, 1,061 MB peak, GSM8K 87%. Session 7/27, llama.cpp, Q4_K_M: 38.8 tok/s, 191 MB mmap, GSM8K 76%. Session 7/28, LiteRT-LM, wNa8o8 QAT: 61.1 tok/s, 497 MB, GSM8K 86%. Session 7/28, MLX Swift, PTQ 4-bit: 49.1 tok/s, 3,010 MB, GSM8K 84%. Session 7/28, MLX Swift, QAT OptiQ int4: 36.0 tok/s, 4,592 MB, GSM8K 91%. Session 7/29, Cactus, CQ4 as shipped: 50.0 tok/s, 632 MB, GSM8K 3%. Session 7/29, Core AI, own int4, patched engine: 47.1 tok/s, 755 MB mmap, GSM8K 88%." src="docs/images/gemma4-e2b-iphone-light.svg" width="860">
</picture>
</p>

| Question | What the runs show so far | More |
|---|---|---|
| Which runtime is fastest on an iPhone? | It depends on the model. In warm runs on an iPhone 17 Pro, each model in one session, LiteRT-LM led on Gemma 4 E2B on 2026-07-28 (61.1 tok/s, MLX 49.1). Core AI led on Qwen3-1.7B on 2026-07-13 (67.6 tok/s on a bundle exported in June, MLX 65.1, LiteRT-LM 49.3 on my own int4 conversion). | [Gemma](docs/results-iphone.md), [Qwen3](LEADERBOARD.md#iphone-17-pro) |
| Does the build matter? | In July 2026 on Gemma 4 E2B, MLX scored 84.0% on GSM8K with a 4-bit PTQ build and 91.0% with a QAT OptiQ build. Cactus replaced its default CQ4 build that month and [renamed the old one uncalibrated](https://huggingface.co/Cactus-Compute/gemma-4-E2B-it/commit/3469ae47509776019b7ae4d4f4f1497ce9abebe7). The two run at the same speed, and on the same harness the old one scored 87.0% and the new default 3.0%. | [iPhone results](docs/results-iphone.md) |
| Does the speed last? | Not fully. In 10-minute runs of Gemma 4 E2B on an unplugged iPhone 17 Pro in July 2026, started at thermal state `nominal`, LiteRT-LM fell from 57.1 tok/s to 35.4 and MLX's 4-bit PTQ build from 45.7 to 25.1, comparing each run's first 5 seconds with its last 10%. | [Sustained load](docs/results-sustained.md) |
| Is Core AI faster than MLX on a Mac? | At 8B they are close. Under Apple's `llm-benchmark` protocol on an M4 Max (macOS 27 beta, June 2026), Qwen3-8B in 4-bit decoded at 94 tok/s on Core AI and 90 on MLX. The table further down uses this repo's own harness, so its MLX number differs. | [Core AI results](docs/results-core-ai.md) |

[LEADERBOARD.md](LEADERBOARD.md) has the standings by model and device. The newest record in this repo's tables is from 2026-09-05, and most later runs are in [edge-llm-bench](https://github.com/john-rocky/edge-llm-bench).

## Running on Mac (CLI)

The command-line runner is `yardstick`. You need Xcode 27 and the macOS 27 SDK. `bootstrap.sh` clones the runtimes it builds against. Replace `<device>` with a name for your machine:

```sh
git clone https://github.com/john-rocky/apple-silicon-llm-bench.git
cd apple-silicon-llm-bench/ios/BenchmarkApp && ./scripts/bootstrap.sh
cd ../.. && GIT_LFS_SKIP_SMUDGE=1 swift build -c release --product yardstick
YS=.build/release/yardstick
$YS list
$YS run --task short-chat --runtime mlx-swift --model mlx-community/Qwen3-0.6B-4bit \
        --output results/raw/<device>-mlx-qwen3-0.6b.jsonl
```

Each `run` writes one JSONL file with the speed, memory and thermal numbers. This build includes MLX and LiteRT-LM. To add llama.cpp and ANEMLL, build with `scripts/build_yardstick_mac.sh`, which also needs `xcodegen`. Use Release for real numbers: a Debug build adds large per-token overhead and understates decode speed.

## Running on iPhone (app)

`bootstrap.sh` can't build Cactus any more, because the commit this repo used is no longer on GitHub. It removes Cactus from the Xcode project. The app still does not build from a fresh clone with Xcode 27 (27A266a): the Core AI package that `bootstrap.sh` clones, apple/coreai-models 0.2.0, does not compile against that SDK ([apple/coreai-models#293](https://github.com/apple/coreai-models/issues/293)).

From the repo root:

```sh
cd ios/BenchmarkApp
./scripts/bootstrap.sh           # fetches the runtimes the app builds against
open BenchmarkApp.xcodeproj      # set your Team in Signing & Capabilities
                                 # ⌘R on a connected iPhone
```

The first launch downloads the chosen model into the app's Documents directory. The default is `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, about 300 MB. Use the picker to swap models.

Start a run from the **Run** tab. To keep the results, choose **Export all (JSONL)** in the **History** tab, AirDrop the file to your Mac, and import it from the repo root. `<label>` is a key of `DEVICE_DISPLAY` in `scripts/bench_common.py`, such as `iphone17pro`:

```sh
python3 scripts/import_ios_export.py <file> --device <label>
```

## How the numbers are made

- Within a table, runtimes get the same prompt and the same token budget where possible.
- Cold and warm runs are reported separately. A run does not count in the `short-chat` headline tables if the device was already running hot when it started.
- Each runtime runs its own best available build, and the recipe for each number is listed with it.
- Every number in the generated tables traces to a stored file in `results/`. The runs that failed are listed too, with the reason, in [RESULTS.md](RESULTS.md#failed-runs-kept-visible).

Core ML is off by default in current builds because it runs on my own library, [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM), and I keep that out of the neutral default. In July 2026 Apple's coreai-models had no Gemma 4 export (one was added on 2026-10-02, [apple/coreai-models#301](https://github.com/apple/coreai-models/pull/301)), so the chart's Core AI row uses my own int4 build and a local patch to Apple's engine. The full rules are in [`methodology/fairness-rules.md`](methodology/fairness-rules.md).

## Headline table

<!-- BEGIN GENERATED: scripts/render_headline.py -->

Decode tok/s, batch 1, greedy, on the `short-chat` task with a 128-token budget. Numbers from different capture sessions can't be compared, so this table keeps only rows measured in one session. The rest are in [all headline tables](docs/headline-tables.md), whose newest record is from 2026-09-05. **Bold** is a warm median and "cold" is the first generation in a fresh process.

| Model | Device | MLX | LiteRT-LM |
|---|---|---:|---:|
| Qwen 3 0.6B | Mac Studio (M4 Max) | **555.9** (2026-08-17) | **309.9** (2026-08-17) |
| Qwen 3 8B | Mac Studio (M4 Max) | 98.3 cold (2026-06-17) | 62.4 cold (2026-06-17) |

<details><summary>Recipes behind these numbers</summary>

- Mac Studio (M4 Max) · Qwen 3 0.6B · MLX: `mlx-community/Qwen3-0.6B-4bit` — Q4, engine 60bd0d7880c82980f9481f8be78862e9b63c58a3, warm median of 2 in-process runs, session 2026-08-17 (`2026-08-17-mac-litert-v0160`)
- Mac Studio (M4 Max) · Qwen 3 0.6B · LiteRT-LM: `litert-community/Qwen3-0.6B` — INT4 (mixed, blockwise gs32), engine v0.16.0, warm median of 3 in-process runs, session 2026-08-17 (`2026-08-17-mac-litert-v0160`)
- Mac Studio (M4 Max) · Qwen 3 8B · MLX: `mlx-community/Qwen3-8B-4bit` — Q4, engine pre-stamp, cold, 3 fresh-process launches in that session (no warm runs), last one shown, session 2026-06-17
- Mac Studio (M4 Max) · Qwen 3 8B · LiteRT-LM: `litert-community/Qwen3-8B` — INT4 (mixed, blockwise gs32), engine pre-stamp, cold, 3 fresh-process launches in that session (no warm runs), last one shown, session 2026-06-17

</details>

<!-- END GENERATED: scripts/render_headline.py -->

<a name="-contributing-a-row"></a>

## Contributing a row

The easiest contribution is one new row:

1. **A new device.** Run the existing models on your iPhone, iPad or Mac. No iPad run is stored yet. The "Devices wanted" list at the bottom of [`RESULTS.md`](RESULTS.md#devices-wanted) is the shortlist.
2. **A new model.** Add the model id to the [`ModelCatalog`](ios/BenchmarkApp/Sources/Models/ModelCatalog.swift) for the runtime that can load it.
3. **A new runtime.** Add an adapter in [`ios/BenchmarkApp/Sources/Runtimes/`](ios/BenchmarkApp/Sources/Runtimes/) following the `LLMRuntime` protocol.

Run it three times for a stable median, then regenerate the tables:

```sh
for run in 1 2 3; do
  $YS run --task short-chat \
          --runtime mlx-swift \
          --model <id-or-hf-repo> \
          --output results/raw/<device>-<runtime>-<model>-short-chat-run${run}.jsonl
done

python3 scripts/render_results.py
python3 scripts/build_summary.py && python3 scripts/render_leaderboard.py
python3 scripts/render_headline.py
```

Commit the JSONLs and the regenerated files, then open a PR. CI fails if the JSONLs and the tables disagree. The full steps are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation

- [LEADERBOARD.md](LEADERBOARD.md): current standings per platform, with prefill, TTFT, memory and GSM8K
- [RESULTS.md](RESULTS.md) and [`results/`](results/): every capture
- [All headline tables](docs/headline-tables.md): with the recipe behind each number
- [Apple Core AI results](docs/results-core-ai.md)
- [iPhone 17 Pro results](docs/results-iphone.md)
- [Sustained load and live-camera VLM](docs/results-sustained.md)
- [Apple M4 Max results](docs/results-mac.md)
- [What gets measured](docs/measuring.md)
- [Project layout, devices, backends and roadmap](docs/project.md)

## License

MIT, see [`LICENSE`](LICENSE).
