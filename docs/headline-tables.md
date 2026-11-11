# Headline tables

<!-- BEGIN GENERATED: scripts/render_headline.py -->

**Decode tok/s, batch 1, greedy.** Generated from stored records by `scripts/render_headline.py` (newest record 2026-09-05), so do not edit inside the markers. Full standings with prefill, TTFT, memory and GSM8K: [LEADERBOARD.md](../LEADERBOARD.md); every capture: [RESULTS.md](../RESULTS.md) and `results/`. Numbers are compared only within one session, so no ratios are printed.

Task `short-chat`, with the same prompt and 128-token budget for every runtime. **Bold** is warm: the median of one session's in-process runs, with run 1 of every launch dropped. "cold" is a fresh-process first generation, shown where a session has no warm runs. Only runs that started thermal-nominal count ([fairness-rules §2](../methodology/fairness-rules.md)). Each number is from its newest qualifying session, the date in parentheses. Several artifacts of one runtime are listed fastest first and never pooled.

| Model | Device | Apple Core AI | MLX | LiteRT-LM | Core ML |
|---|---|---:|---:|---:|---:|
| Qwen 3 0.6B ¹ | iPhone 17 Pro | 193.3 cold `qwen3-0.6b-gpu` (2026-06-18)<br>**122.4** `qwen3-0.6b-ane-june` (2026-07-13)<br>**116.9** `qwen3-0.6b-ane` (2026-07-13) | **178.8** (2026-08-26) | **122.1** (2026-08-26) | 37.7 cold (2026-06-17) |
| Qwen 3 0.6B | Mac Studio (M4 Max) | — | **555.9** (2026-08-17) | **309.9** (2026-08-17) | — |
| Qwen 3 8B | Mac Studio (M4 Max) | — | 98.3 cold (2026-06-17) | 62.4 cold (2026-06-17) | — |

¹ Qwen 3 0.6B · iPhone 17 Pro: the numbers come from 5 capture sessions (Apple Core AI 2026-06-18 and 2026-07-13 (`2026-07-14-iphone-final`); MLX 2026-08-26 (`2026-08-26-iphone-lfm-pair`); LiteRT-LM 2026-08-26 (`2026-08-26-iphone-qwen-3runtime-pair`); Core ML 2026-06-17). Device state moves between sessions, so a ratio between two numbers in this row is not a measurement. Compare within one session. Evidence: `results/raw/2026-07-13-mlx-variance/`.

<details><summary>Recipes behind these numbers</summary>

- iPhone 17 Pro · Qwen 3 0.6B · Apple Core AI: `core-ai/qwen3-0.6b-gpu` — INT4 (dynamic), engine pre-stamp, cold, 3 fresh-process launches in that session (no warm runs), last one shown, session 2026-06-18; newer capture on 2026-08-26 started hot and did not qualify (thermal guard)
- iPhone 17 Pro · Qwen 3 0.6B · Apple Core AI: `core-ai/qwen3-0.6b-ane` — 4-bit palettized (uniform g32), engine pre-stamp, warm median of 3 in-process runs, session 2026-07-13 (`2026-07-14-iphone-final`)
- iPhone 17 Pro · Qwen 3 0.6B · Apple Core AI: `core-ai/qwen3-0.6b-ane-june` — mixed 4/8-bit (June static export), engine pre-stamp, warm median of 3 in-process runs, session 2026-07-13 (`2026-07-14-iphone-final`)
- iPhone 17 Pro · Qwen 3 0.6B · MLX: `mlx-community/Qwen3-0.6B-4bit` — Q4, engine 60bd0d7880c82980f9481f8be78862e9b63c58a3, warm median of 2 in-process runs, session 2026-08-26 (`2026-08-26-iphone-lfm-pair`)
- iPhone 17 Pro · Qwen 3 0.6B · LiteRT-LM: `litert-community/Qwen3-0.6B` — INT4 (mixed, blockwise gs32), engine v0.16.0, warm median of 3 in-process runs, session 2026-08-26 (`2026-08-26-iphone-qwen-3runtime-pair`)
- iPhone 17 Pro · Qwen 3 0.6B · Core ML: `coreml-llm/qwen3-0.6b` — INT8 palettized, engine pre-stamp, cold, 1 fresh-process launch in that session (no warm runs), last one shown, session 2026-06-17
- Mac Studio (M4 Max) · Qwen 3 0.6B · MLX: `mlx-community/Qwen3-0.6B-4bit` — Q4, engine 60bd0d7880c82980f9481f8be78862e9b63c58a3, warm median of 2 in-process runs, session 2026-08-17 (`2026-08-17-mac-litert-v0160`)
- Mac Studio (M4 Max) · Qwen 3 0.6B · LiteRT-LM: `litert-community/Qwen3-0.6B` — INT4 (mixed, blockwise gs32), engine v0.16.0, warm median of 3 in-process runs, session 2026-08-17 (`2026-08-17-mac-litert-v0160`)
- Mac Studio (M4 Max) · Qwen 3 8B · MLX: `mlx-community/Qwen3-8B-4bit` — Q4, engine pre-stamp, cold, 3 fresh-process launches in that session (no warm runs), last one shown, session 2026-06-17
- Mac Studio (M4 Max) · Qwen 3 8B · LiteRT-LM: `litert-community/Qwen3-8B` — INT4 (mixed, blockwise gs32), engine pre-stamp, cold, 3 fresh-process launches in that session (no warm runs), last one shown, session 2026-06-17

</details>

**Apple `llm-benchmark` protocol, Mac** — 512-token prompt, 1024 generated, 5 trials, greedy; Apple Core AI = Apple's `llm-benchmark` release build, MLX = `mlx_lm benchmark` with the same arguments. A different budget from the harness rows above, so never compare a number here with one there. Each number is from its newest session, the date in parentheses.

| Model | Device | Apple Core AI | MLX |
|---|---|---:|---:|
| Qwen 3 0.6B ¹ | Mac Studio (M4 Max) | 503.1 (2026-07-13) | 432.3 (2026-06-11) |
| Qwen 3 8B | Mac Studio (M4 Max) | 94.1 (2026-06-11) | 90.0 (2026-06-11) |

¹ Qwen 3 0.6B · Mac Studio (M4 Max): the numbers come from 2 capture sessions (Apple Core AI 2026-07-13 (`2026-07-13-mac-warm/coreai-ct041`); MLX 2026-06-11 (`2026-06-11-m4max-coreai-matrix`)). Device state moves between sessions, so a ratio between two numbers in this row is not a measurement. Compare within one session. Evidence: `results/raw/2026-07-13-mlx-variance/`.

<details><summary>Recipes behind the llm-benchmark numbers</summary>

- Mac Studio (M4 Max) · Qwen 3 0.6B · Apple Core AI: `qwen3_0_6b_dynamic_ct041` — Apple `llm-benchmark`, 512p/1024g, mean of 5 trials (trial spread 0.1%), `results/raw/2026-07-13-mac-warm/coreai-ct041/`
- Mac Studio (M4 Max) · Qwen 3 0.6B · MLX: `mlx-community/Qwen3-0.6B-4bit` — `mlx_lm benchmark` (mlx-lm 0.31.3), same 512p/1024g/5 arguments, mean of the trials, `results/raw/2026-06-11-m4max-coreai-matrix/mlx_sweep_0.31.3.log`
- Mac Studio (M4 Max) · Qwen 3 8B · Apple Core AI: `qwen3_8b_4bit_dynamic` — Apple `llm-benchmark`, 512p/1024g, mean of 5 trials (trial spread 0.3%), `results/raw/2026-06-11-m4max-coreai-matrix/`
- Mac Studio (M4 Max) · Qwen 3 8B · MLX: `mlx-community/Qwen3-8B-4bit` — `mlx_lm benchmark` (mlx-lm 0.31.3), same 512p/1024g/5 arguments, mean of the trials, `results/raw/2026-06-11-m4max-coreai-matrix/mlx_sweep_0.31.3.log`

</details>

**Hybrid Mamba-2 / Transformer models, CLI runs** — llama.cpp = `llama-bench` tg256 (pp512 prompt), MLX = `mlx_lm generate` 256 tokens; greedy, batch 1, mains power, one machine. Quantization is **not** equal across columns (Q4_K_M is not 4-bit affine — see the recipes and the caveats in each report); ‡ = the report records a coherence problem with that output. Reports: `results/hybrid/hybrid-apple-silicon-bench-2.json`, `results/hybrid/nemotron3-nano-apple-silicon-bench.json`.

| Model | Device | MLX | llama.cpp | Measured |
|---|---|---:|---:|---|
| NVIDIA Nemotron-3 Nano 30B-A3B | Apple M4 Max | 159.7 | 86.2 | 2026-09-04 |
| NVIDIA Nemotron-3 Nano 4B | Apple M4 Max | 176.8 | 88.4 | 2026-09-04 |
| IBM Granite-4.0-H-Tiny | Apple M4 Max | 202.2 | 117.3 | 2026-09-04 |
| Granite-4.0-H 350M | Apple M4 Max | 521.7 | 282.5 | 2026-09-05 |
| Granite-4.0-H 1B | Apple M4 Max | 275.8 | 141.9 | 2026-09-05 |
| Falcon-H1 1.5B-Instruct | Apple M4 Max | 300.0 ‡ | 147.5 | 2026-09-05 |
| Falcon-H1 3B-Instruct | Apple M4 Max | 167.9 ‡ | 86.4 | 2026-09-05 |
| Falcon-H1 7B-Instruct | Apple M4 Max | — | 52.7 | 2026-09-05 |

<details><summary>Recipes behind the hybrid numbers</summary>

- NVIDIA Nemotron-3 Nano 30B-A3B · MLX: mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit 4-bit (mlx-community) (mlx 0.32.2, mlx-lm 0.31.3; 256 tokens, median of 3 processes)
- NVIDIA Nemotron-3 Nano 30B-A3B · llama.cpp: unsloth/Nemotron-3-Nano-30B-A3B-GGUF Q4_K_M (Homebrew build 8680, commit 15f786e65; tg256 mean of 3 (±0.47))
- NVIDIA Nemotron-3 Nano 4B · MLX: mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit 4-bit (mlx-community) (mlx 0.32.2, mlx-lm 0.31.3; 256 tokens, median of 3 processes)
- NVIDIA Nemotron-3 Nano 4B · llama.cpp: nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF Q4_K_M (Homebrew build 8680, commit 15f786e65; tg256 mean of 3 (±0.22))
- IBM Granite-4.0-H-Tiny · MLX: lmstudio-community/granite-4.0-h-tiny-MLX-4bit 4-bit affine, group_size 64, MoE router layers 8-bit (from config.json) (mlx 0.32.2, mlx-lm 0.31.3; 244 tokens (EOS before 256), median of 3 processes)
- IBM Granite-4.0-H-Tiny · llama.cpp: unsloth/granite-4.0-h-tiny-GGUF Q4_K_M (Homebrew build 8680, commit 15f786e65; tg256 mean of 3 (±0.92))
- Granite-4.0-H 350M · MLX: mlx-community/granite-4.0-h-350m-4bit rev 2b96c1f6 (mlx 0.32.2, mlx-lm 0.31.3; 256-token generation, median of 3 processes)
- Granite-4.0-H 350M · llama.cpp: ibm-granite/granite-4.0-h-350m-GGUF Q4_K_M rev a864f823 (llama.cpp build 8680 Homebrew; tg256 mean of 3 (±1.38))
- Granite-4.0-H 1B · MLX: mlx-community/granite-4.0-h-1b-4bit rev a5a21e23 (mlx 0.32.2, mlx-lm 0.31.3; 256-token generation, median of 3 processes)
- Granite-4.0-H 1B · llama.cpp: ibm-granite/granite-4.0-h-1b-GGUF Q4_K_M rev c2cb1972 (llama.cpp build 8680 Homebrew; tg256 mean of 3 (±1.69))
- Falcon-H1 1.5B-Instruct · MLX: mlx-community/Falcon-H1-1.5B-Instruct-4bit rev 6f5e4f68 (mlx 0.32.2, mlx-lm 0.31.3; 256-token generation, median of 3 processes) — ‡ degenerates into repetition after ~2 sentences on the long prompt
- Falcon-H1 1.5B-Instruct · llama.cpp: tiiuae/Falcon-H1-1.5B-Instruct-GGUF Q4_K_M rev 0d3a6cfe (llama.cpp build 8680 Homebrew; tg256 mean of 3 (±0.87))
- Falcon-H1 3B-Instruct · MLX: own 4-bit gs64 affine from tiiuae bf16 rev 01087ec4 with mlx-lm 0.31.3 (4.504 bpw) (mlx 0.32.2, mlx-lm 0.31.3; 256-token generation, median of 3 processes) — ‡ fluent but muddled
- Falcon-H1 3B-Instruct · llama.cpp: own Q4_K_M from tiiuae/Falcon-H1-3B-Instruct bf16 rev 01087ec4 via convert_hf_to_gguf.py (llama.cpp 8680) + llama-quantize, 4.79 BPW (llama.cpp build 8680 Homebrew; tg256 mean of 3 (±0.01))
- Falcon-H1 7B-Instruct · llama.cpp: tiiuae/Falcon-H1-7B-Instruct-GGUF Q4_K_M rev 058c8c8f (llama.cpp build 8680 Homebrew; tg256 mean of 3 (±0.19))

</details>

<!-- END GENERATED: scripts/render_headline.py -->

Not in the generated tables, because this repo holds no record for them: the Apple Core AI numbers for the hybrid models live on the model cards ([`mlboydaisuke/Nemotron-3-Nano-4B-CoreAI`](https://huggingface.co/mlboydaisuke/Nemotron-3-Nano-4B-CoreAI), [`coreai-community/granite-4.0-h-CoreAI`](https://huggingface.co/coreai-community/granite-4.0-h-CoreAI)), and the macOS-26-era Qwen3-0.6B Core AI artifact (the 1,121 tok/s figure in the Core AI section) left only an archived hash, not a benchmark record.
