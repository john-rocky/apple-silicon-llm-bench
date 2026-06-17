# Cross-runtime model matrix (verified)

What the bench can run, per runtime, with comparators **verified to exist on HuggingFace**
(checked with `huggingface_hub`, not guessed). Focus = the families LiteRT optimises (Qwen, Gemma).
LiteRT artifact = the GPU `mixed_int4` `.litertlm` on `litert-community` (each repo also ships
`channelwise_int8`, `-web`, and vendor NPU variants — `mediatek.mt6993`, `Google_Tensor_G5`).

## Qwen3 (Lu's lead family)
| Model | LiteRT `.litertlm` | MLX 4-bit | GGUF Q4_K_M | ~4-bit size | Device fit |
|---|---|---|---|---:|---|
| Qwen3-0.6B | `qwen3_0_6b_mixed_int4` ✅ | `mlx-community/Qwen3-0.6B-4bit` ✅ | `unsloth/Qwen3-0.6B-GGUF` ✅ | ~0.4 GB | iPhone + Mac |
| Qwen3-4B | `qwen3_4b_mixed_int4` ✅ | `…/Qwen3-4B-4bit` ✅ | `unsloth/Qwen3-4B-GGUF` ✅ | ~2.3 GB | iPhone + Mac |
| Qwen3-8B | `qwen3_8b_mixed_int4` ✅ | `…/Qwen3-8B-4bit` ✅ | `unsloth/Qwen3-8B-GGUF` ✅ | ~4.5 GB | Mac (iPhone jetsam risk) |
| Qwen3-14B | `qwen3_14b_mixed_int4` ✅ | `…/Qwen3-14B-4bit` ✅ | `unsloth/Qwen3-14B-GGUF` ✅ | ~8 GB | Mac-only |
| Qwen3-4B-Instruct-2507 | `qwen3_4b_instruct_2507_mixed_int4` ✅ | `…/Qwen3-4B-Instruct-2507-4bit` ✅ | `unsloth/…-GGUF` ✅ | ~2.3 GB | iPhone + Mac (optional) |

## Gemma (gemma-4 / gemma-3n)
| Model | LiteRT `.litertlm` | MLX 4-bit | GGUF Q4_K_M | CoreML/ANE | ~4-bit size | Device fit |
|---|---|---|---|---|---:|---|
| gemma-4-E2B | `gemma-4-E2B-it` ✅ | `…/gemma-4-e2b-it-4bit` ✅ | `unsloth/gemma-4-E2B-it-GGUF` ✅ | local bundle ✅ | ~2.6 GB | iPhone + Mac |
| gemma-4-E4B | `gemma-4-E4B-it` ✅ | `…/gemma-4-e4b-it-4bit` ✅ | `unsloth/gemma-4-E4B-it-GGUF` ✅ | `mlboydaisuke/gemma-4-E4B-coreml` | ~3.75 GB | iPhone borderline / Mac |
| gemma-4-12B | `gemma-4-12B-it` ✅ | `…/gemma-4-12b-it-4bit` ✅ | `unsloth/gemma-4-12B-it-GGUF` ✅ | — | ~7 GB | Mac-only |

## Recommended bench set (in `scripts/full_matrix.sh`)
- **iPhone (devicectl):** Qwen3-{0.6B,4B} + Gemma-{E2B,E4B} on litert/mlx/llama (+CoreML for Gemma);
  Qwen3-8B + Gemma-E4B attempted (≳3 GB → may jetsam; recorded per fairness rule 4).
- **Mac (yardstick CLI, litert+mlx):** all of the above + **Qwen3-{8B,14B}** and **Gemma-12B** —
  the desktop tier runs the models phones can't, giving a full Qwen3 0.6→14B and Gemma E2B→12B
  scaling curve. (CoreML/llama on Mac use the xcodebuild target — see RUNBOOK.)

## Lu's focus models — now benchable via OUR own `.litertlm` conversions
LiteRT doesn't publish Liquid/LFM2 or MiniCPM on `litert-community`, so we converted them ourselves
(`~/code/litertlm-convert/deliverables/`), side-loaded (no HF download). Mac preview (short-chat, M4 Max,
median n=3 where it runs):

| Model | LiteRT (`.litertlm`, ours) | MLX 4-bit | GGUF |
|---|---|---|---|
| **MiniCPM5-1B** | ✅ **runs — 239 tok/s** (CPU, ~610 MB) | `mlx-community/MiniCPM5-1B-4bit` ✅ **526 tok/s** | none on HF |
| **LFM2.5-350M** | ⚠️ **loads but fails to invoke** (`INTERNAL: Failed to invoke the compiled model`, litert-lm 0.13.1 macOS) | `mlx-community/LFM2-350M-4bit` (v2.0 ✅ 1024 tok/s) | `LiquidAI/LFM2.5-350M-GGUF` ✅ |

- **MiniCPM5-1B** is a genuine cross-runtime LiteRT-vs-MLX pair on a Lu-named model — the conversion works.
- **LFM2.5-350M**: the litert bundle loads but the 0.13.1 runtime executor can't invoke it (LFM2's hybrid
  conv/attention ops, likely). A finding for both the conversion and litert-lm's LFM2 support; the iPhone
  Metal-GPU path is untested (may differ — that's what 実機 will tell us).
- MLX LFM comparator is **LFM2-350M (v2.0)**, not 2.5 (no 2.5 on mlx-community) — a version skew, disclosed.

> Quant is each runtime's native 4-bit (LiteRT mixed-INT4 / MLX Q4 / GGUF Q4_K_M) — disclosed per
> row, never equalised. Sizes are approximate; the report uses the runtime-recorded counts.
