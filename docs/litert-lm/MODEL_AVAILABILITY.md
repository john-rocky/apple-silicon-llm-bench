# LiteRT model availability — what's benchmarkable today

> Inventory of the `litert-community` HF org's **text-generation LLMs**, mapped to the model
> families Qwen, Liquid/LFM2, MiniCPM and Gemma, and to whether a
> **deployable bundle** (`.litertlm` / `.task`) actually ships. Snapshot **2026-06-17**, queried live
> via `huggingface_hub` (see *Reproduce* below). Part of the
> [LiteRT-LM package](README.md) / [Apple Silicon LLM Benchmark](../../README.md).

## TL;DR by model family

| Family | litert-community status | Benchmarkable now? |
|---|---|:--:|
| **Qwen** | **Rich** — Qwen3 0.6B / 4B / 4B-Instruct-2507 / 8B / 14B all ship `.litertlm`; Qwen2.5 1.5B + DeepSeek-R1-Distill-Qwen-1.5B too | ✅ yes |
| **Gemma** | **Rich** — gemma-4 E2B / E4B / 12B, Gemma3-1B-IT ship `.litertlm`; larger Gemma3 / MedGemma / TranslateGemma repos present | ✅ yes |
| **Liquid / LFM2** | **None** — no LFM2 / Liquid model in litert-community, and **none anywhere on HF in `.litertlm` form** | ❌ blocked on publication |
| **MiniCPM** | **None in litert-community.** One third-party `.litertlm` exists (`lyafence/MiniCPM5-1B-SFT-litertlm`), not an official LiteRT bundle | ❌ blocked on official publication |

We already report fair iPhone 17 Pro numbers for **Qwen3-0.6B** and **Gemma-4-E2B**
([README](README.md)). The harness is model-agnostic — adding a published `.litertlm` is a catalog
entry, not new code.

## Deployable text-generation LLMs in `litert-community` (verified file-by-file)

Bundle presence below was confirmed with `list_repo_files` on **2026-06-17**.
`.litertlm` = current LiteRT-LM bundle (on-device mobile/desktop). `.task` = legacy MediaPipe
GenAI / web format (the adapter loads `.litertlm` first, then falls back to `.task`).

| Model | Params | `.litertlm`? | Notable variants on the repo |
|---|---|:--:|---|
| `litert-community/Qwen3-0.6B` | 0.6B | ✅ | `qwen3_0_6b_mixed_int4`, generic, MediaTek `mt6993` |
| `litert-community/Qwen3-4B` | 4B | ✅ | `mixed_int4`, `channelwise_int8_float32kv` |
| `litert-community/Qwen3-4B-Instruct-2507` | 4B | ✅ | `mixed_int4` |
| `litert-community/Qwen3-8B` | 8B | ✅ | `mixed_int4`, `channelwise_int8_float32kv` |
| `litert-community/Qwen3-14B` | 14B | ✅ | `mixed_int4`, `channelwise_int8_float32kv` |
| `litert-community/Qwen2.5-1.5B-Instruct` | 1.5B | ✅ | `q8`/`f32` ekv4096 `.litertlm` + several `.task` |
| `litert-community/Qwen2.5-0.5B-Instruct` | 0.5B | ⚠️ `.task` only | `f32`/`q8` ekv1280 `.task` (no `.litertlm`) |
| `litert-community/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ✅ | `q8` ekv4096 `.litertlm` + `.task` |
| `litert-community/gemma-4-E2B-it-litert-lm` | E2B | ✅ | `gemma-4-E2B-it` + web / Tensor-G5 / Intel / Qualcomm |
| `litert-community/gemma-4-E4B-it-litert-lm` | E4B | ✅ | `gemma-4-E4B-it` + web |
| `litert-community/gemma-4-12B-it-litert-lm` | 12B | ✅ | `gemma-4-12B-it` |
| `litert-community/Gemma3-1B-IT` | 1B | ✅ | `int4` `.litertlm` + many per-SoC / `.task` variants |
| `litert-community/Gemma3-4B-IT` | 4B | ⚠️ web `.task` only | `int4`/`int8`/`q4_0` **-web** `.task` (no native mobile `.litertlm`) |
| `litert-community/Phi-4-mini-instruct` | 3.8B | ✅ | `q8` ekv4096 `.litertlm` + `.task` |

Also present in the org (text-gen, **bundle format not individually re-verified in this snapshot**):
`Gemma2-2B-IT`, `Gemma3-12B-IT`, `Gemma3-27B-IT`, `MedGemma-27B-IT`, `TranslateGemma-{4,12,27}B-IT`,
`gemma-3-270m-it`, `SmolLM-135M`, `SmolLM2-{135M,360M}`, `TinyLlama-1.1B-Chat`, `TinySwallow-1.5B`,
`VibeThinker-1.5B`, `Gecko-110m`, FunctionGemma 270M fine-tunes.

> Out of scope here (not text-generation LLMs): `Qwen3-ASR-0.6B`, `embeddinggemma-300m`,
> `FastVLM-0.5B` / `SmolVLM-256M` (vision-language), and the large vision/segmentation/detection zoo
> (MobileNet, EfficientNet, ConvNeXt, DeepLab, …).

## Not yet available

- **Liquid / LFM2:** no LFM2 or Liquid model is published by `litert-community`, and a hub-wide
  search returns **no `.litertlm` LFM2 anywhere** (2026-06-17).
- **MiniCPM:** none in `litert-community`. The only `.litertlm` MiniCPM on the hub is a third-party
  community upload (`lyafence/MiniCPM5-1B-SFT-litertlm`), not an official LiteRT bundle — usable for
  a smoke test, but not the artifact to cite.

**We are set up to run a fair cross-runtime bench (throughput · memory · energy · sustained
throttling, vs MLX / llama.cpp / CoreML on the same device) within days of LiteRT publishing an
LFM2 or MiniCPM `.litertlm`** — it's a `ModelCatalog` entry plus a side-load, on the harness that
already drives Qwen3 and Gemma. We're tracking the `litert-community` org and will pounce when they
land.

## Reproduce this snapshot

```bash
python3 - <<'PY'
from huggingface_hub import list_models, list_repo_files
llm = ("qwen","gemma","phi","smollm","tinyllama","deepseek","llama","minicpm",
       "lfm","liquid","swallow","vibe","gecko")
repos = sorted(m.id for m in list_models(author="litert-community", limit=500))
print("\n".join(r for r in repos if any(k in r.lower() for k in llm)))
print(list_repo_files("litert-community/Qwen3-4B"))    # confirm .litertlm bundles
# cross-hub absence check:
for q in ("LFM2","MiniCPM"):
    print(q, [m.id for m in list_models(search=q, limit=400) if "litert" in m.id.lower()])
PY
```
