# Core AI export TODO — fill every blank for the 10-model study (prep'd 2026-06-24)

Answer to "can all this session's models have their blanks filled?": **Yes — every Core AI blank is fillable, but
each needs a Core AI export class (or config shim) on the `coreai-models` side.** This doc scopes that separate
session precisely. Non-Core-AI gaps (MLX/LiteRT) are listed at the bottom.

Registry state (`coreai-models/.../models/registry.py` + `models/ios/`, `models/macos/`):
- **iOS class exists:** qwen2, qwen3, qwen3_5, mistral, **gemma4**  → iPhone Core AI works
- **macOS class only (no iOS):** gemma3, llama, gpt_oss, mixtral, qwen3_moe …  → Mac works, iPhone blocked
- **no class at all:** phi3, olmo2, smollm3  → both Mac + iPhone blocked

## The 10 models — Core AI status & what each needs

| Model | arch | Mac CoreAI | iPhone CoreAI | What's needed |
|---|---|---|---|---|
| DeepSeek-R1-1.5B | qwen2 | ✅ 319.5 | ✅ ANE 83.3 / GPU 75.9 | **done** |
| TinySwallow-1.5B | qwen2 | ✅ 324.1 | ✅ 74.8 / 75.0 | **done** |
| VibeThinker-1.5B | qwen2 | ✅ 322.7 | ✅ 71.5 / 75.7 | **done** |
| Qwen3-1.7B | qwen3 | ✅ 239.1 | ✅ 64.8 / 67.6 | **done** |
| **Gemma3-1B** | gemma3 | ✅ 327.2 | ❌ | **iOS class only** (macOS done) → write `models/ios/gemma3.py` (adapt `gemma4.py`/`gemma4_ane.py`) |
| **Llama-3.2-3B** | llama | ✅ 198.3 | ❌ | **iOS class only** (macOS `llama.py` exists) → write `models/ios/llama.py` |
| **Phi-4-mini** | phi3 | ❌ | ❌ | **macOS + iOS class** → write `models/macos/phi3.py` + `models/ios/phi3.py` + register |
| **OLMo-2-1B** | olmo2 | ❌ | ❌ | **macOS + iOS class** → write `models/macos/olmo2.py` + `models/ios/olmo2.py` + register |
| **SmolLM3-3B** | smollm3 | ❌ | ❌ | **macOS + iOS class** (note: SmolLM3 uses **NoPE** on some layers) → write `models/macos/smollm3.py` + ios + register |
| **Ministral-3-3B** | ministral3 | ❌ | ❌ | **config shim only** — `mistral` class already has macOS+iOS; map `model_type "ministral3" → mistral` (`MODEL_TYPE_REMAPPING`) + handle the transformers `Mistral3Config` |

**So: 4 done, 6 to fill.** Effort ranking (low→high): Ministral (config shim) < Gemma3-iOS, Llama-iOS (macOS exists, mirror the iOS pattern) < Phi3, OLMo-2 (new arch, both platforms) < SmolLM3 (new arch + NoPE quirk).

## Export commands (after the class/shim lands)
Per model, both bundles for iPhone (mirrors `scripts/export_coreai_qwen3.sh`):
```bash
# GPU (macOS dynamic int4 → iOS-compiled GPU bundle)
uv run coreai.llm.export <hf-id> --platform macOS --compression 4bit --compute-precision float16 \
    --experimental --output-name <name>_dynamic
xcrun coreai-build compile exports/<name>_dynamic/<name>_dynamic.aimodel \
    --platform iOS --preferred-compute gpu --architecture h18p --output <tmp>
# ANE (iOS static palettized → neural-engine bundle)
uv run coreai.llm.export <hf-id> --platform iOS --compression 4bit_weight_palettized_group32 \
    --compute-precision float16 --max-context-length 4096 --experimental --output-name <name>_ios_pure4bit
xcrun coreai-build compile exports/<name>_ios_pure4bit/<name>_ios_pure4bit.aimodel \
    --platform iOS --preferred-compute neural-engine --architecture h18p --output <tmp>
```
HF ids: Phi `microsoft/Phi-4-mini-instruct`, OLMo `allenai/OLMo-2-0425-1B-Instruct`, SmolLM3
`HuggingFaceTB/SmolLM3-3B`, Gemma3 `google/gemma-3-1b-it`, Llama `meta-llama/Llama-3.2-3B-Instruct`,
Ministral `mistralai/Ministral-3-3B-Instruct-2512` (or the local `src_models/ministral3-3b-text`).
Mac Core AI is also benchable via the macOS bundle alone (no iOS compile) for the Mac column.

## After exporting — wire into the app (this repo, then rebuild w/ entitlements)
For each new bundle add a catalog entry (`ModelCatalog.swift`) + a `bundleSpec` case
(`CoreAIRuntime.swift`: `-ane` → `("<name>_ane","static-shape")`, `-gpu` → `("<name>_gpu","coreai-pipelined")`),
then add rows to `results/raw/2026-06-25-comprehensive/manifest.tsv`. Pattern = the qwen3/deepseek entries.

## Non-Core-AI blanks (for completeness)
- **MLX — VibeThinker, OLMo-2:** no mlx-community repo. Fillable via `mlx_lm.convert` **but** the app's MLX path
  downloads from HF Hub, so a local MLX model must be published to HF or pushed into the on-device HubClient cache
  (fiddly — see MLXRuntime/HubDownloaderBridge). Lower ROI.
- **MLX — Ministral-3-3B (iPhone):** **hard block** — MLX-Swift can't load the `ministral3` arch on-device
  (works on Mac mlx_lm). Not fixable from our side.
- **LiteRT — Phi-4-mini (iPhone):** int8 (3.6 GB) OOMs = quant, not LiteRT. An int4 Phi (~2 GB) would load, but
  `litert_torch` Phi3 export is currently blocked by a transformers/remote-code incompat (`LossKwargs`, then a
  `list.keys` export bug) — needs a converter-env fix. Everything else LiteRT is filled.

## Target matrix once the 6 Core AI classes land
All 10 models become 3-way (Core AI / MLX / LiteRT) on Mac, and 3-way on iPhone except: Ministral iPhone-MLX
(hard block) and VibeThinker/OLMo MLX (need repo). i.e. **the only permanently-blank cells are 3 MLX cells**;
every Core AI blank is closable with the classes above.
