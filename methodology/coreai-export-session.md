# Core AI export session — instructions (self-contained, no bench knowledge needed)

**Your job:** write Core AI export classes for 6 model architectures and produce the iPhone bundles
(ANE + GPU). You work ENTIRELY in `~/code/coreai/coreai-models`. You do **not** need the benchmark app, the
device, or any bench details — a separate gap-fill session wires + measures. **Deliverable = assembled bundles
in `exports/<name>_gpu/` and `exports/<name>_ane_pure4bit/`, each containing a `*.h18p.aimodelc` + `tokenizer/` +
`metadata.json`.** Leave a checklist of what's ready at the end.

## Constraints
- `coreai-models` is **NOT a git repo** — hand-edit, no commits; keep a running note of files changed.
- **Never *run* an iOS bundle / compiled `.aimodelc` on the Mac — it kernel-panics the machine.** Exports and
  `xcrun coreai-build compile` are safe (they only *produce* artifacts). Don't execute iOS artifacts on the Mac.

## The 6 models — arch + exactly what to write
Registry: `python/src/coreai_models/models/registry.py`. Classes: `models/macos/`, `models/ios/`.

| Model | HF id | model_type | macOS class | iOS class | Write |
|---|---|---|---|---|---|
| Ministral-3-3B | `mistralai/Ministral-3-3B-Instruct-2512` | ministral3 | (mistral ✓) | (mistral ✓) | **config shim only** — add `"ministral3": "mistral"` to `MODEL_TYPE_REMAPPING`; ensure `Mistral3Config` loads (trust_remote_code or a small config shim). NO new class. |
| Gemma3-1B | `google/gemma-3-1b-it` | gemma3 | `Gemma3ForCausalLM` ✓ | ✗ | **iOS class only** → `models/ios/gemma3.py`, register `ios_class` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | llama | `models/macos/llama.py` ✓ | ✗ | **iOS class only** → `models/ios/llama.py`, register `ios_class` |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | phi3 | ✗ | ✗ | **macOS + iOS class** → `models/macos/phi3.py` + `models/ios/phi3.py` + register both |
| OLMo-2-1B | `allenai/OLMo-2-0425-1B-Instruct` | olmo2 | ✗ | ✗ | **macOS + iOS class** (OLMo-2 = QK-norm + post-norm — check HF modeling) + register both |
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B` | smollm3 | ✗ | ✗ | **macOS + iOS class** + register; **NoPE quirk** — some layers skip RoPE (`no_rope_layers`), replicate it |

**Order (low→high effort):** Ministral (shim) → Gemma3-iOS, Llama-iOS → Phi3, OLMo-2 → SmolLM3.
Each is independently shippable — finish one, hand it off, move on.

## How to write a class (pattern, don't reinvent)
- **macOS class:** mirror `models/macos/qwen2.py` / `qwen3.py` / `llama.py` (Apple `nn.Module` reimpl of the arch;
  weights mapped in `_mutate_state_dict`). Validate attention / MLP / norms against HF `modeling_<arch>.py`.
- **iOS class:** mirror `models/ios/qwen3.py` / `qwen2.py` / `gemma4.py` (static shapes; palettizable for ANE).
- **Register:** add a registry entry with `macos_class=`/`ios_class=` (and `MODEL_TYPE_REMAPPING` for Ministral).

## Export commands (per model, after the class lands)
Template: `scripts/export_coreai_qwen3.sh` in the bench repo (qwen3-named — copy + adjust `<name>` + `<hf-id>`).
```bash
# GPU bundle: macOS dynamic int4 export → iOS-compiled GPU
uv run coreai.llm.export <hf-id> --platform macOS --compression 4bit --compute-precision float16 \
    --experimental --output-name <name>_dynamic
xcrun coreai-build compile exports/<name>_dynamic/<name>_dynamic.aimodel \
    --platform iOS --preferred-compute gpu --architecture h18p --output <tmp> && \
    assemble <tmp> → exports/<name>_gpu/   (copy .h18p.aimodelc + tokenizer + metadata.json w/ assets.main set)
# ANE bundle: iOS static palettized → neural-engine
uv run coreai.llm.export <hf-id> --platform iOS --compression 4bit_weight_palettized_group32 \
    --compute-precision float16 --max-context-length 4096 --experimental --output-name <name>_ios_pure4bit
xcrun coreai-build compile exports/<name>_ios_pure4bit/<name>_ios_pure4bit.aimodel \
    --platform iOS --preferred-compute neural-engine --architecture h18p --output <tmp> && \
    assemble <tmp> → exports/<name>_ane_pure4bit/
```
Note: linear INT4 SIGSEGVs the ANE pre-compiler → ANE must use `4bit_weight_palettized_group32`. ARCH = `h18p`
(iPhone 17 Pro).

## Verify each bundle (your deliverable)
- `exports/<name>_gpu/` and `exports/<name>_ane_pure4bit/` each have: a `*.h18p.aimodelc` dir + `metadata.json`
  (with `assets.main` = the aimodelc filename) + `tokenizer/`.
- `coreai-build compile` exited 0. (Optional Mac-column sanity: bench the **macOS** bundle with coreai-models'
  `llm-benchmark` — allowed, it's a macOS artifact.)

## Hand-off
Write a checklist: `model → exports/<name>_gpu , exports/<name>_ane_pure4bit (ready ✓/✗)`. The gap-fill session
reads `methodology/gap-fill-session.md` and takes it from there.
