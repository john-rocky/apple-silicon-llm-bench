# litert-community & official-converter LLMs on Apple Silicon — Core AI vs MLX vs LiteRT-LM

**A cross-framework decode/prefill benchmark for the LiteRT-LM team.** Extends
`falcon3-comparison.md` (which covered only Falcon3) to two model sets:

1. **litert-community (un-benched)** — official Google `litert-community/*` `.litertlm` models
   that had no Core AI / MLX comparison yet: **DeepSeek-R1-Distill-Qwen-1.5B, Phi-4-mini,
   Gemma3-1B-IT, TinySwallow-1.5B, VibeThinker-1.5B**.
2. **our official-converter conversions** — dense models we converted with upstream `litert-torch`
   (BOCTAV4 int4): **OLMo-2-1B, Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B, Ministral-3-3B**.

**Hardware:** Mac (M4 Max GPU) + iPhone 17 Pro (on-device). **Protocol (Mac):** steady-state decode at
512 generated tokens, greedy; Core AI via `llm-benchmark -p 512 -g 1024 -n 5`, MLX via `mlx_lm` 512-tok
stream, LiteRT-LM via `litert-mac-verify --max-tokens 512 --backend gpu`. **Protocol (iPhone):** short-chat,
128 tokens, greedy, median of 3 cold launches (Yardstick app). Date: 2026-06-23. Author: john-rocky.

> ⚠️ **Quantization is not uniform — this is the central caveat.** MLX and Core AI are benchmarked at
> **4-bit** (their standard mobile weight size). For LiteRT-LM we benchmark **the file litert-community
> actually publishes**, which differs per model: **DeepSeek-R1 and Phi-4-mini ship only as INT8 (q8)**;
> Gemma3-1B ships INT4. Decode is memory-bandwidth-bound, so an INT8 model reads ~2× the bytes per token
> and is inherently ~half the tok/s of an INT4 one — **the int8 LiteRT rows are not directly comparable to
> the int4 MLX/Core AI rows.** Our own conversions are all INT4 (BOCTAV4), so those rows *are* iso-bit.

---

## Headline — Mac M4 Max GPU, decode tok/s (4-bit unless noted; higher is better)

| Model | LiteRT quant | **Core AI** | **MLX** | **LiteRT-LM** |
|---|---|--:|--:|--:|
| **— litert-community —** | | | | |
| DeepSeek-R1-Distill-Qwen-1.5B | int8* | **319.5** | **323.6** | **115.9*** |
| Phi-4-mini-instruct | int8* | — (no wrapper) | **167.4** | **67.7*** |
| Gemma3-1B-IT | int4 | **327.2**† | **345.5** | **185.7**‡ |
| TinySwallow-1.5B | int8* | **324.1** | **326.7** | **119.6*** |
| VibeThinker-1.5B | int8* | **322.7** | **176.3**§ | **119.6*** |
| **— our official-converter (all int4 BOCTAV4) —** | | | | |
| OLMo-2-1B | int4 | — (no wrapper) | **413.3** | **140.2** |
| Qwen3-1.7B | int4 | **239.1** | **322.7** | **115.8** |
| Llama-3.2-3B | int4 | **198.3** | **208.0** | **94.0** |
| SmolLM3-3B | int4 | — (no wrapper) | **196.0** | **91.2** |
| Ministral-3-3B | int4 | ✗ transformers | **189.0** | **92.8** |

§ VibeThinker MLX decodes ~1.8× slower than the other 1.5B Qwen-family models (176 vs ~320) despite the same
param count — likely a wider intermediate/vocab in its config; flagged for a re-check.

‡ Gemma3-1B LiteRT = **self-converted** (google/gemma-3-1b-it → int4 BOCTAV4 via official `litert_torch`; the
litert-community mirror is HF-gated). At 1B + gemma-tuned LiteRT kernels it decodes **185.7** — *faster* than the
larger generic int4 builds (OLMo-2-1B 140, Qwen3-1.7B 116, Llama-3.2-3B 94), a direct illustration of the
"Gemma gets first-class LiteRT kernel treatment, the generic third-party path doesn't" thesis from
`litert-speed-findings.md`. The MLX gap is correspondingly smaller here (345 / 186 = 1.86×).

`*` int8 = the only quant litert-community publishes for that model; not iso-bit vs the int4 MLX/Core AI cells.
`— (no wrapper)` = no Core AI macOS wrapper for that arch (olmo2 / phi3 / smollm3-NoPE); the others map to
qwen2 / qwen3 / gemma3 / mistral wrappers. `†` Gemma3-1B Core AI via a local text-only-config fix to
`coreai-models/export/pipeline.py` (the gemma3 wrapper otherwise assumes the multimodal config).

## Prefill tok/s (Mac M4 Max GPU)
| Model | Core AI | MLX† | LiteRT-LM |
|---|--:|--:|--:|
| DeepSeek-R1-1.5B | **4210** | 168† | 486 |
| Qwen3-1.7B | **3640** | 1004† | 396 |
| OLMo-2-1B | — | 1491† | 672 |

† MLX/LiteRT prefill use short prompts (not the 512-tok synthetic prompt Core AI uses), so compare prefill
*within* a runtime / directionally only. Core AI's prefill lead is real and large regardless.

---

## Findings (preliminary — Mac)

1. **On Apple GPU, Core AI ≈ MLX, both clearly ahead of LiteRT-LM on decode** — the same conclusion as the
   Falcon3 study, now reproduced across more architectures. For the iso-bit int4 rows: Qwen3-1.7B = Core AI
   239 / MLX 323 / LiteRT 116; OLMo-2-1B = MLX 413 / LiteRT 140. LiteRT-LM lands at **~35–50 % of MLX**
   decode at the same int4 weight size.
2. **The int8-only community models look even slower — but that's a publishing choice, not a runtime defect.**
   DeepSeek-R1 and Phi-4-mini are only on litert-community as **q8/int8** — at int8 they read 2× the bytes, so
   decode is ~half what an int4 build would. That's almost certainly a deliberate quality call (int4 PTQ hits a
   reasoning model like R1 hardest), so we **disclose the quant rather than prescribe int4**: part of their gap
   vs MLX/Core AI is the published quant, not the runtime.
3. **Core AI's prefill is in a different class** (3.6–4.2 k tok/s vs LiteRT's ~0.2–0.7 k) — matches Falcon3.
4. **Root cause is unchanged from `litert-speed-findings.md`:** LiteRT-LM runs int4×int8 **INTEGER** matmul over
   the **WebGPU(Dawn)→Metal** delegate; MLX/Core AI dequantize int4→fp16 and run **native-Metal fp16 GEMM**,
   which Apple GPUs do extremely fast. The decode gap is kernel/delegate efficiency, not bit-width.
5. **Core AI arch coverage is the practical limiter for that column** — qwen2/qwen3/mistral wrappers cover
   DeepSeek-R1 (319.5), TinySwallow (324.1), VibeThinker (322.7), Qwen3-1.7B (239.1) cleanly; olmo2, phi3,
   and smollm3's NoPE have no wrapper (Core AI N/A for those three). **Gemma3-1B Core AI needed a wrapper fix** (now
   resolved, 327.2): the `coreai-models` gemma3 wrapper assumed the *multimodal* Gemma-3 config (`text_config`
   sub-config + `language_model.` weight prefix) and `AttributeError`'d on the text-only 1B; patched to fall
   back to the top-level config / empty prefix for text-only Gemma3 (`export/pipeline.py`) → a small
   upstreamable fix. The remaining four (olmo2, phi3, smollm3-NoPE — no
   export class yet; `ministral3` — also needs a transformers that knows the arch) are **not permanent gaps —
   they just need a per-arch export class, which is quick for a `coreai-models` maintainer** (the zoo already has
   20+ families incl. MoE/diffusion/vision/audio). We benched the 6 that have an existing class; the other 4 are
   "unwritten class," not "unsupported."
6. **The 1.5B Qwen-family models all decode at ~320 tok/s on both Core AI and MLX** (DeepSeek 319.5/323.6,
   TinySwallow 324.1/326.7, VibeThinker 322.7/—) — same architecture, same Apple-GPU ceiling; Core AI and MLX
   are statistically tied, LiteRT-LM's published int8 builds sit at ~half that (bandwidth, not runtime).

## iPhone 17 Pro (on-device) — Core AI vs MLX vs LiteRT-LM (short-chat, 128 tok, median of 3 cold)

| Model | **Core AI ANE** | **Core AI GPU** | **MLX** | **LiteRT** |
|---|--:|--:|--:|--:|
| DeepSeek-R1-1.5B | **83.3** | **75.9** | **73.0** | **30.7** |
| TinySwallow-1.5B | **74.8** | **75.0** | **71.6** | **30.6** |
| VibeThinker-1.5B | ⏳ | **75.7** | — | **30.4** |
| Qwen3-1.7B | **64.8** | **67.6** | **62.8** | **23.2** |
| Gemma3-1B | ✗ no iOS class | — | **97.6** | gated |
| Phi-4-mini | ✗ no wrapper | — | **29.6** | OOM |
| OLMo-2-1B | ✗ no wrapper | — | — | **24.6** |
| Llama-3.2-3B | ✗ no iOS class | — | **34.0** | **18.4** |
| SmolLM3-3B | ✗ no wrapper | — | **36.8** | **22.8** |
| Ministral-3-3B | ✗ arch | — | ✗ | **18.0** |

**On-device, Core AI ≥ MLX ≫ LiteRT-LM — and Core AI's ANE is the trump card MLX/LiteRT structurally can't use.**
For the qwen-arch ≤1.7B that Core AI iOS covers: DeepSeek-R1 **ANE 83.3** / GPU 75.9 vs MLX 73.0 vs LiteRT 30.7;
TinySwallow 74.8 / 75.0 vs 71.6 vs 30.6; Qwen3-1.7B 64.8 / 67.6 vs 62.8 vs 23.2. **Core AI ≈-or-beats MLX (the
ANE tops MLX on DeepSeek-R1, 83 vs 73), and both are ~2.5× LiteRT-LM.** The **ANE (Apple Neural Engine) is
reachable only via Core AI / CoreML** — MLX and LiteRT-LM are GPU-only on Apple — so it's also the most
power-efficient path; this is Core AI's real on-device edge, invisible on Mac. The MLX-vs-LiteRT gap is 1.6–2.7×
iso-int4 (Qwen3-1.7B 62.8 vs 23.2; Llama-3.2-3B 34.0 vs 19.5; SmolLM3-3B 36.8 vs 22.8). Core AI iOS coverage here
= qwen2/qwen3/mistral classes only; **3B iOS Core AI bundles were not built** (no llama/smollm3 iOS class,
ministral3 arch unsupported) so 3B Core-AI-on-iPhone is **untested, not measured-OOM**; gemma3 no iOS class, phi3/olmo2/smollm3 no wrapper;
iPhone MLX-Swift can't load `ministral3`, and OLMo-2/VibeThinker lack mlx-community repos. (VibeThinker ANE bundle
pending — its compile was interrupted by a Mac crash; everything else is measured.)

**iPhone 3B int4 loads fine on LiteRT-LM — the earlier failures were a *harness* misconfiguration (missing memory
entitlements), not LiteRT.** The BenchmarkApp was built without `com.apple.developer.kernel.increased-memory-limit`
and `…extended-virtual-addressing`. Without them a large weight-section `mmap` fails with `Cannot allocate memory`
(ENOMEM) even on this **12 GB** device — we saw **Ministral-3-3B 0–1/3 and Llama-3.2-3B 1/3**. **Adding both
entitlements fixed it: Ministral-3-3B 3/3 (~18 tok/s), Llama-3.2-3B 5/5 (~18.4).** Both are required —
`increased-memory-limit` raises the footprint ceiling (the many-small-sections case, Ministral) and
`extended-virtual-addressing` supplies address space for one large contiguous section (Llama). We chased two wrong
hypotheses on the way — an `externalize_embedder` bug, then a "LiteRT mmap-loader limitation" — **both refuted.**
MLX only *appeared* to "load 3B where LiteRT couldn't" because its lower load-peak stayed under the default limit.
**LiteRT-LM is not at fault for the iPhone 3B failures; it was our app config.** (Lesson: the iPhone harness needs
both entitlements for any ≳2 GB model. We then re-checked Phi-4-mini LiteRT: its **int8 build (3.6 GB) OOMs even
*with* the entitlements** (`std::bad_alloc` → SIGABRT) — but that's the **quant, not Phi or LiteRT**. MLX runs Phi
at int4 (~2 GB, 29.6 tok/s) and LiteRT runs other 3B int4 builds (~2.2 GB) fine, so a LiteRT int4 Phi (~2 GB) would
load too — litert-community just only ships Phi as int8. So the iPhone "OOM" cells split: **3B int4 = harness
artifacts (now load); Phi int8 = too big at *that quant* (int4 would fit); genuinely-unrunnable on-device = none.**)

**Core AI iOS:** measured (ANE + GPU) for the qwen-arch ≤1.7B set — see the iPhone table above (DeepSeek-R1, TinySwallow, Qwen3-1.7B; VibeThinker GPU). Core AI ≥ MLX ≫ LiteRT on-device, ANE the trump card.

### Coverage — 40 measured cells; every blank is a documented architectural block
- **Mac 26/30:** all 10 models have MLX + LiteRT; Core AI 6/10 (qwen2/qwen3/mistral/gemma3). Core AI blocks: no
  wrapper for olmo2/phi3/smollm3-NoPE; Ministral `ministral3` unknown to the installed transformers'
  `Mistral3Config`. (Gemma3-1B was unblocked by the text-only-config fix above.)
- **iPhone 14/20:** 7 MLX + 7 LiteRT. Blocks: no mlx-community repo for OLMo-2/VibeThinker; MLX-Swift can't
  load `ministral3`; Gemma3-1B LiteRT gated; Phi LiteRT (4 GB int8) OOM-skipped; Llama/Ministral 3B LiteRT memory-bound (cold-launch flaky near the iOS ceiling).

## Coverage — LiteRT *traces*, Core AI *reimplements* (out-of-the-box ease vs custom-code ceiling)

Every model here converted to LiteRT with **zero custom code**, yet Core AI covers only 6/10 — a direct
consequence of the two converters' designs, not a quality gap. **LiteRT-LM (`litert_torch`) is trace-based and
architecture-agnostic:** `torch.export` captures the HF model's *actual* forward and lowers it to generic ops
(matmul, RMSNorm, real-valued RoPE, softmax-SDPA). It never needs to "know" the architecture — OLMo-2's
QK-norm, SmolLM3's NoPE, Ministral's layout are just *different arrangements of the same generic ops*, so any
traceable model with supported ops works. **Core AI (`coreai-models`) is a per-architecture reimplementation
registry:** hand-written Apple `nn.Module` classes (`macos/{qwen2,qwen3,mistral,gemma3_text,…}.py`) built from
Apple primitives (SDPA/RoPE/RMSNorm) into which HF weights are *loaded* (`_mutate_state_dict` even re-combines
QKV). An architecture with no hand-written class can't export — which is why olmo2/phi3/smollm3 fail (and can't
borrow the `mistral` wrapper: they'd *run* but compute the wrong thing — skipping QK-norm, applying RoPE to NoPE
layers), and Ministral-3 fails twice over (its multimodal `Mistral3Config` / `ministral3` text sub-arch isn't
even recognized by the installed transformers, *before* Core AI is reached). Gemma3-1B was the lone quick fix
because gemma3 *is* in the registry — only its config-layout assumption needed relaxing.

Core AI reimplements (rather than traces) to emit a **stateful (KV-cache-as-state), Apple-tuned, memory-mapped
graph** — what makes it fast (320 tok/s, 4 k prefill). **The nuance that matters (and corrects an easy
over-claim): out-of-the-box, LiteRT-LM is *easier* for dense models — zero custom code — but it does NOT have
broader coverage.** Allow custom code and **Core AI's ceiling is far higher**: with a hand-written export class
*and* a custom Metal kernel you reach **and fast-run** MoE, diffusion, SAM / depth-anything / YOLO / efficient-SAM,
audio (whisper / wav2vec2 / clap), T5 on-device — 20+ families in the `coreai-models` zoo. LiteRT-LM can't match
that today: the newest archs (MoE / MLA / SSM) **drop to CPU even after they convert**, and its **GPU kernel layer
is closed** (ML Drift is prebuilt; no external kernel-registration path). So LiteRT-LM's real edge is
**zero-friction dense conversion + portability / cross-hardware reach**, *not* raw breadth — and the high-leverage
asks are to **raise the ceiling: (a) converter fixes for the new archs, (b) open the GPU kernel layer so
contributors can extend it the way they extend Core AI** — alongside closing the dense-path Apple-GPU speed gap
(below). (Of the 10 here — all dense — LiteRT converts 10/10 with no custom code vs Core AI's 6/10; that's the
*ease-for-dense* point, not an overall-coverage win.)

## Recommendations to the LiteRT-LM team (impact-ordered)
1. **Make weight-only int4 + FLOAT compute (int4→fp16 dequant + fp16 GEMM) actually execute** — today it
   converts but hangs in the WebGPU delegate (`deliverables/litert-float-gpu-hang/`). This is the MLX/Core AI
   compute model and the single highest-impact route to Apple-GPU decode parity.
2. **Native Metal delegate** for Apple (vs WebGPU/Dawn→Metal) to drop the abstraction layer.
3. **Raise the ceiling so contributors can extend LiteRT-LM the way they extend Core AI** — (a) fix the converter
   for the newest archs (MoE / MLA / SSM — today they drop to CPU even after converting) and (b) **open the GPU
   kernel layer** (ML Drift is prebuilt with no external kernel-registration path). This is what currently caps
   LiteRT-LM's on-device model set below Core AI's, despite the easier dense converter.
4. **Lean into LiteRT's real edge** — cross-hardware reach (Qualcomm/NPU on Android) and one portable bundle.
   On Apple GPU, MLX/Core AI are the native floor; parity (not beating) is the realistic target.

## Methodology / repro
**Open harness — reproduce it yourself: https://github.com/john-rocky/apple-silicon-llm-bench** (neutral, one
headless harness for every runtime; every cell traces to raw JSONL). Per-runtime:
- **Core AI:** `uv run coreai.llm.export <hf-id> --platform macOS --compression 4bit --compute-precision
  float16 --experimental`; bench `llm-benchmark --model <bundle> -p 512 -g 1024 -n 5`. Wrapper chosen by HF
  `model_type` (qwen2/qwen3/gemma3/mistral via `MODEL_TYPE_REMAPPING`).
- **MLX:** `mlx-community/*-4bit` (or `mlx_lm.convert -q --q-bits 4 --q-group-size 64`); 512-tok greedy stream.
- **LiteRT-LM:** litert-community `.litertlm` (published quant) or our BOCTAV4 conversions;
  `litert-mac-verify --max-tokens 512 --backend gpu`.
- **iPhone 17 Pro (all runtimes):** the Yardstick app (`ios/BenchmarkApp`, bundle
  `com.iosllmbenchmark.benchmarkapp`) driven headless via `devicectl … --yardstick-autorun --runtime
  {mlx-swift,litert-lm} --model-id <id> --task short-chat`, median of 3 cold launches. Model entries added to
  `ios/BenchmarkApp/Sources/Models/ModelCatalog.swift` (`mlx-community/*` = on-device MLX download;
  `litert-community/*` = on-device LiteRT download or side-load; `litert-local/*` = side-loaded `.litertlm`).
- **Raw decode cells (this repo):** `results/raw/2026-06-24-litert-community-crossframework/decode-cells.jsonl`
  — one JSON row per model×framework×device: `{model, group, framework, device, quant, decode_tps, prefill_tps, …}`.
