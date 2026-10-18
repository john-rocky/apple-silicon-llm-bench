# Fidelity claims run — Gemma 4 E2B exported with litert-torch's `odml.rope` composite, LiteRT-LM v0.18.0, Mac: the LiteRT-LM #3913 cases

> On LiteRT-LM v0.18.0, Mac (M4 Max, macOS 27.0), `--backend gpu` (WebGPU delegate on the Metal adapter),
> default fp16 activations: a Gemma 4 E2B file exported from `google/gemma-4-E2B-it` with litert-torch
> main's `export_hf` and `--use_rope_composite=True` returns the 48-digit reference exactly in every case of
> PowerBeef's #3913 sets: `short` 15/15, `baseline` 42/42, `long` 25/25 (prompts 2094–3983 tokens),
> `cross4096` 30/30 (3983–4449). The same export without that flag (same weights, same recipe, the default
> rotary path: INT32 → float CAST → MUL → COS / SIN inside the graph) returns 1/25 on `long`, with the first
> wrong digit at positions 2097–3998, like the public `gemma-4-E2B-it.litertlm` run in the same session
> (0/25, 2097–3998). With FLOAT32 activations and on the CPU both exports are 25/25 on `long`, with
> byte-identical text to each other, to the public file at FLOAT32, and to the composite file at fp16.
>
> This answers question 1 of litert-torch #1343 on this Mac: the composite-exported file passes the #3913
> cases at fp16 while the default export fails them.

Cases and runner are PowerBeef's (`PowerBeef/agello-litert-lm`, branch `research/3913` at 39d5a014,
`research/issue-3913/`): a Swift CLI on the LiteRT-LM v0.18.0 Swift package (`CLiteRTLM_mac.xcframework`
checksum `5f6ee68d…`, `libCLiteRTLM_mac.dylib` sha256 `9526db9e…a362` — the same binary as the 10-15 run),
greedy (`topK 1, topP 1, temperature 0`), `maxOutputTokens 80`, `maxNumTokens 8192`, one
`Conversation.sendMessage` per case, one runner process per arm. Each case asks the model to copy a
48-digit number placed before a filler text; `exact` means the reply is the number, digit for digit. Every
count below was recounted from the jsonl (`claims-gemma4-v0.18.0-mac-ropecomposite/count.py`; its output is
`count.txt` next to it, with his `research/issue-3913` folder as the second argument).

## Files (sha256 in `claims-gemma4-v0.18.0-mac-ropecomposite/SHA256SUMS.txt`)

| file | made by | sha256 | size | rotary path in the graph |
|---|---|---|---|---|
| `gemma-4-E2B-it-ropecomposite.litertlm` | litert-torch `export_hf`, `--use_rope_composite=True` (command below) | `7b1e36f0…0302` | 5,085,831,824 B | `odml.rope` composites: `decode` 50 (40 × {base 1e4, proportion 1.0} + 10 × {base 1e6, proportion 0.25}), `prefill_128` / `prefill_1024` 29 each; SIN / COS 0, no INT32 → float CAST in any main graph |
| `gemma-4-E2B-it-ltdefault.litertlm` | the same command without `--use_rope_composite` | `b0d12ee6…57ee` | 5,082,964,624 B | in each of the 3 signatures one INT32 → FLOAT32 CAST from `input_pos` feeding `MUL → CONCATENATION → COS`; SIN / COS 4; no `odml.rope` |
| `gemma-4-E2B-it.litertlm` (`original`) | litert-community, revision b3ca0d2f | `18193810…a63c` | 2,588,147,712 B | one CAST from `input_pos` feeding `MUL → RESHAPE → SIN` per signature; no `odml.rope` (read on 2026-10-17) |

The two exports share everything but the flag: `google/gemma-4-E2B-it` (HF snapshot 3e22461f, bf16
safetensors loaded as float32), litert-torch 0.10.0 editable at upstream main 0a81bf6 (2026-10-12),
`prefill_lengths 128,1024`, `cache_length 8192` (fixed; the public file's cache is a magic number the
runtime sets to 8192), `quantization_recipe dynamic_wi8_afp32` (the default), `externalize_embedder True`
(litert-torch requires it for Gemma 4), text only, `apply_gpu_composites False`, `fuse_qkv False`,
`fuse_gate_up False`, `use_qkv_norm_rope_composite False`, `use_sdpa_composite False`, `use_bool_mask False`.
The export log's `Gemma4 model patch applied … use_rope_composite=True` replaces all 35 attention modules
(70 `Replacing Attention` lines, prefill + decode traces); the control export replaces none. Graph counts:
composite `decode` 1296 ops, 292 composites (`odml.rms_norm` 242 + `odml.rope` 50); control `decode` 1534
ops, 242 composites (`odml.rms_norm` only), 4 SIN / COS; each `odml.rope` carries a decomposition subgraph
(`odml.rope.impl*`, 16–18 ops with SIN / COS) that is the CPU fallback body
(`claims-gemma4-v0.18.0-mac-ropecomposite/composite_count.py`, `rope_cast_probe.py`).

The exports differ from the public file in quantization (dynamic int8 weights here), tokenizer section
(HF tokenizer here, SentencePiece there), chat template revision (the HF canonical template of 2026-07-09
here) and the absence of the MTP drafter and the audio / vision sections. The rendered prompts are the
same: every one of the 112 case ids has one prefill token count across all arms here, and that count equals
the 10-15 run of the public file for all 112 and PowerBeef's published M6 records for all 82 shared ids
(`count.txt`, top lines). Neither export carries `prefer_activation_type`; the engine log says
`System's default activation type for Text decoder is fp16` for them (the public file carries
`prefer_activation_type: fp16`), so the `fp16` arms are fp16 for all three files.

## Exact replies per arm

| file | activations | `short` (237–1639 tok) | `baseline` (1705–2592) | `long` (2094–3983) | `cross4096` (3983–4449) |
|---|---|---|---|---|---|
| ropecomposite | gpu fp16 (default) | 15/15 | 42/42 | 25/25 | 30/30 |
| ropecomposite | gpu FLOAT32 | – | – | 25/25 | – |
| ropecomposite | cpu (XNNPACK, FLOAT32) | – | – | 25/25 | – |
| ltdefault | gpu fp16 (default) | – | – | 1/25 | – |
| ltdefault | gpu FLOAT32 | – | – | 25/25 | – |
| ltdefault | cpu (XNNPACK, FLOAT32) | – | – | 25/25 | – |
| original (public) | gpu fp16 (default) | – | – | 0/25 | – |

Prompt lengths are the engine's prefill token counts. Every reply of the exact arms is 49 decode tokens
(48 digits + end of turn); no reply hit the 80-token cap except 2 of the public file's failing replies.

`baseline` grouped by the positions of the 48 reply digits (prefill−1 … prefill+46), composite file,
gpu fp16: 1704–1751 3/3, 1904–1951 6/6, 1952–2044 9/9, 2011–2638 24/24 (the public file on this Mac on
10-15: 3/3, 6/6, 9/9, 2/24).

### Where the default export goes wrong (fp16)

- ltdefault gpu fp16 `long`: 24 misses, first wrong digit at positions 2097–3998 (prompts 2094–3983); the
  one exact reply is `s161-seed5` (prompt 2592 tokens, digits at 2591–2638).
- original gpu fp16 `long` (same session): 25 misses, first wrong digit 2097–3998 — the same 25 case ids as
  the 10-15 run of the same file, and the same replies (identical text 25/25).
- ropecomposite gpu fp16: 0 misses in 112 cases; the highest last-digit position among its exact replies is
  4495 (`cross4096`).

### Text identity (`count.txt` §3)

- Composite file, `long`: gpu fp16 = gpu FLOAT32 = cpu, 25/25 each pair.
- Control file, `long`: gpu fp16 vs gpu FLOAT32 identical in 1/25; cpu = gpu FLOAT32 25/25.
- Composite vs control at FLOAT32 (gpu and cpu): identical 25/25 — the composite changes nothing in exact
  arithmetic; both equal the public file's FLOAT32 replies of 10-15 (25/25 each).
- Composite fp16 vs the public file's fp16 replies of 10-15: `short` 15/15, `baseline` 20/42 (the public
  file fails 22 of them), `long` 0/25 and `cross4096` 0/30 (the public file fails all of those).

## This repo's own suite (`evaldata/fidelity_suite_v1.jsonl`, tier core, 49 cases)

Runner `tools/litert-fidelity-runner/` built against v0.18.0 (`LITERTLM_VERSION=v0.18.0 ./setup.sh`,
`wrapper-v0.18.0.patch`; binary sha256 `aa15cdb8…2df0`), `--tier core --runtime-version v0.18.0`,
`maxNumTokens` 5760, greedy. Pass = the expected value appears verbatim in the output
(`scripts/fidelity_score.py`, report in `claims-gemma4-v0.18.0-mac-ropecomposite/suite_report.md`; `count.txt`
§6 is an independent recount and agrees).

| file | arm | pass / generations | failing cases / 49 | failing cases by context bucket (128 / 1024 / 2048 / 4096) |
|---|---|---|---|---|
| ropecomposite | gpu, 10 reps | 480 / 490 | 1 | 1 / 0 / 0 / 0 |
| ropecomposite | cpu, 1 rep | 49 / 49 | 0 | 0 / 0 / 0 / 0 |
| ltdefault | gpu, 10 reps | 390 / 490 | 10 | 1 / 0 / 2 / 7 |

- The composite file's one failing case is `copy-date_iso-c128-mid-en-11` (prompt 129 tokens): the gpu
  returns the filler sentence after the `REFERENCE:` line, no digits, on 10 of 10 reps (scorer class
  MISSING); the cpu returns the date. The control export fails the same case the same way, and the public
  E2B file did the same on v0.17.1 with FLOAT32 activations (`claims-gemma4-v0.17.1-mac.md`). It is not a
  position-2048 failure and is reported as is.
- The control export's other nine failures are all at 2,294 tokens or longer (2048 bucket: `copy-date_text-c2048-mid-de-50`
  2,294 tok, `copy-id_digits-c2048-mid-de-50` 2,306 tok; 4096 bucket: 7 of 9 cases, 3,126–4,588 tok), with
  digits dropped or doubled (`743486394834120` → `74343486394834120`), the shape of the public file's
  failures (6/49 on 10-15, all at 2,306 tokens or longer). The composite file passes all 26 cases of the
  2048 and 4096 buckets on every rep.
- Every gpu arm returned the same output on all 10 reps of every case.

Raw: `claims-gemma-4-E2B-ropecomposite-v0.18.0-mac-{gpu,cpu}.jsonl` and
`claims-gemma-4-E2B-ltdefault-v0.18.0-mac-gpu.jsonl` (next to this file; one JSON line per generation).

## Not claimed

Root cause, and what the delegate's RoPE kernel computes in (the observation is that the composite-exported
file is exact at fp16 where the cast-chain exports are not; the kernel source is not public — see
litert-torch #1343). The behaviour on other GPUs, in particular the Apple7-class loss below position 2048
reported on the M1 Max in #3913 (not measured here). E4B (not exported). Comparison with any other
runtime. Speed: the runner's tok/s medians on `long` (prefill / decode) are composite fp16 5060 / 60.5,
control fp16 5045 / 61.2, composite FLOAT32 4303 / 58.2, control FLOAT32 4303 / 57.8, cpu 1087 / 28.2 and
1115 / 28.7 — recorded by the runner, not a speed benchmark; the public file's 6739 / 136.4 is a different
quantization and is not comparable.

## Run record

JST, 2026-10-18. Exports ran outside measurement windows through `quiet_wait.py` (`nice -n 10`);
measurement arms ran inside this lane's own windows (`quiet_hold.py`), one runner process per arm.

| step | from | to | note |
|---|---|---|---|
| export `ropecomposite` | 00:34:23 | 00:38:44 | 261 s wall; two short windows of another lane (`aa-b1-c1-checks`, 00:35:59–00:36:47 and 00:38:11–00:38:53) overlapped it |
| export `ltdefault` | 00:39:58 | 00:43:45 | 221 s wall |
| window A: `long` × {composite, control} × {gpu fp16, cpu} | 00:44:06 | 00:49:12 | 4 arms |
| window B: `long` FLOAT32 × 2, `long` public fp16, `short` / `baseline` / `cross4096` composite fp16 | 00:49:21 | 00:53:05 | 6 arms |
| own runner rebuilt at v0.18.0 (`quiet_wait.py`) | 00:53:16 | 00:53:37 | |
| window C: own suite, composite gpu 10 reps | 00:53:38 | 01:00:53 | 1 arm |
| window D: own suite, composite cpu 1 rep, control gpu 10 reps | 01:00:53 | 01:09:53 | 2 arms |

No timeouts, no errors. Engine logs: `claims-gemma4-v0.18.0-mac-ropecomposite/logs/` (the complete stderr
of each runner process, home directory shortened to `~`; the own-suite logs as `suite__*.log`). Raw records:
`claims-gemma4-v0.18.0-mac-ropecomposite/raw/<cases>__E2B-<file>__<backend>-<activations>.jsonl`, one JSON
line per case with `id`, `prefill`, `exact`, `expected`, `output`, `firstMismatchPosition`, `prefillTPS`,
`decodeTPS`. The gpu arms ran on the WebGPU delegate with the Metal adapter (engine log: `Selected adapter:
Apple M4 Max, arch=metal-3, vendor=apple, backend=Metal, adapterType=Integrated GPU`, `RegisterAccelerator:
… name=GPU WebGPU`); the cpu arms on XNNPACK (`XNNPACK CPU accelerator registered.`).

## Repro

```bash
# export (venv: litert-torch 0.10.0 editable at upstream main 0a81bf6, torch 2.12.0, transformers 5.19.0,
# ai-edge-litert-nightly 2.4.0.dev20261011, litert-converter 0.5.0, litert-lm-builder 0.18.0; a 3-file
# `tensorflow/lite/python/schema_py_generated` shim re-exporting ai_edge_litert's schema on PYTHONPATH,
# because model_ext/bonsai_flux2 imports it and a real TensorFlow cannot share the process with litert-converter)
python -m litert_torch.generative.export_hf --model=<google/gemma-4-E2B-it snapshot> --output_dir=<dir> \
  --task=text_generation --prefill_lengths=128,1024 --cache_length=8192 --externalize_embedder=True \
  --export_vision_encoder=False --export_audio_encoder=False --bundle_litert_lm=True --use_rope_composite=True
# control: the same command without --use_rope_composite=True

# PowerBeef's runner and cases (research/issue-3913 of PowerBeef/agello-litert-lm @ 39d5a014; the tree must sit under a
# directory named LiteRT-LM): skills/fidelity-check/scripts/fetch_runner.sh <work dir>
runner/.build/release/runner <model>.litertlm gpu fp16 cases/long.json > long__<label>__gpu-fp16.jsonl 2> long__<label>__gpu-fp16.log
runner/.build/release/runner <model>.litertlm gpu fp32 cases/long.json > long__<label>__gpu-fp32.jsonl
runner/.build/release/runner <model>.litertlm cpu fp32 cases/long.json > long__<label>__cpu-fp32.jsonl
python3 -I claims-gemma4-v0.18.0-mac-ropecomposite/count.py claims-gemma4-v0.18.0-mac-ropecomposite/raw <research/issue-3913> > count.txt

# own suite (this repo)
LITERTLM_VERSION=v0.18.0 tools/litert-fidelity-runner/setup.sh
litert-mac-verify gemma-4-E2B-it-ropecomposite.litertlm --suite evaldata/fidelity_suite_v1.jsonl \
  --out claims-gemma-4-E2B-ropecomposite-v0.18.0-mac-gpu.jsonl --backend gpu --tier core --reps 10 --runtime-version v0.18.0
python3 scripts/fidelity_score.py results/fidelity/claims-gemma-4-E2B-*-v0.18.0-mac-*.jsonl --report /tmp/report.md
```
