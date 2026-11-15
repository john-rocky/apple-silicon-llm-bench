# Fidelity claims run — Gemma 4 E2B / E4B, LiteRT-LM v0.18.0, Mac: the rotary-position patches from LiteRT-LM #3913

> On LiteRT-LM v0.18.0, Mac (M4 Max, macOS 27.0), `--backend gpu` (WebGPU delegate on the Metal
> adapter), default fp16 activations: the unmodified `gemma-4-E2B-it.litertlm` and
> `gemma-4-E4B-it.litertlm` return the 48-digit reference exactly in every case whose reply ends
> before token position 2048 (`short` 15/15, `baseline` 18/18 at positions 1704–2044), and in
> 2 of 24 (E2B) and 1 of 24 (E4B) of the `baseline` cases whose reply starts at position 2011 or
> later. The `long` set (prompts 2094–3983 tokens) and the `cross4096` set (3983–4449) fail
> 0/25 and 0/30 with both models.
>
> The same two files patched with PowerBeef's `patch_rope.py --split` (block 2048) and with the
> block-512 variant from picitujeromanov's comment return every case exactly with fp16
> activations: 112/112 per model per patch (`short` 15, `baseline` 42, `long` 25, `cross4096` 30).
>
> With FLOAT32 activations the four patched files return the same text as the unpatched file on
> every `long` case (25/25 each).
>
> On this repo's own 49-case copy suite (10 runs per case) the unmodified files fail the same cases
> on v0.18.0 as on v0.17.1, case for case (E2B 6, E4B 7); the block-512 files fail 0 (E2B) and 1
> (E4B: one amount returned without its `$`, digits intact).

Cases, runner and patch script are PowerBeef's, from `PowerBeef/LiteRT-LM` branch `research/3913`
at 39d5a014 (`research/issue-3913/`): the runner is a Swift CLI on the LiteRT-LM v0.18.0 Swift
package (`CLiteRTLM_mac.xcframework` checksum `5f6ee68d…`), greedy (`topK 1, topP 1,
temperature 0`), `maxOutputTokens 80`, `maxNumTokens 8192`, one `Conversation.sendMessage` per
case, one runner process per arm. Each case asks the model to copy a 48-digit number placed before
a filler text; `exact` means the reply is the number, digit for digit. Every count below was
recounted from the jsonl with two separate scripts (`claims-gemma4-v0.18.0-mac-rope/count.py` is
one of them; `count.txt` next to it is its output, with his `research/issue-3913` folder given as
the second argument for the rows that compare with his records).

Model files (sha256 in `claims-gemma4-v0.18.0-mac-rope/SHA256SUMS.txt`, checked twice on the Mac):

| file | made by | sha256 | matches |
|---|---|---|---|
| `gemma-4-E2B-it.litertlm` | litert-community, revision b3ca0d2f | `181938105e…9a63c` | the HF blob |
| `gemma-4-E4B-it.litertlm` | litert-community, revision 2eee7ac3 | `0b2a8980ce…d52e0` | the HF blob |
| `gemma-4-E4B-it-rope2048.litertlm` | `patch_rope.py --split` on the E4B file | `4a691f036f…cb15f` | PowerBeef's `PowerBeef02/gemma-4-E4B-it-litert-lm-fp16-rope` (commit 13d5aa4c) |
| `gemma-4-E2B-it-rope2048.litertlm` | `patch_rope.py --split` on the E2B file | `0b89958c5d…57a4` | no published value |
| `gemma-4-E2B-it-rope512.litertlm` | the same script with `BLOCK = 512`, `BLOCKS = 64` | `a963992d90…40b17` | picitujeromanov's comment on #3913 |
| `gemma-4-E4B-it-rope512.litertlm` | the same script with `BLOCK = 512`, `BLOCKS = 64` | `6353a23321…f655c` | picitujeromanov's comment on #3913 |

## Exact replies per arm (PowerBeef's cases)

| model | file | activations | `short` (237–1639 tok) | `baseline` (1705–2592) | `long` (2094–3983) | `cross4096` (3983–4449) |
|---|---|---|---|---|---|---|
| E2B | original | fp16 (default) | 15/15 | 20/42 | 0/25 | 0/30 |
| E2B | original | FLOAT32 | 15/15 | 42/42 | 25/25 | 30/30 |
| E2B | rope2048 | fp16 | 15/15 | 42/42 | 25/25 | 30/30 |
| E2B | rope512 | fp16 | 15/15 | 42/42 | 25/25 | 30/30 |
| E4B | original | fp16 (default) | 15/15 | 19/42 | 0/25 | 0/30 |
| E4B | original | FLOAT32 | 15/15 | 42/42 | 25/25 | 29/30 |
| E4B | rope2048 | fp16 | 15/15 | 42/42 | 25/25 | 30/30 |
| E4B | rope512 | fp16 | 15/15 | 42/42 | 25/25 | 30/30 |

Prompt lengths are the engine's prefill token counts; every case has one prefill value across all
arms (the patches do not change tokenization), and the 82 case ids that also appear in PowerBeef's
published M6 records have the same prefill count there.

### `baseline` by the positions of the reply digits

The 42 `baseline` cases put the 48 reply digits at positions prefill−1 … prefill+46. Grouped by
those positions (the groups of picitujeromanov's comment):

| model | file | activations | 1704–1751 (3 cases) | 1904–1951 (6, incl. 3 `late`) | 1952–2044 (9) | 2011–2638 (24, incl. 3 `late`) |
|---|---|---|---|---|---|---|
| E2B | original | fp16 | 3/3 | 6/6 | 9/9 | 2/24 |
| E4B | original | fp16 | 3/3 | 6/6 | 9/9 | 1/24 |
| E2B, E4B | original | FLOAT32 | 3/3 | 6/6 | 9/9 | 24/24 |
| E2B, E4B | rope2048, rope512 | fp16 | 3/3 | 6/6 | 9/9 | 24/24 |

So on this Mac every reply that ends at or before position 2044 is exact with the unmodified
files. The E4B original fp16 row has the same pass/fail verdict on all 42 cases as PowerBeef's
published M6 run of the same file (his `baseline__original__gpu-fp16.jsonl`: 19/42, same groups).
The three exact replies in the 2011–2638 group are the prefill-2012 cases `s124-seed1` and
`s124-seed3` (E2B) and `s124-seed3` (E4B), whose digits end at position 2058.

### Where the unmodified files first go wrong (fp16)

First-mismatch positions of the non-exact replies (position of the first wrong digit):

| model | `baseline` | `long` | `cross4096` |
|---|---|---|---|
| E2B original | 22 misses, 2051–2604 | 25 misses, 2097–3998 | 30 misses, 3989–4460 |
| E4B original | 23 misses, 2052–2604 | 25 misses, 2097–4001 | 30 misses, 3989–4467 |

No digit before position 2051 was wrong in any arm.

### Null test: FLOAT32 activations, patched file vs unmodified file

With FLOAT32 activations each patched file produced byte-identical replies to the unmodified file
on all 25 `long` cases: E2B rope2048 25/25, E2B rope512 25/25, E4B rope2048 25/25, E4B rope512
25/25 (and 25/25 exact in every one of those arms). With fp16 activations the patched files'
replies equal the unmodified file's FLOAT32 replies on every case except the one below.

### One FLOAT32 miss

`cross4096` case `s257-seed4` (prompt 4089 tokens): the unmodified E4B file with FLOAT32
activations replied with a different 48-digit number from the first digit on
(expected `241571346546417478885087663709890053284445109638`, got
`358435599326985997656161133549419581389590607599`). The same case is exact with the E4B rope2048
and rope512 files (fp16) and with the E2B file (FLOAT32); with E4B fp16 it fails from digit 12.
The wrong reply is, digit for digit, the reference number of the case that ran just before it in
the same process (`s257-seed3`; the runner opens a fresh conversation per case on one engine). In
four reruns of the same arm (the case alone three times, and the pair `s257-seed3` → `s257-seed4`
once in one process; `claims-gemma4-v0.18.0-mac-rope/raw/recheck/`) the case was exact every
time. Recorded as an observation; not chased further.

### CPU control

`--backend cpu` (XNNPACK, FLOAT32), PowerBeef's `long3` set (3 cases, 2094 / 2592 / 3983
tokens): E2B original 3/3, E4B original 3/3, with the same text as the gpu FLOAT32 arm on each.

## This repo's own suite (`evaldata/fidelity_suite_v1.jsonl`, tier core, 49 cases × 10 reps)

Runner `tools/litert-fidelity-runner/` built against v0.18.0 (`LITERTLM_VERSION=v0.18.0 ./setup.sh`,
`wrapper-v0.18.0.patch`), `--backend gpu --tier core --reps 10 --runtime-version v0.18.0`,
`maxNumTokens` 5760 (recorded per generation), greedy. Pass = the expected value appears verbatim
in the output; the scorer (`scripts/fidelity_score.py`) and a separate substring recount agree.

| model | file | arm | pass / generations | failing cases / 49 | failing cases by context bucket (128 / 1024 / 2048 / 4096) |
|---|---|---|---|---|---|
| E2B | original | gpu, 10 reps | 430 / 490 | 6 | 0 / 0 / 0 / 6 |
| E2B | original | cpu, 1 rep | 49 / 49 | 0 | 0 / 0 / 0 / 0 |
| E2B | rope512 | gpu, 10 reps | 490 / 490 | 0 | 0 / 0 / 0 / 0 |
| E2B | rope512 | cpu, 1 rep | 49 / 49 | 0 | 0 / 0 / 0 / 0 |
| E4B | original | gpu, 10 reps | 420 / 490 | 7 | 0 / 0 / 1 / 6 |
| E4B | rope512 | gpu, 10 reps | 480 / 490 | 1 | 0 / 0 / 1 / 0 |

- The unmodified files fail exactly the cases they failed on v0.17.1 (`claims-gemma4-v0.17.1-mac.md`),
  case for case: E2B `copy-amount-c4096-mid-de-50`, `copy-date_iso-c4096-mid-en-42`,
  `copy-id_digits-c4096-mid-de-50`, `copy-id_digits-c4096-mid-en-42`, `extract-mixed-c4096-spr-en-40`,
  `extract-mixed-c4096-spr-en-41`; E4B the same six minus `copy-date_iso-c4096-mid-en-42`, plus
  `copy-date_text-c4096-mid-de-50` and `copy-id_digits-c2048-mid-de-50` (prompt 2,306 tokens). Every
  failing case fails on all 10 reps; measured prompt lengths of the failing cases 2,306–4,588 tokens.
- The block-512 E4B file passes all seven of those and fails one other case, `copy-amount-c2048-ear-en-30`
  (prompt 1,579 tokens): it returns `2,753,770.56` where `$2,753,770.56` is expected, on 10 of 10 reps
  (the scorer calls it FORMAT_DRIFT: the digits are intact). The unmodified E4B file and both E2B files
  return `$2,753,770.56`.
- Every gpu arm returned the same output on all 10 reps of every case.
- cpu (XNNPACK) control, one rep per case: the E2B original and E2B rope512 files pass all 49
  (the E2B rope512 file returns `$2,753,770.56` on `copy-amount-c2048-ear-en-30` on the cpu). The E4B
  cpu arms were not run (the shared window queue had grown to 12–24 minutes per arm), so the cpu
  side of the E4B rope512 case above is not measured.

Raw: `claims-gemma-4-{E2B,E4B}-{original,rope512}-v0.18.0-mac-gpu.jsonl` and
`claims-gemma-4-E2B-{original,rope512}-v0.18.0-mac-cpu.jsonl` (next to this file; one JSON line per
generation with the output, prefill tokens and `maxNumTokens`).

## Not claimed

Root cause (the observation is that the patched files remove every fp16 failure of PowerBeef's cases
on this Mac; why is in PowerBeef's analysis on #3913), comparison with any other runtime, incidence
on other devices, and anything about the sets not run here (`xl`, `mtp`, `speed`). The runner's own
tok/s medians (prefill / decode, `baseline`): E2B original 6668 / 131, rope2048 6618 / 130, rope512
6571 / 132; E4B original 2135 / 82, rope2048 2130 / 82, rope512 2125 / 83 — recorded by the runner,
not a speed benchmark. The gpu arms ran on the WebGPU delegate with the Metal adapter, per the
engine log (`Selected adapter: Apple M4 Max, arch=metal-3, vendor=apple, backend=Metal,
adapterType=Integrated GPU`, `RegisterAccelerator: … name=GPU WebGPU`, `delegate_webgpu.cc`; the
engine log has no version line, the version is the xcframework checksum above).

## Run record

JST, 2026-10-15. Each arm is one runner process. The machine's shared measurement window was held
by this run for the duration of each group below, so no other benchmark ran alongside. Two E4B
original fp16 arms (`short`, `baseline`) ran at 07:12:08–07:13:44 in a gap between other runs, before
the window form was adopted; everything else ran inside the windows listed.

| window | taken | released | arms |
|---|---|---|---|
| E4B original fp16 (`long`, `cross4096`) | 07:22:37 | 07:24:47 | 2 |
| E4B rope2048 fp16 | 07:25:49 | 07:29:50 | 4 |
| E4B rope512 fp16 | 07:35:47 | 07:39:26 | 4 |
| E2B original fp16 | 07:51:48 | 07:53:35 | 4 |
| E2B rope2048 fp16 | 07:54:33 | 07:56:06 | 4 |
| E2B rope512 fp16 | 08:04:05 | 08:05:39 | 4 |
| E4B original FLOAT32 | 08:07:42 | 08:11:48 | 4 |
| E2B original FLOAT32 | 08:16:37 | 08:18:22 | 4 |
| null test, `long` FLOAT32, four patched files | 08:30:15 | 08:33:09 | 4 |
| cpu `long3`, E4B and E2B original | 08:47:56 | 08:49:10 | 2 |
| recheck of `s257-seed4` (E4B original FLOAT32) | 08:59:27 | 08:59:45 | 4 processes |
| own suite, smoke | 09:13:26 | 09:13:37 | 1 |
| own suite, E2B original gpu | 09:33:51 | 09:37:27 | 1 |
| own suite, E4B original gpu | 09:59:23 | 10:08:24 | 1 |
| own suite, E2B rope512 gpu | 10:23:17 | 10:27:02 | 1 |
| own suite, E4B rope512 gpu | 10:50:49 | 10:59:50 | 1 |
| own suite, E2B original cpu and E2B rope512 cpu (1 rep) | 11:17:16 | 11:23:47 | 2 |

No timeouts, no errors in the recorded arms (one earlier `long` E4B original fp16 run was killed
after 21 cases when another job's measurement window opened, and rerun in full inside the first window; its
partial file is not included). Engine logs: `claims-gemma4-v0.18.0-mac-rope/logs/` (the complete
stderr of each runner process, with the home directory in the wrapper's header lines shortened to
`~`). Raw records: `claims-gemma4-v0.18.0-mac-rope/raw/<cases>__<model>-<file>__<backend>-<activations>.jsonl`,
one JSON line per case with `id`, `prefill`, `exact`, `expected`, `output`, `firstMismatchPosition`,
`prefillTPS`, `decodeTPS`.

## Repro

```bash
# PowerBeef/LiteRT-LM, branch research/3913 @ 39d5a014 — research/issue-3913/
python3 -m venv venv && venv/bin/pip install -r requirements.txt   # ai-edge-litert 2.3.0, litert-lm-builder 0.18.0
python3 patch_rope.py gemma-4-E4B-it.litertlm --split gemma-4-E4B-it-rope2048.litertlm
# block 512: the same script with BLOCK = 512 and BLOCKS = 64 (claims-gemma4-v0.18.0-mac-rope/patch_rope_512.diff)
(cd runner && swift build -c release)                               # LiteRT-LM v0.18.0 Swift package
runner/.build/release/runner <model>.litertlm gpu fp16 cases/baseline.json > baseline__<label>__gpu-fp16.jsonl
runner/.build/release/runner <model>.litertlm gpu fp32 cases/long.json     > long__<label>__gpu-fp32.jsonl
runner/.build/release/runner <model>.litertlm cpu fp32 cases/long3.json    > long3__<label>__cpu-fp32.jsonl

# own suite (this repo)
LITERTLM_VERSION=v0.18.0 tools/litert-fidelity-runner/setup.sh
litert-mac-verify gemma-4-E4B-it-rope512.litertlm --suite evaldata/fidelity_suite_v1.jsonl \
  --out claims-gemma-4-E4B-rope512-v0.18.0-mac-gpu.jsonl --backend gpu --tier core --reps 10 --runtime-version v0.18.0
python3 scripts/fidelity_score.py results/fidelity/claims-gemma-4-*-v0.18.0-mac-*.jsonl --report /tmp/report.md
```
