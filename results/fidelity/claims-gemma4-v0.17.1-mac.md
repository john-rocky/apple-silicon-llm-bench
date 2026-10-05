# Fidelity claims run — Gemma 4 E2B / E4B, LiteRT-LM v0.17.1, Mac

> On LiteRT-LM v0.17.1, `gemma-4-E2B-it.litertlm`, Mac (M4 Max, macOS 27.0): gpu fails 6 of 49
> claim-eligible cases that cpu passes, with 20 DIGIT_CORRUPT generations
> (example: `648733850143053` → `688338501430533`).
>
> On LiteRT-LM v0.17.1, `gemma-4-E4B-it.litertlm`, same Mac: gpu fails 7 of 49 claim-eligible
> cases that cpu passes, with 20 DIGIT_CORRUPT generations
> (example: `743486394834120` → `743434863948341`).

Suite `evaldata/fidelity_suite_v1.jsonl`, tier core (49 cases) × 10 reps, greedy. Scorer output:
`claims-gemma4-v0.17.1-mac-report.md`. Every count below was also recounted from the jsonl
with a separate script (verbatim-substring pass test) and matches the scorer.

| model | arm | pass / 490 | failing cases / 49 | DIGIT_CORRUPT gens | max distinct outputs per case |
|---|---|---|---|---|---|
| E2B | cpu | 490 | 0 | 0 | 1 |
| E2B | gpu | 430 | 6 | 20 | 1 |
| E2B | gpu, maxNumTokens 8192 (control) | 420 | 7 | 10 | 1 |
| E2B | gpu, FLOAT32 activations | 480 | 1 | 0 | 1 |
| E4B | cpu | 490 | 0 | 0 | 1 |
| E4B | gpu | 420 | 7 | 20 | 1 |
| E4B | gpu, maxNumTokens 8192 (control) | 400 | 9 | 20 | 1 |
| E4B | gpu, FLOAT32 activations | 490 | 0 | 0 | 1 |

Failing cases by context bucket (gpu, default arm): E2B 0/6 at 128, 0/17 at 1024, 0/17 at 2048,
6/9 at 4096. E4B 0/6, 0/17, 1/17, 6/9. Measured prefill tokens per bucket (from the control
arms, which record them): 113–167, 791–1180, 1573–2306, 3126–4588.

Every arm returned the same output on all 10 reps of every case. The gpu failures are
therefore repeatable inside one engine, not intermittent, in this run.

Capability-limited cases (cpu < 90%): none.

## Outputs, verbatim

- E2B gpu, `copy-id_digits-c4096-mid-de-50`: expected `648733850143053`, got `REFERENCE: 688338501430533`
- E4B gpu, `copy-date_text-c4096-mid-de-50`: expected `5. Februar 2038`, got `REFERENCE: 5. 028`
- E2B gpu, `extract-mixed-c4096-spr-en-40`: expected order id `ORD-Y5VW-74744-S` plus three more
  fields, got ```` ```json\n{\n  "orderid": "ORD-Y5VW-744-S",\n}\n``` ````

## Controls

- **Context sizing.** The main arms ran with `maxNumTokens` 5760; the largest measured prompt is
  4588 tokens. Raising it to 8192 does not remove the failures (7 and 9 failing cases).
- **FLOAT32 activations** (`litert_lm_engine_settings_set_activation_data_type(settings, 0)`,
  the setting reported in LiteRT-LM #2814). Every case that fails on the default gpu arm passes.
  The setting changed outputs on 8 of 49 cases per model, so it reached the engine. One new
  failure on E2B: `copy-date_iso-c128-mid-en-11` returns the filler line after the reference
  line instead of `2033-04-09`, on 10 of 10 reps; the default gpu arm and cpu pass it.
- Both model files match the current `litert-community` revisions (E2B `b3ca0d2f`, E4B `2eee7ac3`).

## Not claimed

Root cause, comparison with any other runtime, incidence on other devices or OS versions.
Which GPU accelerator the Mac build selects for `--backend gpu` was not inspected.

## Run record

Arms ran one at a time, with no other benchmark process on the machine
(checked before each arm). JST.

| arm | start | end |
|---|---|---|
| E2B cpu | 10-05 23:06:49 | 23:39:48 |
| E2B gpu | 23:39:48 | 23:43:18 |
| E4B cpu | 23:43:18 | 10-06 01:00:09 |
| E4B gpu | 01:00:09 | 01:09:00 |
| E2B gpu ctx8192 | 01:09:48 | 01:13:40 |
| E4B gpu ctx8192 | 01:13:40 | 01:23:05 |
| E2B gpu f32 | 01:23:05 | 01:27:18 |
| E4B gpu f32 | 01:27:18 | 01:37:19 |

No timeouts, no errors, no resumes. Median prefill tok/s (cpu / gpu / gpu f32): E2B 477 / 6169 /
5130, E4B 190 / 1922 / 1660 — recorded by the runner, not a speed benchmark.

## Repro

Runner: `litert-mac-verify` built against swift-litert-lm with the xcframework pins moved to
v0.17.1; the exact diffs are in `claims-gemma4-v0.17.1-mac-runner.patch`.

```bash
litert-mac-verify gemma-4-E2B-it.litertlm --suite evaldata/fidelity_suite_v1.jsonl \
  --out claims.jsonl --backend gpu --tier core --reps 10 --runtime-version v0.17.1
# controls: add --suite-max-num-tokens 8192, or --activation f32
python3 scripts/fidelity_score.py results/fidelity/claims-*.jsonl \
  --report results/fidelity/claims-gemma4-v0.17.1-mac-report.md
# smoke report (report.md) regenerates with:
python3 scripts/fidelity_score.py results/fidelity/smoke-*.jsonl
```
