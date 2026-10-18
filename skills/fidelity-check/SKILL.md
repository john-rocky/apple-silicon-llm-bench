---
name: fidelity-check
description: Run one LiteRT-LM text bundle (.litertlm) through the LiteRT-LM #3913 copy-fidelity cases on a Mac — PowerBeef's runner on the LiteRT-LM v0.18.0 Swift package, gpu fp16 as the arm under test, gpu fp32 and cpu as controls — recount the raw records, and write the claims table with every number traceable to count.txt. Use for a new export, a patched file or a new runtime build when the question is "does the GPU copy a 48-digit number exactly, also past token position 2048?". Cold-start: this file, the scripts next to it and a bundle path are the only inputs.
---

# fidelity-check

One bundle in, one claims table out. The table says, per arm, how many of PowerBeef's #3913 cases the
bundle copies exactly, with the prompt lengths, the first wrong digit's position, the engine identity
and the file's sha256. It does not say why; root cause belongs to the thread, not to this table.

## 0. What is measured

- Task: the prompt holds a 48-digit `REFERENCE:` number and digit-free filler text; the model is asked to
  reply with the digits only. `exact` = the reply's digits equal the reference, digit for digit.
- Case sets (`cases/*.json` of PowerBeef's research folder; one token per digit, so the 48 reply digits sit
  at token positions prefill−1 … prefill+46): `long3` 3 cases (smoke), `short` 15 (237–1639 prompt tokens),
  `baseline` 42 (1705–2592), `long` 25 (2094–3983), `cross4096` 30 (3983–4449), `xl` 18 (5995–29960, needs
  `maxNumTokens` 32768 — not covered here).
- Arms: `gpu fp16` = the arm under test (the runtime's default activation type; LiteRT-LM v0.18.0 runs
  text models at fp16 on the GPU unless the bundle says otherwise), `gpu fp32` = the precision control,
  `cpu fp32` = the backend control (XNNPACK; the engine computes in fp32 there whatever the label).
- Decoding: greedy (`topK 1, topP 1, temperature 0`), `maxOutputTokens 80`, `maxNumTokens 8192` = total
  context (prompt + reply must fit), one `Conversation` per case, one runner process per arm.

## 1. Inputs

| name | meaning | example |
|---|---|---|
| `BUNDLE` | path of the `.litertlm` to test (must load on LiteRT-LM v0.18.0; prefill and decode signatures; context ≥ 8192 or the `long` prompts do not fit) | `~/models/x/gemma-4-E2B-it-ropecomposite.litertlm` |
| `LABEL` | `<model>-<file>`: a model token and a file token, one dash between them, no other dash or underscore (count.py splits on the first dash) | `E2B-ropecomposite` |
| `W` | a work dir of your own (runner source + build, the bundle symlink, raw, logs, a Python venv); ≥ 3 GB free. It may already hold the caller's harness files (a prompt, a run log); leave them alone | `~/models/litertlm/fidelity-check-x` |
| `SETS` | case sets to run, smoke first | `long3` then `long` (add `short baseline cross4096` when the question is wider than 2048) |
| `ARMS` | `<cases>:<backend>:<act>` triples | `long:gpu:fp16 long:gpu:fp32 long:cpu:fp32` |
| `REF` (optional) | another run's raw dir in the same naming, for the prompt-token and text comparison | `results/fidelity/claims-gemma4-v0.18.0-mac-rope/raw` (the public Gemma 4 files on the same runner) |

## 2. Prerequisites (check before step 3; stop and report if one is missing)

- macOS on Apple Silicon with a Swift toolchain: `swift --version` prints a version (Xcode 26/27 toolchains work).
- Network once, for the runner source (GitHub archive) and the v0.18.0 xcframeworks (SwiftPM binary targets).
- `python3` (3.10+; count.py is stdlib). For the graph probe: a Python env with `ai-edge-litert`,
  `litert-lm-builder` and `flatbuffers`. Do not hunt for one: make it inside the work dir,
  `/opt/homebrew/bin/python3.12 -m venv "$W/venv" && "$W/venv/bin/pip" install ai-edge-litert==2.3.0 litert-lm-builder==0.18.0 flatbuffers`
  (about a minute; python3.13 / 3.14 work as well — the wheels exist), and run the probe with `"$W/venv/bin/python" -I`.
- The Mac measurement window. If `~/code/standup/tools/quiet/quiet_hold.py` exists on this machine, the
  scripts use it: `run_arms.sh` takes one window per call and `fetch_runner.sh` builds through
  `quiet_wait.py`. Keep one `run_arms.sh` call at ≤ 15 minutes of work (GPU arms take ~1 min per 25 cases
  for a 2B model, cpu arms a few minutes). Never run anything heavy of your own outside your window.
  Without those tools, make sure no other benchmark or export runs on the machine during the arms.
- Disk: the runner writes a weight / program cache next to the bundle (`W/models/cache/`, up to the bundle's size).

## 3. Procedure

Work from the repo root; `S` = `skills/fidelity-check/scripts`.

1. **Runner.** `S/fetch_runner.sh "$W"` → downloads PowerBeef's research folder at commit 39d5a014
   (LiteRT-LM v0.18.0 Swift package) into `W/LiteRT-LM` (the directory must be named `LiteRT-LM`, the
   script does that), checks the eight case files against `S/cases.SHA256SUMS`, builds
   `W/LiteRT-LM/research/issue-3913/runner/.build/release/runner`, and prints the runner's sha256 and the
   sha256 of `libCLiteRTLM_mac.dylib`. Record both. The runner binary's sha256 differs from build to
   build (Swift builds are not reproducible); the dylib's is the runtime's identity. Expected dylib for
   v0.18.0: `9526db9ee35e7eb4a5a58ac1e53330032f22a13890bd4ee01818fb380a57a362` (a different value means a
   different LiteRT-LM binary: record it and say so in the claims; do not continue silently).
2. **Bundle.** `shasum -a 256 "$BUNDLE"`; `mkdir -p W/models && ln -s "$BUNDLE" W/models/$LABEL.litertlm`.
   Probe the graph (read-only, seconds): `"$W/venv/bin/python" -I S/composite_count.py "$BUNDLE" > "$W/logs/composite_count.txt"`
   → the verdict line `ROPE-LOWERING <composite|cast-chain|both|neither>` and the decode signature's counts
   go into the claims; the full output stays in `logs/`. `cast-chain` = the rotary angle is computed in the
   graph from an INT32→float CAST (the #3913 structure); `composite` = `odml.rope` / `odml.qkv_norm_rope`
   composites and no such chain. If nobody told you how the bundle was made, the claims say
   "made by: not recorded in this run (the file was given as the input)" — never guess a recipe.
3. **Smoke.** `S/run_arms.sh "$W" "$LABEL" long3:gpu:fp16` → `W/raw/long3__$LABEL__gpu-fp16.jsonl` with 3
   lines. Read it: every line parses as JSON, has `prefill` > 0, `decode` > 0 and an `output`. If all 3
   replies are empty or off-task (no digits at all), the bundle or its prompt template is the problem, not
   the GPU: stop and report with the engine log `W/logs/long3__…log`.
4. **Arms.** One call per group of ≤ 15 minutes, e.g.
   `S/run_arms.sh "$W" "$LABEL" long:gpu:fp16 long:cpu:fp32` then `S/run_arms.sh "$W" "$LABEL" long:gpu:fp32`.
   The script skips complete records (resume) and moves an incomplete record to `*.failed-<time>`; a
   `WATCHDOG` line in the log means the 900 s limit hit. Window times land in `W/logs/holds.txt`.
5. **Recount.** `python3 -I S/count.py "$W/raw" "$W/LiteRT-LM/research/issue-3913" [REF] > "$W/count.txt"`.
   Required: the line `integrity: OK …`. A `PROBLEM` line means a record is inconsistent (line count,
   runner `exact` vs recomputed, model / backend fields vs file name): fix the cause and rerun that arm;
   never edit a record.
6. **Claims.** Write `claims.md` from the template in §5. Every number is copied from `count.txt` (name the
   section in a comment or footnote when it is not obvious); sha256 from step 2; runner and dylib sha256
   from step 1; engine identity lines from count.txt §6; window times from `holds.txt`.
7. **Keep** `raw/`, `logs/`, `count.txt`, `claims.md`, `SHA256SUMS.txt` (bundle, runner, dylib),
   `W/LiteRT-LM/` (source + build, reusable) and the symlink `W/models/$LABEL.litertlm`. Delete
   `W/models/cache/` (the runner's weight / program caches, GBs), `W/dl/` and `W/venv/`. Never delete the
   bundle itself. A number without its raw record is not a measurement.

## 4. Stop conditions (report, do not improvise)

- `fetch_runner.sh` fails to build, or the case-file checksums differ.
- The engine does not load the bundle (the log's first lines; a `maxNumTokens` / context error means the
  bundle's cache is smaller than 8192 — say so; the `long` set cannot be run on it).
- Smoke replies carry no digits (§3 step 3).
- `count.txt` shows `PROBLEM` lines, or "prefill identity across arms" is not N/N (the same case id got
  different prompt token counts in two arms = the arms did not see the same prompt).
- A `WATCHDOG` line, or a window could not be taken within the time you have (quiet_hold blocks; say
  how long you waited).
- Anything that would make you change a record, the cases, or the runner's settings to get a result.

## 5. Claims template

```markdown
# Fidelity claims — <model file>, LiteRT-LM v0.18.0, Mac (<chip>): PowerBeef's #3913 cases

> One-paragraph verdict: exact replies / cases per arm, where the first wrong digit sits (count.txt §4),
> and the controls. No cause.

File: `<name>` sha256 `<64 hex>` (made by: <export or patch command, tool versions>); graph probe:
`ROPE-LOWERING <verdict>` (<composite / cast counts of the decode signature>).
Runner: PowerBeef/agello-litert-lm research/issue-3913 @ 39d5a014, runner sha256 `<…>`, `libCLiteRTLM_mac.dylib`
sha256 `<…>` (LiteRT-LM v0.18.0 xcframework); greedy, `maxOutputTokens 80`, `maxNumTokens 8192`, one process per arm.

| file | backend-precision | `<set>` (<min>–<max> tok) | … |
|---|---|---|---|
| <label> | gpu-fp16 | a/b | … |
| <label> | gpu-fp32 | … | … |
| <label> | cpu-fp32 | … | … |

- Prompt token counts: "prefill identity across arms N/N"; vs the reference raw dir: "<n> equal, <m> differ"
  (count.txt prints the reference lines only when a shared case id exists; `long3` has none in PowerBeef's folder).
- First wrong digit (fp16 failures): positions <min>–<max>; highest last-digit position among exact replies <p>.
- Text identity: <the lines of count.txt §3 that matter: fp16 vs fp32 same file, cpu vs gpu>.
- Engine (count.txt §6): `Selected adapter: …`, `RegisterAccelerator … GPU WebGPU` / `XNNPACK CPU accelerator registered.`
- Windows: <taken–released per run_arms.sh call, from holds.txt>. Raw: `raw/`, logs: `logs/`, recount: `count.txt`.

## Not claimed
Root cause, other devices, other runtimes, speed (the runner's tok/s are recorded, not benchmarked).
```

## 6. Traps already paid for

- The runner's package resolves only under a parent directory named `LiteRT-LM` (SwiftPM path dependency
  identity = directory name). `fetch_runner.sh` extracts there; do not move the tree.
- The GitHub archive's sha256 changes between downloads (the fork was renamed); verify the case files.
- The runner writes its cache next to the model: pass a symlink in `W/models/`, never a Hugging Face snapshot path.
- `fp16` as the runner's third argument means "runtime default" (`activationDataType: nil`); `fp32` forces
  FLOAT32. The record's `activation` field copies the argument string.
- `maxNumTokens` is total context. A bundle exported with a 4096 cache cannot run `long` at 8192.
- The hook that guards the measurement window reads command text: `python3` plus words like `probe`,
  `export`, `bench` in the same command is denied while another lane's window is open. Run the Python
  scripts from a script file or outside a window; the paths in this repo's name contain `bench`.
- Compare prompt token counts per case id before trusting a cross-file comparison (`count.txt` top lines).
  A different chat template shows up there first.
- One runner process per arm; never two arms at once on one GPU.
