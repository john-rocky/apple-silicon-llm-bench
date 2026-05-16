## What this PR adds

- Device(s):
- Runtime(s):
- Model(s):
- Task(s):

(If this PR is wiring a new runtime / adapter or tooling, describe instead.)

## Checklist

- [ ] JSONLs added under `results/raw/` with the `<device>-<runtime>-<model>-<task>-runN.jsonl` convention
- [ ] Device label added to `scripts/render_results.py::DEVICE_DISPLAY` if new
- [ ] `devices/<device>.md` page added if new
- [ ] `python scripts/render_results.py` re-run and `RESULTS.md` committed in the same PR
- [ ] `python scripts/render_results.py --check` is clean locally
- [ ] No hand-edits inside the `<!-- BEGIN: generated ... -->` / `<!-- END: ... -->` block of `RESULTS.md`
- [ ] If a run failed, added a row to `RESULTS.md`'s "Failed runs" section with the upstream cause

## Methodology sanity

- [ ] Same prompt as Task A (`"Explain what on-device AI means in simple terms."`), temperature 0.0, max 128 tokens
- [ ] Device plugged in, idle, no concurrent benchmarks
- [ ] `n ≥ 3` for short-chat rows that are meant to be apples-to-apples (otherwise OK to land as `n = 1` with a note)

## Anything weird worth flagging
