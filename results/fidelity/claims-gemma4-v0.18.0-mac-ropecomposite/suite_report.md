# Backend fidelity report

Suite: `fidelity_suite_v1.jsonl` · gate: cpu pass-rate ≥ 90%

## litert-lm · gemma-4-E2B-it-ltdefault.litertlm · mac

**gpu** — 49 cases · 490 generations · DIGIT_CORRUPT 10, MISSING 90, PASS 390

_No cpu arm — backend-delta table skipped (no baseline)._

## litert-lm · gemma-4-E2B-it-ropecomposite.litertlm · mac

**cpu** — 49 cases · 49 generations · PASS 49

**gpu** — 49 cases · 490 generations · MISSING 10, PASS 480

### cpu → gpu delta (49 claim-eligible cases)

| case | cpu | gpu | failure classes |
|---|---|---|---|
| copy-date_iso-c128-mid-en-11 | 100% | 0% | {'MISSING': 10} |

**DIGIT_CORRUPT evidence (expected → got):**
- (none — regressions are FORMAT_DRIFT / MISSING only)
