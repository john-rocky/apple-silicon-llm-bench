# Backend fidelity report

Suite: `fidelity_suite_v1.jsonl` · gate: cpu pass-rate ≥ 90%

## litert-lm · gemma-4-E2B-it.litertlm · mac

**cpu** — 49 cases · 490 generations · PASS 490

**gpu** — 49 cases · 490 generations · DIGIT_CORRUPT 20, MISSING 40, PASS 430

**gpu-ctx8192** — 49 cases · 490 generations · DIGIT_CORRUPT 10, MISSING 60, PASS 420

**gpu-f32** — 49 cases · 490 generations · MISSING 10, PASS 480

### cpu → gpu delta (49 claim-eligible cases)

| case | cpu | gpu | failure classes |
|---|---|---|---|
| copy-amount-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-date_iso-c4096-mid-en-42 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c4096-mid-de-50 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| copy-id_digits-c4096-mid-en-42 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| extract-mixed-c4096-spr-en-40 | 100% | 0% | {'MISSING': 10} |
| extract-mixed-c4096-spr-en-41 | 100% | 0% | {'MISSING': 10} |

**DIGIT_CORRUPT evidence (expected → got):**
- `copy-id_digits-c4096-mid-de-50` rep 1: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 2: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 3: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 4: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 5: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 6: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 7: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 8: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 9: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-de-50` rep 10: `648733850143053` → `688338501430533`
- `copy-id_digits-c4096-mid-en-42` rep 1: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 2: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 3: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 4: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 5: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 6: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 7: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 8: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 9: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 10: `650094782026699` → `650947820266999`

### cpu → gpu-ctx8192 delta (49 claim-eligible cases)

| case | cpu | gpu-ctx8192 | failure classes |
|---|---|---|---|
| copy-amount-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-amount-c4096-mid-en-42 | 100% | 0% | {'MISSING': 10} |
| copy-date_iso-c4096-mid-en-42 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c4096-mid-en-42 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| extract-mixed-c4096-spr-en-40 | 100% | 0% | {'MISSING': 10} |
| extract-mixed-c4096-spr-en-41 | 100% | 0% | {'MISSING': 10} |

**DIGIT_CORRUPT evidence (expected → got):**
- `copy-id_digits-c4096-mid-en-42` rep 1: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 2: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 3: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 4: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 5: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 6: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 7: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 8: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 9: `650094782026699` → `650947820266999`
- `copy-id_digits-c4096-mid-en-42` rep 10: `650094782026699` → `650947820266999`

### cpu → gpu-f32 delta (49 claim-eligible cases)

| case | cpu | gpu-f32 | failure classes |
|---|---|---|---|
| copy-date_iso-c128-mid-en-11 | 100% | 0% | {'MISSING': 10} |

**DIGIT_CORRUPT evidence (expected → got):**
- (none — regressions are FORMAT_DRIFT / MISSING only)

## litert-lm · gemma-4-E4B-it.litertlm · mac

**cpu** — 49 cases · 490 generations · PASS 490

**gpu** — 49 cases · 490 generations · DIGIT_CORRUPT 20, MISSING 50, PASS 420

**gpu-ctx8192** — 49 cases · 490 generations · DIGIT_CORRUPT 20, FORMAT_DRIFT 10, MISSING 60, PASS 400

**gpu-f32** — 49 cases · 490 generations · PASS 490

### cpu → gpu delta (49 claim-eligible cases)

| case | cpu | gpu | failure classes |
|---|---|---|---|
| copy-amount-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-date_text-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c2048-mid-de-50 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| copy-id_digits-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c4096-mid-en-42 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| extract-mixed-c4096-spr-en-40 | 100% | 0% | {'MISSING': 10} |
| extract-mixed-c4096-spr-en-41 | 100% | 0% | {'MISSING': 10} |

**DIGIT_CORRUPT evidence (expected → got):**
- `copy-id_digits-c2048-mid-de-50` rep 1: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 2: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 3: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 4: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 5: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 6: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 7: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 8: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 9: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 10: `743486394834120` → `743434863948341`
- `copy-id_digits-c4096-mid-en-42` rep 1: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 2: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 3: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 4: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 5: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 6: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 7: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 8: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 9: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 10: `650094782026699` → `650009478202026`

### cpu → gpu-ctx8192 delta (49 claim-eligible cases)

| case | cpu | gpu-ctx8192 | failure classes |
|---|---|---|---|
| copy-amount-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-date_iso-c4096-mid-en-42 | 100% | 0% | {'MISSING': 10} |
| copy-date_text-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c2048-mid-de-50 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| copy-id_digits-c4096-mid-de-50 | 100% | 0% | {'MISSING': 10} |
| copy-id_digits-c4096-mid-en-42 | 100% | 0% | {'DIGIT_CORRUPT': 10} |
| copy-uuid-c4096-mid-en-42 | 100% | 0% | {'FORMAT_DRIFT': 10} |
| extract-mixed-c4096-spr-en-40 | 100% | 0% | {'MISSING': 10} |
| extract-mixed-c4096-spr-en-41 | 100% | 0% | {'MISSING': 10} |

**DIGIT_CORRUPT evidence (expected → got):**
- `copy-id_digits-c2048-mid-de-50` rep 1: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 2: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 3: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 4: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 5: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 6: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 7: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 8: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 9: `743486394834120` → `743434863948341`
- `copy-id_digits-c2048-mid-de-50` rep 10: `743486394834120` → `743434863948341`
- `copy-id_digits-c4096-mid-en-42` rep 1: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 2: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 3: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 4: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 5: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 6: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 7: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 8: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 9: `650094782026699` → `650009478202026`
- `copy-id_digits-c4096-mid-en-42` rep 10: `650094782026699` → `650009478202026`

### cpu → gpu-f32 delta (49 claim-eligible cases)
No regressions: gpu-f32 matches cpu on every claim-eligible case.
