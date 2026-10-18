#!/usr/bin/env python3
"""count.py — recount the raw records of a fidelity-check run (stdlib only; reads only).

    python3 -I count.py <raw dir> <research/issue-3913 dir> [reference raw dir] > count.txt

Reads <raw dir>/<cases>__<model>-<file>__<backend>-<precision>.jsonl (PowerBeef's #3913 runner, one JSON line
per case) and <raw dir>/../logs/*.log (the engine logs). The second argument is the runner's research folder
(cases/ for the line counts; results/mac/ for PowerBeef's published M6 rows). The optional third argument (or
env FIDELITY_REF_RAW) is another run's raw dir in the same naming; its prompt token counts and texts are compared
per case id (e.g. the public file's records: results/fidelity/claims-gemma4-v0.18.0-mac-rope/raw).

Integrity (every record): `expected` == the runner's LCG (referenceDigits(seed)); exactness recomputed from
`output` and `expected` (ASCII digits only) == the runner's `exact`; `firstMismatchPosition` == prefill − 1 +
firstMismatch; backend / activation / model fields match the file name (model == "<model>-<file>.litertlm").
Every number in a claims table comes from this output, never from a runner's stdout summary.
"""
import glob, json, os, re, statistics, sys
from collections import Counter, defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)
RAW = os.path.abspath(sys.argv[1])
UP = os.path.abspath(sys.argv[2])
REF = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.environ.get("FIDELITY_REF_RAW", "")
LOGS = os.path.join(os.path.dirname(RAW), "logs")
CASES_ORDER = ["long3", "short", "baseline", "long", "cross4096", "xl", "mtp", "speed"]
CONFIGS = ["gpu-fp16", "gpu-fp32", "cpu-fp32", "cpu-fp16"]
DIG = set("0123456789")


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def ref_digits(seed):
    m, a, c = (1 << 64) - 1, 6364136223846793005, 1442695040888963407
    s = (seed * a + c) & m
    out = []
    for _ in range(48):
        s = (s * a + c) & m
        out.append(str((s >> 33) % 10))
    return "".join(out)


def exact(r):
    return "".join(ch for ch in r["output"] if ch in DIG) == r["expected"]


def frac(rows):
    return f"{sum(exact(r) for r in rows)}/{len(rows)}"


def corder(c):
    return (CASES_ORDER.index(c) if c in CASES_ORDER else 99, c)


# ------------------------------------------------------------------ load + integrity
arms = {}  # (cases, model, file, config) -> {id: row}
problems = []
for path in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
    base = os.path.basename(path)[:-6]
    try:
        cases, label, config = base.split("__")
        model, file = label.split("-", 1)
    except ValueError:
        problems.append(f"unparsed file name {base} (want <cases>__<model>-<file>__<backend>-<precision>.jsonl)")
        continue
    rows = load(path)
    cj = os.path.join(UP, "cases", f"{cases}.json")
    if os.path.exists(cj):
        want = len(json.load(open(cj)))
        if len(rows) != want:
            problems.append(f"{base}: {len(rows)} lines, cases file has {want}")
    else:
        problems.append(f"{base}: no cases file {cj}")
    for r in rows:
        if r["expected"] != ref_digits(r["seed"]):
            problems.append(f"{base} {r['id']}: expected != referenceDigits(seed)")
        if exact(r) != r["exact"]:
            problems.append(f"{base} {r['id']}: recomputed exact {exact(r)} != runner {r['exact']}")
        if not r["exact"] and r.get("firstMismatchPosition") != r["prefill"] - 1 + r["firstMismatch"]:
            problems.append(f"{base} {r['id']}: firstMismatchPosition inconsistent")
        if r["backend"] + "-" + r["activation"] != config:
            problems.append(f"{base} {r['id']}: record backend/activation {r['backend']}/{r['activation']} != {config}")
        if r["model"] != f"{label}.litertlm":
            problems.append(f"{base} {r['id']}: record model {r['model']} != {label}.litertlm")
    arms[(cases, model, file, config)] = {r["id"]: r for r in rows}

models = sorted({k[1] for k in arms})
files = sorted({k[2] for k in arms})
present = sorted({k[0] for k in arms}, key=corder)
print(f"# count.py — recount of {RAW}")
print(f"arms {len(arms)}, replies {sum(len(v) for v in arms.values())}, models {models}, files {files}, case sets {present}")
print("integrity:", "OK (line counts, expected = LCG(seed), recomputed exact = runner exact, firstMismatchPosition, record model/backend/activation = file name)" if not problems else f"{len(problems)} PROBLEM(S)")
for p in problems:
    print("  PROBLEM", p)

# prefill identity: the same case id must have the same prefill in every arm (same prompt, same template)
pre = defaultdict(set)
for (cases, model, file, config), rows in arms.items():
    for i, r in rows.items():
        pre[(cases, i)].add(r["prefill"])
bad = {k: v for k, v in pre.items() if len(v) > 1}
print(f"prefill identity across arms: {len(pre) - len(bad)}/{len(pre)} case ids have one prefill value"
      + ("" if not bad else f"; differing: {sorted(bad.items())[:10]}"))
ref_rows = {}
if REF and os.path.isdir(REF):
    for path in sorted(glob.glob(os.path.join(REF, "*.jsonl"))):
        try:
            cases, label, config = os.path.basename(path)[:-6].split("__")
        except ValueError:
            continue
        ref_rows[(cases, label, config)] = {r["id"]: r for r in load(path)}
    refpre = {}
    for (cases, label, config), rows in ref_rows.items():
        for i, r in rows.items():
            refpre.setdefault((cases, i), set()).add(r["prefill"])
    same = [k for k in pre if k in refpre and pre[k] == refpre[k]]
    diff = [k for k in pre if k in refpre and pre[k] != refpre[k]]
    if same or diff:
        print(f"prefill vs the reference raw dir ({REF}, same case id): {len(same)} equal, {len(diff)} differ"
              + ("" if not diff else f" {[(k, sorted(pre[k]), sorted(refpre[k])) for k in diff[:8]]}"))
    else:
        print(f"prefill vs the reference raw dir ({REF}): no shared case id")
pb = {}
for path in glob.glob(os.path.join(UP, "results", "mac", "*__original__*.jsonl")):
    cases = os.path.basename(path).split("__")[0]
    for r in load(path):
        pb[(cases, r["id"])] = r["prefill"]
if pb:
    same = [k for k in pre if k in pb and pre[k] == {pb[k]}]
    diff = [k for k in pre if k in pb and pre[k] != {pb[k]}]
    if same or diff:
        print(f"prefill vs PowerBeef's M6 original records (same case id): {len(same)} equal, {len(diff)} differ"
              + ("" if not diff else f" {[(k, sorted(pre[k]), pb[k]) for k in diff[:6]]}"))
    else:
        print("prefill vs PowerBeef's M6 original records: no shared case id (his results/mac/ has no original-file record of these sets)")

# ------------------------------------------------------------------ 1. exact / total
print("\n## 1. exact / total (one line = one arm; a cell = exact replies / cases)\n")
print("| model | file | backend-precision | " + " | ".join(f"`{c}`" for c in present) + " |")
print("|---|---|---|" + "---|" * len(present))
for model in models:
    for file in files:
        for config in CONFIGS:
            cells = [frac(arms[(c, model, file, config)].values()) if (c, model, file, config) in arms else "–" for c in present]
            if any(x != "–" for x in cells):
                print(f"| {model} | {file} | {config} | " + " | ".join(cells) + " |")
for c in present:
    ps = [r["prefill"] for k, v in arms.items() if k[0] == c for r in v.values()]
    print(f"- `{c}`: prompts {min(ps)}–{max(ps)} tokens (engine prefill count)")
for (cases, model, file, config), rows in sorted(arms.items(), key=lambda kv: (corder(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
    dec = sorted(r["decode"] for r in rows.values())
    print(f"- {cases} {model}-{file} {config}: decode tokens {dec[0]}–{dec[-1]}, replies at the 80-token cap {sum(d >= 80 for d in dec)}")

# ------------------------------------------------------------------ 2. baseline groups (if run)
GROUPS = [("1704–1751", 1751), ("1904–1951 (incl. late)", 1951), ("1952–2044", 2044), ("2011–2638", 10 ** 9)]


def group_of(r):
    end = r["prefill"] + 46
    for name, hi in GROUPS:
        if end <= hi:
            return name


if any(k[0] == "baseline" for k in arms):
    print("\n## 2. `baseline` by reply digit positions (prefill − 1 … prefill + 46), exact / total\n")
    print("| model | file | backend-precision | " + " | ".join(n for n, _ in GROUPS) + " |")
    print("|---|---|---|---|---|---|---|")
    for model in models:
        for file in files:
            for config in CONFIGS:
                rows = arms.get(("baseline", model, file, config))
                if rows:
                    g = defaultdict(list)
                    for r in rows.values():
                        g[group_of(r)].append(r)
                    print(f"| {model} | {file} | {config} | " + " | ".join(frac(g[n]) if g[n] else "–" for n, _ in GROUPS) + " |")

# ------------------------------------------------------------------ 3. text identity
print("\n## 3. text identity (same case ids)\n")


def ident(a, b, what):
    keys = [k for k in b if k in a]
    if keys:
        print(f"- {what}: identical text {sum(a[k]['output'] == b[k]['output'] for k in keys)}/{len(keys)}")


for model in models:
    for file in files:
        for cases in present:
            ident(arms.get((cases, model, file, "gpu-fp16"), {}), arms.get((cases, model, file, "gpu-fp32"), {}), f"{model}-{file} gpu fp16 vs gpu fp32, `{cases}`")
            ident(arms.get((cases, model, file, "cpu-fp32"), {}), arms.get((cases, model, file, "gpu-fp32"), {}), f"{model}-{file} cpu fp32 vs gpu fp32, `{cases}`")
            ident(arms.get((cases, model, file, "gpu-fp16"), {}), arms.get((cases, model, file, "cpu-fp32"), {}), f"{model}-{file} gpu fp16 vs cpu fp32, `{cases}`")
    for cases in present:
        for config in CONFIGS:
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    ident(arms.get((cases, model, fa, config), {}), arms.get((cases, model, fb, config), {}), f"{model}: {fa} vs {fb}, {config}, `{cases}`")
for (cases, label, config), b in sorted(ref_rows.items()):
    for model in models:
        for file in files:
            ident(arms.get((cases, model, file, config), {}), b, f"{model}-{file} vs reference {label}, {config}, `{cases}`")

# ------------------------------------------------------------------ 4. failures by position
print("\n## 4. failures by position (first wrong digit = firstMismatchPosition; the reply digits sit at prefill − 1 … prefill + 46)\n")
for model in models:
    for file in files:
        for config in CONFIGS:
            allf, oks = [], []
            for cases in present:
                rows = arms.get((cases, model, file, config))
                if not rows:
                    continue
                fails = [r for r in rows.values() if not exact(r)]
                oks += [r for r in rows.values() if exact(r)]
                allf += fails
                if fails:
                    pos = sorted(r["firstMismatchPosition"] for r in fails)
                    print(f"- {model}-{file} {config} `{cases}`: {len(fails)} fail, firstMismatchPosition {pos[0]}–{pos[-1]};"
                          f" prefill {min(r['prefill'] for r in fails)}–{max(r['prefill'] for r in fails)}; ids {[r['id'] for r in fails][:8]}")
                else:
                    print(f"- {model}-{file} {config} `{cases}`: 0 fail")
            if allf:
                print(f"  = {model}-{file} {config}: min firstMismatchPosition {min(r['firstMismatchPosition'] for r in allf)}"
                      f" ({min(allf, key=lambda r: r['firstMismatchPosition'])['id']});"
                      f" highest last-digit position among exact replies {max((r['prefill'] + 46 for r in oks), default='-')}")
            elif oks:
                print(f"  = {model}-{file} {config}: 0 fail; highest last-digit position among exact replies {max(r['prefill'] + 46 for r in oks)}")

# ------------------------------------------------------------------ 5. speed (recorded by the runner, not a benchmark)
print("\n## 5. runner-recorded prefill / decode tok/s, medians (recorded by the runner, not a speed benchmark)\n")
for cases in present:
    for model in models:
        for file in files:
            for config in CONFIGS:
                rows = arms.get((cases, model, file, config))
                if rows:
                    v = [r for r in rows.values() if "prefillTPS" in r]
                    if v:
                        print(f"- `{cases}` {model}-{file} {config}: prefill {statistics.median(r['prefillTPS'] for r in v):.0f},"
                              f" decode {statistics.median(r['decodeTPS'] for r in v):.1f}, seconds/reply {statistics.median(r['seconds'] for r in v):.2f} (n={len(v)})")

# ------------------------------------------------------------------ 6. engine identity
print("\n## 6. engine log lines (verbatim, deduplicated; count = logs containing the line)\n")
pat = re.compile(r"Selected adapter|delegate_webgpu|registered|[Vv]ersion|RegisterAccelerator|backend=|WebGPU|Metal|OpenCL|xnnpack|XNNPACK|engine loaded|WATCHDOG|magic_number|ctivation|[Ee]rror")
seen = Counter()
logs = sorted(glob.glob(os.path.join(LOGS, "*.log")))
for path in logs:
    lines = set()
    for l in open(path, errors="replace"):
        if pat.search(l):
            l = re.sub(r"^[IWE]\d{4} \S+ \d+ ", "", l.strip())          # absl prefix: level+date, time, thread
            l = re.sub(r"ptr=0x[0-9a-f]+", "ptr=…", re.sub(r"\d+\.\d+ s$", "<t> s", l))
            lines.add(l[:220])
    for l in lines:
        seen[l] += 1
print(f"logs: {len(logs)}")
for l, n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"- [{n}/{len(logs)}] {l}")
