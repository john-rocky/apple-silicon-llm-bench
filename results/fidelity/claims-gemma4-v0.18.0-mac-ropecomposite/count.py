#!/usr/bin/env python3
"""Recount the raw results of the rope-composite run (stdlib only; reads only).

Reads raw/<cases>__<model>-<file>__<backend>-<precision>.jsonl (PowerBeef's #3913 runner, one line per case),
logs/*.log (engine logs), the own-suite records ../claims-gemma-4-*-ropecomposite*-v0.18.0-mac-*.jsonl and, for the
prompt-token comparison, the 10-15 records of the public file in ../claims-gemma4-v0.18.0-mac-rope/raw/ (the same
runner, cases and LiteRT-LM build). Exactness is recomputed from `output` and `expected` (ASCII digits only) and
checked against the runner's `exact`; `expected` is checked against the runner's LCG (referenceDigits).

    python3 -I count.py [raw dir] [research/issue-3913 dir] > count.txt
    (defaults: raw/ next to this script; PowerBeef's folder from the second argument or G3913_UPSTREAM)

Adapted from ../claims-gemma4-v0.18.0-mac-rope/count.py (file labels are discovered instead of fixed).
"""
import glob, json, os, re, statistics, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "raw")
UP = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("G3913_UPSTREAM", "")
PREV_RAW = os.path.join(os.path.dirname(HERE), "claims-gemma4-v0.18.0-mac-rope", "raw")
SUITE_GLOB = os.path.join(os.path.dirname(HERE), "claims-gemma-4-*-v0.18.0-mac-*.jsonl")
SUITE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "evaldata", "fidelity_suite_v1.jsonl")
LOGS = os.path.join(HERE, "logs")
CASES_ORDER = ["short", "baseline", "long", "cross4096", "long3", "xl"]
FILE_ORDER = ["ropecomposite", "ltdefault", "original", "rope2048", "rope512"]
CONFIGS = ["gpu-fp16", "gpu-fp32", "cpu-fp32"]
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


def fkey(f):
    return (FILE_ORDER.index(f) if f in FILE_ORDER else len(FILE_ORDER), f)


# ------------------------------------------------------------------ load + integrity
arms = {}  # (cases, model, file, config) -> {id: row}
problems = []
for path in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
    base = os.path.basename(path)[:-6]
    try:
        cases, label, config = base.split("__")
        model, file = label.split("-", 1)
    except ValueError:
        problems.append(f"unparsed file name {base}")
        continue
    rows = load(path)
    if UP:
        want = len(json.load(open(os.path.join(UP, "cases", f"{cases}.json"))))
        if len(rows) != want:
            problems.append(f"{base}: {len(rows)} lines, cases file has {want}")
    for r in rows:
        if r["expected"] != ref_digits(r["seed"]):
            problems.append(f"{base} {r['id']}: expected != referenceDigits(seed)")
        if exact(r) != r["exact"]:
            problems.append(f"{base} {r['id']}: recomputed exact {exact(r)} != runner {r['exact']}")
        if not r["exact"] and r.get("firstMismatchPosition") != r["prefill"] - 1 + r["firstMismatch"]:
            problems.append(f"{base} {r['id']}: firstMismatchPosition inconsistent")
        if r["backend"] + "-" + r["activation"] != config:
            problems.append(f"{base} {r['id']}: record backend/activation {r['backend']}/{r['activation']} != {config}")
        if r["model"] != f"gemma-4-E2B-it-{file}.litertlm":
            problems.append(f"{base} {r['id']}: record model {r['model']} != gemma-4-E2B-it-{file}.litertlm")
    arms[(cases, model, file, config)] = {r["id"]: r for r in rows}

models = sorted({k[1] for k in arms})
files = sorted({k[2] for k in arms}, key=fkey)
print("# count.py — Gemma 4 E2B rope-composite export vs default export, LiteRT-LM v0.18.0 (M4 Max), recount of",
      os.path.relpath(RAW, HERE) + "/" + ("" if UP else " (PowerBeef's folder not given: line counts against his cases/ skipped)"))
print(f"arms {len(arms)}, replies {sum(len(v) for v in arms.values())}, files {files}")
print("integrity:", ("OK (" + ("line counts, " if UP else "") + "expected = LCG(seed), recomputed exact = runner exact, firstMismatchPosition, record model/backend/activation = file name)") if not problems else "")
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
prev = {}
for path in glob.glob(os.path.join(PREV_RAW, "*__E2B-original__*.jsonl")):
    cases = os.path.basename(path).split("__")[0]
    for r in load(path):
        prev[(cases, r["id"])] = r["prefill"]
if prev:
    same = [k for k in pre if k in prev and pre[k] == {prev[k]}]
    diff = [k for k in pre if k in prev and pre[k] != {prev[k]}]
    print(f"prefill vs the 10-15 run of the public gemma-4-E2B-it.litertlm (same runner, cases, build; same case id): {len(same)} equal, {len(diff)} differ"
          + ("" if not diff else f" {[(k, sorted(pre[k]), prev[k]) for k in diff[:8]]}"))
pb = {}
for path in glob.glob(os.path.join(UP, "results", "mac", "*__original__*.jsonl")) if UP else []:
    cases = os.path.basename(path).split("__")[0]
    for r in load(path):
        pb[(cases, r["id"])] = r["prefill"]
if pb:
    same = [k for k in pre if k in pb and pre[k] == {pb[k]}]
    diff = [k for k in pre if k in pb and pre[k] != {pb[k]}]
    print(f"prefill vs PowerBeef's M6 original records (same case id): {len(same)} equal, {len(diff)} differ"
          + ("" if not diff else f" {[(k, sorted(pre[k]), pb[k]) for k in diff[:6]]}"))

# ------------------------------------------------------------------ table 1
present = [c for c in CASES_ORDER if any(k[0] == c for k in arms)]
print("\n## 1. exact / total (M4 Max, LiteRT-LM v0.18.0, PowerBeef's runner and cases)\n")
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
for (cases, model, file, config), rows in sorted(arms.items(), key=lambda kv: (CASES_ORDER.index(kv[0][0]) if kv[0][0] in CASES_ORDER else 99, fkey(kv[0][2]), kv[0][3])):
    dec = sorted(r["decode"] for r in rows.values())
    print(f"- {cases} {model}-{file} {config}: decode tokens {dec[0]}–{dec[-1]}, replies at the 80-token cap {sum(d >= 80 for d in dec)}")

# ------------------------------------------------------------------ table 2 (baseline groups, if run)
GROUPS = [("1704–1751", 1751), ("1904–1951 (incl. late)", 1951), ("1952–2044", 2044), ("2011–2638", 10 ** 9)]


def group_of(r):
    end = r["prefill"] + 46
    for name, hi in GROUPS:
        if end <= hi:
            return name


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[group_of(r)].append(r)
    return g


if any(k[0] == "baseline" for k in arms):
    print("\n## 2. `baseline` by reply digit positions (prefill − 1 … prefill + 46), exact / total\n")
    print("| model | file | backend-precision | " + " | ".join(n for n, _ in GROUPS) + " |")
    print("|---|---|---|---|---|---|---|")
    for model in models:
        for file in files:
            for config in CONFIGS:
                rows = arms.get(("baseline", model, file, config))
                if rows:
                    g = groups(rows.values())
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
            ident(arms.get((cases, model, file, "gpu-fp16"), {}), arms.get((cases, model, file, "gpu-fp32"), {}),
                  f"{model} {file} gpu fp16 vs gpu fp32 (same file), `{cases}`")
            ident(arms.get((cases, model, file, "cpu-fp32"), {}), arms.get((cases, model, file, "gpu-fp32"), {}),
                  f"{model} {file} cpu fp32 vs gpu fp32 (same file), `{cases}`")
            ident(arms.get((cases, model, file, "gpu-fp16"), {}), arms.get((cases, model, file, "cpu-fp32"), {}),
                  f"{model} {file} gpu fp16 vs cpu fp32 (same file), `{cases}`")
    for cases in present:
        for config in CONFIGS:
            for fa in files:
                for fb in files:
                    if fkey(fa) < fkey(fb):
                        ident(arms.get((cases, model, fa, config), {}), arms.get((cases, model, fb, config), {}),
                              f"{model} {fa} vs {fb}, {config}, `{cases}`")
# against the 10-15 public-file records (a different export: text may legitimately differ; shown for information)
for path in sorted(glob.glob(os.path.join(PREV_RAW, "*__E2B-original__*.jsonl"))):
    cases, _, config = os.path.basename(path)[:-6].split("__")
    b = {r["id"]: r for r in load(path)}
    for file in files:
        ident(arms.get((cases, "E2B", file, config), {}), b, f"E2B {file} vs the public file (10-15 run), {config}, `{cases}`")

# ------------------------------------------------------------------ 4. failures by position
print("\n## 4. failures by position\n")
for model in models:
    for file in files:
        for config in CONFIGS:
            allf = []
            for cases in present:
                rows = arms.get((cases, model, file, config))
                if not rows:
                    continue
                fails = [r for r in rows.values() if not exact(r)]
                allf += fails
                if fails:
                    pos = sorted(r["firstMismatchPosition"] for r in fails)
                    print(f"- {model} {file} {config} `{cases}`: {len(fails)} fail, firstMismatchPosition {pos[0]}–{pos[-1]};"
                          f" prefill {min(r['prefill'] for r in fails)}–{max(r['prefill'] for r in fails)}; ids {[r['id'] for r in fails][:8]}")
                else:
                    print(f"- {model} {file} {config} `{cases}`: 0 fail")
            oks = [r for cases in present for r in (arms.get((cases, model, file, config)) or {}).values() if exact(r)]
            if allf:
                print(f"  = {model} {file} {config}: min firstMismatchPosition {min(r['firstMismatchPosition'] for r in allf)}"
                      f" ({min(allf, key=lambda r: r['firstMismatchPosition'])['id']}),"
                      f" highest last-digit position among exact replies {max((r['prefill'] + 46 for r in oks), default='-')}")
            elif oks:
                print(f"  = {model} {file} {config}: 0 fail; highest last-digit position among exact replies {max(r['prefill'] + 46 for r in oks)}")

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
                        print(f"- `{cases}` {model} {file} {config}: prefill {statistics.median(r['prefillTPS'] for r in v):.0f},"
                              f" decode {statistics.median(r['decodeTPS'] for r in v):.1f}, seconds/reply {statistics.median(r['seconds'] for r in v):.2f} (n={len(v)})")

# ------------------------------------------------------------------ 6. own suite
suite_files = sorted(p for p in glob.glob(SUITE_GLOB) if "ropecomposite" in os.path.basename(p) or "ltdefault" in os.path.basename(p))
if suite_files and os.path.exists(SUITE):
    suite = {json.loads(l)["id"]: json.loads(l) for l in open(SUITE)}
    print("\n## 6. own fidelity suite (verbatim-substring pass test, recounted here)\n")
    print("| file | model | backend | version | reps | pass / gens | failing cases / cases | failing by bucket | max distinct outputs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in suite_files:
        rows = load(path)
        if not rows:
            continue
        ok = lambda r: not r.get("error") and all(v in r["output"] for v in suite[r["case_id"]]["expected"].values())
        ids = sorted({r["case_id"] for r in rows})
        fails = sorted({r["case_id"] for r in rows if not ok(r)})
        nb = Counter(suite[i]["context_bucket"] for i in ids)
        fb = Counter(suite[i]["context_bucket"] for i in fails)
        distinct = max(len({r["output"] for r in rows if r["case_id"] == i}) for i in ids)
        print(f"| {os.path.basename(path)} | {sorted({r['model'] for r in rows})} | {sorted({r['backend'] for r in rows})}"
              f" | {sorted({r.get('runtime_version') for r in rows})} | {min(r['rep'] for r in rows)}–{max(r['rep'] for r in rows)}"
              f" | {sum(ok(r) for r in rows)}/{len(rows)} | {len(fails)}/{len(ids)}"
              f" | {', '.join(f'{b}: {fb.get(b, 0)}/{nb[b]}' for b in sorted(nb))} | {distinct} |")
        for c in fails:
            got = next(r["output"] for r in rows if r["case_id"] == c and not ok(r))
            pt = sorted({r.get("prefill_tokens") for r in rows if r["case_id"] == c})
            print(f"  - FAIL `{c}` bucket {suite[c]['context_bucket']} prefill {pt}: expected {suite[c]['expected']} got {got[:100]!r}")

# ------------------------------------------------------------------ 7. engine identity
print("\n## 7. engine log lines (verbatim, deduplicated; count = logs containing the line)\n")
pat = re.compile(r"Selected adapter|delegate_webgpu|registered|[Vv]ersion|RegisterAccelerator|backend=|WebGPU|Metal|OpenCL|xnnpack|XNNPACK|engine loaded|WATCHDOG|magic_number|activation|Activation|rope|composite|[Ee]rror")
seen = Counter()
for path in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
    lines = set()
    for l in open(path, errors="replace"):
        if pat.search(l):
            l = re.sub(r"^[IWE]\d{4} \S+ \d+ ", "", l.strip())          # absl prefix: level+date, time, thread
            l = re.sub(r"ptr=0x[0-9a-f]+", "ptr=…", re.sub(r"\d+\.\d+ s$", "<t> s", l))
            lines.add(l[:220])
    for l in lines:
        seen[l] += 1
nlogs = len(glob.glob(os.path.join(LOGS, "*.log")))
print(f"logs: {nlogs}")
for l, n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"- [{n}/{nlogs}] {l}")
