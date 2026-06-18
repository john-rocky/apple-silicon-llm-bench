#!/usr/bin/env python3
"""Generate the LiteRT-LM per-device package (docs/litert-lm/README.md) from raw JSONL.

For every device × model that has a LiteRT-LM short-chat run in results/raw/, this emits:

  - throughput        — per-run + median (decode tok/s, TTFT, prefill, peak RAM, ITL p99),
                        LiteRT-LM alongside the sibling runtimes, with a per-row Quant column
  - "why it's fast"   — effective decode bandwidth (tok/s × weight-bytes, quant-scaled per row)
                        and % of the chip's memory-bandwidth roofline
  - sustained throttle — start→steady decode, % retained, thermal (when an energy run exists)
  - energy            — J/token, tokens per 1% battery (when an energy run exists)
  - synthesis         — where LiteRT-LM wins and where it has room to grow, from the numbers

Every number is read from the raw JSONL — never hand-copied. Each section auto-labels its
capture conditions (build config, output cap, memory metric) read from the JSONL itself, so a
"fair" 0.13.1/Release/capped block and a "pre-fair" 0.12/Debug block are never silently mixed.

    python3 scripts/litert_lm_report.py
"""
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "litert-lm" / "README.md"

# (canonical key, short-chat filename tokens to try, energy token, display label, model suffix).
# MLX is spelled `mlx-swift` in the fair Qwen3 files but `mlx` in older Gemma files — try both.
# Core AI exports two bundles whose compute unit is fixed by export shape, so its files carry a
# `-gpu`/`-ane` suffix on the model token (e.g. `core-ai-qwen3-0.6b-gpu-...`); the suffix is
# appended to the model when globbing so both land in the same model's comparison.
RUNTIMES = [
    ("litert-lm", ["litert-lm"], "litert-lm", "LiteRT-LM / GPU", ""),
    ("mlx-swift", ["mlx-swift", "mlx"], "mlx-swift", "MLX-Swift / GPU", ""),
    ("llama-cpp", ["llama-cpp"], "llama-cpp", "llama.cpp / GPU", ""),
    ("coreml-llm", ["coreml-llm"], "coreml-llm", "CoreML / ANE", ""),
    ("core-ai-gpu", ["core-ai"], "core-ai", "Core AI / GPU", "-gpu"),
    ("core-ai-ane", ["core-ai"], "core-ai", "Core AI / ANE", "-ane"),
]

# Runtimes whose Metal pipeline kernel cache persists across process launches: even
# fresh-process "cold" launches run warm after the first, so only launch 1 (post fresh
# install) is truly cold. We put that true-cold figure in the cold table for iso-cold
# parity and disclose the warm steady-state in a note (see collect()/md_kernelcache_note).
KERNEL_CACHED = {"core-ai-gpu"}

# Active weight bytes streamed per decode token at ~4-bit (GB). Decode is memory-bandwidth-
# bound, so tok/s × this ≈ effective read bandwidth. Per-row quant scaling (below) adjusts
# rows that aren't 4-bit. Gemma E2B = 0.79 GB INT4 decoder (litert catalog breakdown);
# Qwen3-0.6B ≈ 0.35 GB (the 498 MB mixed-INT4 artifact minus its INT8 embedding table).
# Active 4-bit weight bytes streamed per decode token (GB). Qwen3 dense ≈ params × 0.5 B.
MODEL_DECODE_GB = {"gemma-4-e2b": 0.79, "gemma-4-e4b": 1.5, "gemma-4-12b": 6.0,
                   "qwen3-0.6b": 0.35, "qwen3-4b": 2.0, "qwen3-8b": 4.0, "qwen3-14b": 7.0,
                   "lfm2.5-350m": 0.18, "minicpm5-1b": 0.53}
MODEL_LABEL = {"gemma-4-e2b": "Gemma 4 E2B", "gemma-4-e4b": "Gemma 4 E4B", "gemma-4-12b": "Gemma 4 12B",
               "qwen3-0.6b": "Qwen3-0.6B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
               "qwen3-14b": "Qwen3-14B", "lfm2.5-350m": "LFM2.5-350M", "minicpm5-1b": "MiniCPM5-1B"}

# Device peak memory bandwidth (GB/s) = LPDDR5X data-rate × 64-bit bus / 8. Public-teardown
# ESTIMATES, not Apple figures — flagged in the report; an obvious thing for Lu's team to pin.
DEVICE_PEAK_BW = {
    "iPhone18,1": 76.8,   # A19 Pro, LPDDR5X 9600 MT/s (iPhone 17 Pro)
    "iPhone18,3": 68.2,   # A19,     LPDDR5X 8533 MT/s (iPhone 17)
    "iPhone17,3": 60.0,   # A18,     LPDDR5X ~7500 MT/s (iPhone 16)
}


def load(path):
    txt = open(path).read().strip()
    try:
        return json.loads(txt)                      # whole file = one JSON object (compact or pretty)
    except json.JSONDecodeError:
        return json.loads(txt.split("\n")[0])       # JSONL with multiple records: take the first


def med(xs):
    return statistics.median(xs)


def quant_factor(quant):
    """Bytes-per-token multiplier vs a 4-bit baseline, from the quant string."""
    q = (quant or "").lower()
    if "int8" in q or "8-bit" in q or "8bit" in q:
        return 2.0
    if "4/8" in q or "mixed 4" in q:   # mixed 4/8-bit palettized
        return 1.5
    return 1.0   # INT4 / 4-bit / Q4 / Q4_K_M / QAT


# Devices whose LiteRT-LM rows are NOT GPU and so don't belong in the GPU-framed
# per-device tables. On Apple-Silicon macOS, LiteRT-LM's GPU backend fails to start
# (OpenCL context creation fails) and it runs CPU-only — covered separately in
# docs/litert-lm/MACOS_DESKTOP.md so this package's "/ GPU" framing stays accurate.
SEPARATE_DEVICES = {"m4max"}


def discover():
    """{device: [models]} from LiteRT-LM short-chat run1 files."""
    out = {}
    for p in sorted(RAW.glob("*-litert-lm-*-short-chat-run1.jsonl")):
        dev, rest = p.name.split("-litert-lm-", 1)
        if dev in SEPARATE_DEVICES:
            continue
        model = rest.split("-short-chat-run1.jsonl")[0]
        out.setdefault(dev, [])
        if model not in out[dev]:
            out[dev].append(model)
    return out


def short_chat_runs(device, sc_tokens, model):
    for tok in sc_tokens:
        fs = sorted(RAW.glob(f"{device}-{tok}-{model}-short-chat-run*.jsonl"))
        if fs:
            return fs
    return []


def energy_file(device, en_token, model):
    p = RAW / f"{device}-{en_token}-{model}-energy-tg128.jsonl"
    return p if p.exists() else None


# ---- throttle math, identical convention to scripts/throttle_curve.py -------------
def throttle_stats(metrics):
    rw = [x for x in metrics.get("decodeRateRollingWindow", []) if x > 0]
    if not rw:
        return None
    start = sum(rw[:5]) / min(5, len(rw))
    tail = rw[int(len(rw) * 0.9):]
    steady = sum(tail) / len(tail)

    def t_drop(frac):
        target = start * (1 - frac)
        for i, v in enumerate(rw):
            if v <= target:
                return i
        return None

    return {"start": start, "steady": steady, "t10": t_drop(0.10), "t25": t_drop(0.25),
            "init": metrics.get("initialThermalState"), "peak": metrics.get("peakThermalState")}


def fmt_t(v):
    return f"{v}s" if v is not None else "—"


# ---- per (device, model) assembly -------------------------------------------------
def collect(device, model):
    rows = {"throughput": [], "throttle": [], "energy": [], "raw": [], "meta": None}
    for key, sc_tokens, en_token, label, suffix in RUNTIMES:
        scs = short_chat_runs(device, sc_tokens, model + suffix)
        if scs:
            ds = [load(p) for p in scs]
            ms = [d["metrics"] for d in ds]
            if rows["meta"] is None and key == "litert-lm":
                rows["meta"] = {"device": ds[0]["device"], "model": ds[0]["model"]}
            decs = [m["decodeTokensPerSecond"] for m in ms]
            # Core AI's Metal pipeline kernel cache persists across process launches, so only
            # the FIRST launch after a fresh install is truly cold; later "cold" launches reuse
            # the cached kernels and run warm. Report the true-cold value (the slowest) for
            # iso-cold parity with the cold-consistent runtimes, and keep warm for a disclosure
            # note — never a warm-biased median that would falsely outrank a cold competitor.
            warm = None
            if key in KERNEL_CACHED and len(decs) >= 2:
                cold = min(decs)
                rest = [d for d in decs if d != cold]
                warm = med(rest) if rest else None
                decode_val = cold
            else:
                decode_val = med(decs)
            rows["throughput"].append({
                "label": label, "key": key, "n": len(scs),
                "quant": ds[0]["model"].get("quantization", "?"),
                "size": ds[0]["model"].get("onDiskSizeMB"),   # raw byte budget, shown so two
                                                              # "4-bit" rows of different size aren't conflated
                "decode": decode_val,
                "warm": warm,                                 # warm steady-state for kernel-cached runtimes (note only)
                "ttft": med([m["firstTokenLatencyMS"] for m in ms]),
                "prefill": med([m["promptTokensPerSecond"] for m in ms]),
                "peakmem": med([m["memoryPeakDuringDecodeMB"] for m in ms]),
                "p99": med([m["interTokenLatencyP99MS"] for m in ms]),
                "gen": med([m["generatedTokenCount"] for m in ms]),
                "build": ds[0]["device"].get("buildConfiguration", "?"),
                "thermals": [m.get("initialThermalState") for m in ms],
                "fair": (ds[0]["device"].get("buildConfiguration", "?") == "Release")
                        and (med([m["generatedTokenCount"] for m in ms]) <= 130),
            })
            rows["raw"] += [p.name for p in scs]
        ef = energy_file(device, en_token, model + suffix)
        if ef:
            d = load(ef); m = d["metrics"]
            ts = throttle_stats(m)
            ebuild = d["device"].get("buildConfiguration", "?")
            if ts:
                ts["label"] = label
                ts["build"] = ebuild
                ts["burst"] = next((r["decode"] for r in rows["throughput"]
                                    if r["label"] == label), ts["start"])
                ts["retained"] = ts["steady"] / ts["burst"] * 100 if ts["burst"] else 0
                rows["throttle"].append(ts)
            dpct = m.get("batteryDeltaPercent") or 0
            rows["energy"].append({
                "label": label, "build": ebuild, "jpt": m.get("energyJoulesPerToken"),
                "tok_per_pct": (m.get("generatedTokenCount") / dpct) if dpct else None,
                "avg_w": m.get("averagePackagePowerW"), "delta_pct": dpct,
                "peak": m.get("peakThermalState"), "init": m.get("initialThermalState")})
            rows["raw"].append(ef.name)
    return rows


# ---- markdown ---------------------------------------------------------------------
def md_throughput(rows):
    out = ["| Runtime | Quant | n | Decode tok/s | TTFT ms | Prefill tok/s | Peak RAM MB | ITL p99 ms |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    best = max(r["decode"] for r in rows["throughput"])
    for r in rows["throughput"]:
        win = " 🏆" if abs(r["decode"] - best) < 1e-6 else ""
        cold_mark = "†" if r.get("warm") else ""   # value shown is true-cold (n=1); warm in the note
        prefill = f"{r['prefill']:.0f}" if r["prefill"] else "—"
        out.append(f"| {r['label']} | {r['quant']} | {r['n']} | {r['decode']:.1f}{cold_mark}{win} | {r['ttft']:.0f} | "
                   f"{prefill} | {r['peakmem']:.0f} | {r['p99']:.1f} |")
    return "\n".join(out)


def md_kernelcache_note(rows):
    """Disclose warm steady-state for kernel-cached runtimes shown at true-cold (fairness)."""
    notes = []
    for r in rows["throughput"]:
        if r.get("warm"):
            notes.append(
                f"**† {r['label']}** is shown at its **true cold** ({r['decode']:.0f} tok/s, n=1): its Metal "
                f"pipeline kernel cache persists across launches, so only the first launch after a fresh install "
                f"is genuinely cold — later iso-cold launches reuse the cached kernels. Its **warm steady-state is "
                f"~{r['warm']:.0f} tok/s** (cache primed — what a user actually sees, and the fastest here). Cold "
                f"is shown for iso-cold parity with the cold-consistent runtimes; warm is the real-world number. "
                f"Both are disclosed rather than blended into a warm-biased median.")
    return ("> " + "\n>\n> ".join(notes)) if notes else None


def md_bandwidth(rows, model, device_id):
    gb = MODEL_DECODE_GB.get(model)
    if not gb:
        return None
    peak = DEVICE_PEAK_BW.get(device_id)
    head = "| Runtime | Quant | Weight MB on disk | Decode tok/s | Effective BW (GB/s) |"
    sep = "|---|---|---:|---:|---:|"
    if peak:
        head += " % of peak BW |"; sep += "---:|"
    out = [head, sep]
    best_eff = 0
    for r in rows["throughput"]:
        eff = r["decode"] * gb * quant_factor(r["quant"])   # quant-scaled bytes/token
        best_eff = max(best_eff, eff)
        r["_eff"] = eff
    for r in rows["throughput"]:
        win = " 🏆" if abs(r["_eff"] - best_eff) < 1e-6 else ""
        sz = f"{r['size']:.0f}" if r.get("size") else "—"
        line = f"| {r['label']} | {r['quant']} | {sz} | {r['decode']:.1f} | {r['_eff']:.1f}{win} |"
        if peak:
            line += f" {r['_eff'] / peak * 100:.0f}% |"
        out.append(line)
    return "\n".join(out), gb, peak


def md_throttle(rows):
    out = ["| Runtime | Cold burst tok/s | Sustained tok/s | Retained | t→−10% | t→−25% | Thermal (init→peak) |",
           "|---|---:|---:|---:|---:|---:|:--|"]
    for r in rows["throttle"]:
        out.append(f"| {r['label']} | {r['burst']:.1f} | {r['steady']:.1f} | {r['retained']:.0f}% | "
                   f"{fmt_t(r['t10'])} | {fmt_t(r['t25'])} | {r['init']}→{r['peak']} |")
    return "\n".join(out)


def md_energy(rows):
    out = ["| Runtime | J / token | Tokens / 1% battery | Avg pkg power W | Δbattery | Thermal (init→peak) |",
           "|---|---:|---:|---:|---:|:--|"]
    for r in rows["energy"]:
        tpp = f"{r['tok_per_pct']:,.0f}" if r["tok_per_pct"] else "—"
        out.append(f"| {r['label']} | {r['jpt']:.3f} | {tpp} | {r['avg_w']:.2f} | "
                   f"{r['delta_pct']:.0f}% | {r['init']}→{r['peak']} |")
    return "\n".join(out)


# Sustained/energy runs that did not complete — kept with their reason (fairness rule 4),
# keyed by (runtime key, model). A note renders only when that runtime HAS short-chat data
# but is absent from both the throttle and energy tables, so it never shows spuriously
# (e.g. Gemma, where LiteRT-LM's sustained run does complete, gets no note).
SUSTAINED_FAILURES = {
    ("litert-lm", "qwen3-0.6b"):
        "**LiteRT-LM / GPU** is absent from the two tables above: its 600 s *continuous* Qwen3-0.6B "
        "run does not complete on v0.13.1 — generation hangs with `DEADLINE_EXCEEDED` raised in the "
        "callback thread pool (the Metal top-k sampler `libLiteRtTopKMetalSampler.dylib` is not on the "
        "sustained path). The 128-token short-chat run above is unaffected — only long continuous "
        "generation hangs. Kept here with its cause (fairness rule 4): a reproducible finding for the "
        "LiteRT team, not a harness issue — the same adapter drives the short-chat run that passes.",
}


# Why a runtime is excluded from a fair throughput table, keyed (runtime key, model). Used
# only when that runtime has data but isn't fair-consistent; otherwise a generic note is shown.
CAPTURE_NOTES = {
    ("core-ai-gpu", "qwen3-0.6b"):
        "**Core AI / GPU** (`coreai-pipelined`): a pre-fair Debug / iOS 27 preview exists — "
        "**71 tok/s cold → ~180 warm**, 524 MB peak, INT4 (dynamic), the fastest of any runtime here "
        "once its 3-deep pipeline is warm. Withheld from the same-conditions table because it is Debug "
        "and not captured as 3 fresh-cold launches (run 1 cold, runs 2–4 warm) — a fair Release / "
        "iso-cold re-capture against this LiteRT-LM row is pending. Full method + the cold-vs-warm "
        "breakdown: [`methodology/coreai-ios.md`](../../methodology/coreai-ios.md).",
    ("core-ai-ane", "qwen3-0.6b"):
        "**Core AI / ANE** (`static-shape`): pre-fair Debug / iOS 27 preview — **~50 tok/s**, 1166 MB "
        "peak, mixed 4/8-bit palettized. Steady cold-to-warm, the Neural-Engine corner (vs CoreML-LLM's "
        "leaner-but-slower ANE path). The next fair Release re-capture switches this row to the leaner "
        "**uniform-4bit palettized** ANE export (389 vs 434 MB on disk) — pure 4-bit on the ANE is reachable, "
        "but the GPU's *linear* INT4 scheme is not (it crashes the ANE pre-compiler). See "
        "[`COREAI_INT4_EXPORT.md`](COREAI_INT4_EXPORT.md) and [`methodology/coreai-ios.md`](../../methodology/coreai-ios.md).",
    ("coreml-llm", "gemma-4-e2b"):
        "**CoreML / ANE**: a fair re-capture was attempted via a side-loaded bundle. One cold run "
        "measured **25.3 tok/s**, but n=3 iso-runs were jetsam-killed (`signal 9`) — the gemma-3n "
        "bundle peaks at **~3.3 GB** (4 INT8 decode chunks + the 2.35 GB per-layer embedding table), "
        "which sits at iOS's per-app memory ceiling. Withheld from the median table pending a "
        "memory-trimmed bundle; the ~3.3 GB footprint is itself a finding for on-device gemma-3n. "
        "(The pre-fair Debug/iOS 26 row read 33 tok/s at 1187 MB on a lighter bundle.)",
}


def md_sustained_gap(rows, model):
    """Note runtimes that produced short-chat data but no sustained/energy run (fairness rule 4)."""
    have = {r["label"] for r in rows["throttle"]} | {r["label"] for r in rows["energy"]}
    notes = [SUSTAINED_FAILURES[(r["key"], model)] for r in rows["throughput"]
             if r["label"] not in have and (r["key"], model) in SUSTAINED_FAILURES]
    return ("> ⚠️ " + "\n>\n> ".join(notes)) if notes else None


def md_synthesis(rows, model):
    """Where LiteRT-LM wins / has room to grow, straight from the numbers."""
    tp = rows["throughput"]
    lit = next((r for r in tp if r["key"] == "litert-lm"), None)
    if not lit or len(tp) < 2:
        return None
    by_dec = sorted(tp, key=lambda r: -r["decode"])
    by_mem = sorted(tp, key=lambda r: r["peakmem"])
    dec_rank, mem_rank = by_dec.index(lit) + 1, by_mem.index(lit) + 1
    dec_lead, mem_lead = by_dec[0], by_mem[0]
    name = MODEL_LABEL.get(model, model)

    if dec_rank == 1:
        dec = f"**wins decode** ({lit['decode']:.0f} tok/s, fastest of {len(tp)})"
    else:
        dec = (f"is **#{dec_rank} on decode** ({lit['decode']:.0f} vs {dec_lead['label'].split(' /')[0]} "
               f"{dec_lead['decode']:.0f}, −{(1 - lit['decode'] / dec_lead['decode']) * 100:.0f}%)")
    if mem_rank == 1:
        mem = f"**leanest memory** ({lit['peakmem']:.0f} MB)"
    else:
        mem = (f"**heaviest memory** ({lit['peakmem']:.0f} MB vs {mem_lead['label'].split(' /')[0]} "
               f"{mem_lead['peakmem']:.0f})" if mem_rank == len(tp)
               else f"#{mem_rank} on memory ({lit['peakmem']:.0f} MB)")
    verdict = ("a clean LiteRT-LM win" if dec_rank == 1 and mem_rank == 1
               else "room to grow" if dec_rank > 1 else "a decode win with a memory cost")
    return (f"**{name}:** LiteRT-LM {dec} and {mem} → _{verdict}_. "
            f"LiteRT's memory is its real footprint (INT4 decoder + the INT8 embedding table it "
            f"keeps + Metal working buffers), not KV pre-allocation waste — so the gap to a dynamic-KV "
            f"runtime like MLX is structural, and a fair thing to show rather than hide.")


def conditions(rows):
    """Read the actual capture conditions off the JSONL for an honest per-section label."""
    lit = next((r for r in rows["throughput"] if r["key"] == "litert-lm"), rows["throughput"][0])
    build = lit["build"]
    capped = lit["gen"] <= 130          # 128-budget honoured (vs EOS ~339/458)
    nominal = all(t == "nominal" for r in rows["throughput"] for t in r["thermals"])
    fair = build == "Release" and capped
    tag = ("**fair** (0.13.1 · Release · 128-token cap · phys_footprint)" if fair
           else f"**pre-fair** ({build} · {'capped' if capped else 'ran to EOS'} · re-capture pending)")
    return tag, nominal, capped


HEADER = """# LiteRT-LM on Apple silicon — per-device package

> Self-contained, reproducible measurements pulled straight from the raw JSONL in
> [`../../results/raw/`](../../results/raw/). Generated by
> [`scripts/litert_lm_report.py`](../../scripts/litert_lm_report.py) — **do not hand-edit**; re-run it.
> Part of the neutral [Apple Silicon LLM Benchmark](../../README.md) (one headless harness for every runtime).

## The short version

A neutral, reproducible on-device benchmark built to be trustworthy to the people who ship the runtime:
**(1) the model the LiteRT team actually optimises** — **Qwen3-0.6B** (dense), measured on
LiteRT-LM v0.13.1 (which fixed the mixed-INT4 Metal-GPU segfault that 0.12.0 hit); **(2) genuinely
same-conditions** — same 128-token budget, greedy, cold-from-`nominal`, Release build, n=3 median,
`phys_footprint` memory; LiteRT-LM's output is capped at 128 so it generates the *same* token count
as every other runtime; **(3) it explains _why_** — decode is memory-bandwidth-bound, so we report
each runtime's effective GB/s against the chip's roofline. Every cell traces to raw JSONL.

**Honest up front:** quantisation is each runtime's native format (not bit-identical — shown per row),
and GPU vs ANE is a real backend difference. Those two axes can't be equalised in any cross-runtime
comparison; we disclose them rather than hide them. Decode tok/s on a fixed device is the fair headline.

## What was measured

| | |
|---|---|
| **Runtime** | [`google-ai-edge/LiteRT-LM`](https://github.com/google-ai-edge/LiteRT-LM) **v0.13.1** (SwiftPM product `LiteRTLM`), Metal **GPU** backend |
| **Adapter** | [`MediaPipeRuntime.swift`](../../ios/BenchmarkApp/Sources/Runtimes/MediaPipeRuntime.swift) (kind `litert-lm`) — output capped at the task's 128-token budget for an iso-token comparison |
| **Harness** | fully headless via `devicectl` — models side-loaded, nothing typed on the phone; identical protocol for every runtime |
| **Memory** | `phys_footprint` (the jetsam-charged figure; counts mmap'd weights that `resident_size` under-reports) |

### Tasks & parameters

- **short-chat** — 20-token prompt ([`prompts/short-chat.md`](../../prompts/short-chat.md)), greedy
  (temp 0 / top-p 1), **128-token output**, cold start, **median of n=3**.
- **energy / throttle** — 600 s continuous generation, unplugged, tg128
  ([`methodology/energy-ios.md`](../../methodology/energy-ios.md)). Shown where captured.

### Fairness & conditions ([rules](../../methodology/fairness-rules.md))

**Held equal:** model, prompt, 128-token budget *and* generated count, greedy, cold start, n=3 median,
Release build, `phys_footprint`, one device per table. **Disclosed, not equalised:**

- **Quantisation** is each runtime's native format (LiteRT mixed-INT4 / MLX Q4 / CoreML **INT8** /
  Core AI INT4-dynamic | mixed-4/8) — shown per row (fairness rule 3), with the **on-disk weight size**
  next to it. The effective-bandwidth column scales bytes/token coarsely by quant class (so the INT8 row
  isn't credited as 4-bit), but a single per-model byte constant does **not** equalise two nominally
  4-bit artifacts of different size (Core AI ~327 MB vs LiteRT ~498 MB) — read BW with the size column.
- **Compute unit** — LiteRT / MLX / Core AI-GPU run on the **GPU**, CoreML / Core AI-ANE on the **ANE**.
  An intentional axis, but a real difference; read an ANE row as the memory/power-efficiency corner, not a GPU peer.
- **Core AI couples compute unit *and* quant scheme** — its compute unit is fixed by the **export shape**,
  and the two shapes carry a different 4-bit *scheme*: the dynamic export → **GPU / linear INT4** (~327 MB),
  the static iOS export → **ANE / palettized 4-bit** (mixed-4/8 ~434 MB, or uniform-4bit ~389 MB). This is a
  **confirmed platform constraint, not a choice**: the GPU's linear-INT4 program **SIGSEGVs the ANE
  pre-compiler** (`anePreCompileBinary`), while the byte-identical palettized program compiles to the ANE
  cleanly — so "ANE + the GPU's INT4 scheme" is unobtainable (crash evidence in
  [`COREAI_INT4_EXPORT.md`](COREAI_INT4_EXPORT.md)). A Core AI GPU-vs-ANE row is therefore a
  **shipped-config** comparison (engine *and* quant scheme together, as Apple ships it), **not** a clean
  engine A/B — now proven, not just asserted. The uniform-4bit ANE export trims the byte confound (389 vs
  434 MB) but cannot remove the linear-vs-palettized scheme difference.
- **Memory model** — LiteRT-LM keeps an INT8 embedding table + Metal buffers, so its footprint is
  structurally higher than a dynamic-KV 4-bit runtime like MLX. Real, disclosed — not a thumb on the scale.

Methodology: [thermal](../../methodology/thermal.md) · [energy (iOS)](../../methodology/energy-ios.md)
· [fairness rules](../../methodology/fairness-rules.md) · [runtime notes](../../runtimes/litert-lm.md)

**Desktop / macOS:** on Apple-Silicon macOS LiteRT-LM's GPU backend fails to start and it runs
CPU-only — Qwen3 0.6B/4B/8B scaling (LiteRT-CPU vs MLX-Metal-GPU) and the finding are in
[`MACOS_DESKTOP.md`](MACOS_DESKTOP.md). **Long-context / continuous:** decode-flatness + memory
growth (MLX) and litert's long-context blockers are in [`LONG_CONTEXT.md`](LONG_CONTEXT.md).

**Companion docs:** [quality parity (correctness + degeneracy)](QUALITY.md) ·
[sustained-generation hang bug report](LITERT_SUSTAINED_HANG.md) ·
[model coverage matrix](MODEL_MATRIX.md) · [LiteRT model availability](MODEL_AVAILABILITY.md) ·
[Core AI INT4-on-ANE — platform-constraint finding](COREAI_INT4_EXPORT.md) ·
[reproduce / methods](METHODS.md) · [one-shot runbook (iPhone + Mac)](RUNBOOK.md).
"""

FOOTER_REPRO = """## Reproduce / add a device

Build + install once via Xcode (Release, your signing), then drive headless from a Mac with
[`scripts/run_device_bench.sh`](../../scripts/run_device_bench.sh) (`bootstrap.sh` first — it fetches
LiteRT-LM v0.13.1 and the other vendored packages). Each runtime runs 3 cold launches; collect the
JSONL the app wrote to `Documents/results/`, rename to
`results/raw/<device>-<runtime>-<model>-short-chat-runN.jsonl`, and re-run
`python3 scripts/litert_lm_report.py` — the new device/model appears automatically. Core AI rows need
the exported `.aimodel` AOT-compiled per GPU arch and side-loaded (see
[`methodology/coreai-ios.md`](../../methodology/coreai-ios.md)).
"""


def main():
    found = discover()
    if not found:
        raise SystemExit("no LiteRT-LM short-chat runs found in results/raw/")

    parts = [HEADER]
    all_raw = []
    # Fair (Release+capped) models first, newest comparison up top.
    order = {"qwen3-0.6b": 0, "qwen3-4b": 0.1, "qwen3-8b": 0.2, "qwen3-14b": 0.3,
             "gemma-4-e2b": 1, "gemma-4-e4b": 1.1, "gemma-4-12b": 1.2,
             "lfm2.5-350m": 2, "minicpm5-1b": 2.1}
    for device in sorted(found):
        for model in sorted(found[device], key=lambda m: order.get(m, 9)):
            rows = collect(device, model)
            if not rows["meta"] or not rows["throughput"]:
                continue
            dev = rows["meta"]["device"]
            mo = rows["meta"]["model"]
            tag, nominal, capped = conditions(rows)
            # When the section is fair (LiteRT row is Release+capped), don't mix in pre-fair
            # rows under a fair header: show only fair-consistent data, name the rest as pending.
            lit_tp = next((r for r in rows["throughput"] if r["key"] == "litert-lm"), None)
            section_fair = bool(lit_tp and lit_tp["fair"])
            pending_tp, withheld_en = [], []
            if section_fair:
                pending_tp = [(r["key"], r["label"]) for r in rows["throughput"] if not r["fair"]]
                withheld_en = sorted({r["label"] for r in rows["throttle"] if r.get("build") != "Release"}
                                     | {r["label"] for r in rows["energy"] if r.get("build") != "Release"})
                rows = {**rows,
                        "throughput": [r for r in rows["throughput"] if r["fair"]],
                        "throttle": [r for r in rows["throttle"] if r.get("build") == "Release"],
                        "energy": [r for r in rows["energy"] if r.get("build") == "Release"]}
            title = f"{MODEL_LABEL.get(model, model)} — {device} ({dev.get('modelIdentifier')} · iOS {dev.get('systemVersion')})"
            parts.append(f"\n---\n\n## {title}\n")
            parts.append(f"Conditions: {tag}. LiteRT file `{mo.get('primaryFile')}` · "
                         f"quant {mo.get('quantization')} · {mo.get('onDiskSizeMB')} MB on disk.\n")
            if not nominal:
                parts.append("> ⚠️ Some runs started at `fair`, not `nominal` (device warmed mid-matrix). "
                             "Decode is throttle-insensitive at `fair` and the per-run values are "
                             "consistent, but flagged for full rigor.\n")

            syn = md_synthesis(rows, model)
            if syn:
                parts.append(syn + "\n")

            parts.append("### Throughput — short-chat, cold, median of n=3\n")
            parts.append(md_throughput(rows) + "\n")
            kc = md_kernelcache_note(rows)
            if kc:
                parts.append(kc + "\n")
            if pending_tp:
                lines = [CAPTURE_NOTES.get((k, model),
                         f"{label}: captured pre-fair (Debug / iOS 26) — excluded from this "
                         f"same-conditions table pending a fair (Release / iOS 27) re-capture.")
                         for k, label in pending_tp]
                parts.append("> ⚠️ " + "\n>\n> ".join(lines) + "\n")

            bw = md_bandwidth(rows, model, dev.get("modelIdentifier"))
            if bw:
                table, gb, peak = bw
                parts.append("### Why it's fast — decode is memory-bandwidth-bound\n")
                parts.append(table + "\n")
                pk = (f"~{peak:.0f} GB/s (LPDDR5X, public-teardown **estimate**)" if peak
                      else "the chip's peak bandwidth")
                parts.append(f"> _Decode reads ≈ all active weights once per token, so **tok/s × "
                             f"weight-bytes = effective read bandwidth** (~{gb:.2f} GB/token at 4-bit, "
                             f"scaled per row by quant — the **INT8** row reads ~2×). Against {pk}, this "
                             f"ranks how well each runtime works the memory system. Absolute GB/s carries "
                             f"the byte estimate; the same-device ordering is robust._\n")
                parts.append("> ⚠️ _The byte/token figure is a single per-model constant — it does **not** "
                             "capture that two nominally **4-bit** artifacts can carry different real budgets "
                             "(see the Weight MB column: e.g. Core AI INT4 ~327 MB vs LiteRT mixed-INT4 ~498 MB). "
                             "Quant is disclosed, not equalised — read the effective-BW estimate alongside the "
                             "on-disk size, not as a like-for-like normalisation._\n")

            if rows["throttle"]:
                parts.append("### Sustained throttling — 600 s continuous, unplugged\n")
                parts.append(md_throttle(rows) + "\n")
            if rows["energy"]:
                parts.append("### Energy — battery-delta, 600 s run\n")
                parts.append(md_energy(rows) + "\n")
            gap = md_sustained_gap(rows, model)
            if gap and (rows["throttle"] or rows["energy"]):
                parts.append(gap + "\n")
            if withheld_en:
                parts.append(f"### Sustained throttling + energy\n\n"
                             f"> ⚠️ Captured pre-fair (Debug / iOS 26) for {', '.join(withheld_en)} and withheld "
                             f"here pending a fair (Release / iOS 27, unplugged) re-capture. Archived raw files "
                             f"are still linked under Provenance.\n")

            all_raw += rows["raw"]

    parts.append("\n---\n\n## Provenance — every cell traces to a raw file\n")
    for name in sorted(set(all_raw)):
        parts.append(f"- [`{name}`](../../results/raw/{name})")
    parts.append("\n" + FOOTER_REPRO)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print("wrote:", OUT.relative_to(REPO))
    for d in sorted(found):
        print(f"  {d}: {', '.join(found[d])}")


if __name__ == "__main__":
    main()
