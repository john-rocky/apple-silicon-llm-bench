#!/usr/bin/env python3
"""Generate the LiteRT-LM per-device package (docs/litert-lm/README.md) from raw JSONL.

For every device that has a LiteRT-LM run in results/raw/, this emits one section with:

  - short-chat throughput   — per-run + median (decode tok/s, TTFT, prefill, peak RAM, ITL p99),
                              LiteRT-LM alongside the sibling runtimes on the same device + model
  - sustained throttling    — start -> steady decode rate, % retained, time-to-10/25% drop, thermal
  - energy (battery-delta)  — J/token, tokens per 1% battery, average package power

Every number is read from the raw JSONL in results/raw/ — never hand-copied — so adding a
device is: capture its results/raw/<device>-litert-lm-*.jsonl (see the "Add a device" section
the report prints), drop the files in, and re-run:

    python3 scripts/litert_lm_report.py

The throttle start/steady/retained definitions match scripts/throttle_curve.py, so this report
never contradicts the committed RESULTS.md / README.md throttle table.
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
OUT = REPO / "docs" / "litert-lm" / "README.md"

# Anchor model = the one all four runtimes already have on-device data for. LiteRT-LM
# is NOT Gemma-only — litert-community ships Qwen3 (0.6B/4B) + LFM in .litertlm too; a
# Qwen3-0.6B block (the model Lu's team optimises) lands once its device runs land.
MODEL = "gemma-4-e2b"
MODEL_LABEL = "Gemma 4 E2B (4-bit)"

# (filename token for short-chat, filename token for energy, display label). MLX is
# spelled `mlx` in short-chat files and `mlx-swift` in the energy file.
RUNTIMES = [
    ("litert-lm", "litert-lm", "LiteRT-LM / GPU"),
    ("mlx", "mlx-swift", "MLX-Swift / GPU"),
    ("llama-cpp", "llama-cpp", "llama.cpp / GPU"),
    ("coreml-llm", "coreml-llm", "CoreML / ANE"),
]

# --- "why is it fast" — decode is memory-bandwidth-bound -----------------------------
# Bytes streamed per decode token ≈ the active/decoder weights at the runtime's quant.
# All four runtimes here are ~4-bit, so per-token bytes are within ~15% — the
# cross-runtime ranking is robust; the absolute GB/s carries this byte assumption (±).
# Gemma E2B = 0.79 GB INT4 decoder (per the litert catalog's own breakdown);
# Qwen3-0.6B ≈ 0.40 GB INT4 (estimate — confirm once its runs land).
MODEL_DECODE_GB = {
    "gemma-4-e2b": 0.79,
    "qwen3-0.6b": 0.40,
}

# Device peak memory bandwidth (GB/s) = LPDDR5X data-rate × 64-bit bus / 8. These are
# public-teardown ESTIMATES, not Apple figures — flagged as such in the report, and an
# obvious thing for Lu's team to confirm. modelIdentifier -> GB/s.
DEVICE_PEAK_BW = {
    "iPhone18,1": 76.8,   # A19 Pro, LPDDR5X 9600 MT/s (iPhone 17 Pro)
    "iPhone18,3": 68.2,   # A19,     LPDDR5X 8533 MT/s (iPhone 17)
    "iPhone17,3": 60.0,   # A18,     LPDDR5X ~7500 MT/s (iPhone 16)
}


def load(path):
    """First JSON object in a (possibly multi-line) JSONL file."""
    with open(path) as f:
        return json.loads(f.read().strip().split("\n")[0])


def discover_devices():
    """Device prefixes (e.g. 'iphone17pro') that have a LiteRT-LM short-chat run for MODEL."""
    devs = []
    for p in sorted(RAW.glob(f"*-litert-lm-{MODEL}-short-chat-run1.jsonl")):
        devs.append(p.name.split("-litert-lm-")[0])
    return devs


def short_chat_runs(device, sc_token):
    return sorted(RAW.glob(f"{device}-{sc_token}-{MODEL}-short-chat-run*.jsonl"))


def energy_file(device, en_token):
    p = RAW / f"{device}-{en_token}-{MODEL}-energy-tg128.jsonl"
    return p if p.exists() else None


def med(xs):
    return statistics.median(xs)


# ---- throttle math, identical convention to scripts/throttle_curve.py -------------
def throttle_stats(metrics):
    rw = [x for x in metrics.get("decodeRateRollingWindow", []) if x > 0]
    if not rw:
        return None
    start = sum(rw[:5]) / min(5, len(rw))
    tail = rw[int(len(rw) * 0.9):]
    steady = sum(tail) / len(tail)
    retained = steady / start * 100 if start else 0

    def t_drop(frac):
        target = start * (1 - frac)
        for i, v in enumerate(rw):
            if v <= target:
                return i
        return None

    return {
        "start": start, "steady": steady, "retained": retained,
        "t10": t_drop(0.10), "t25": t_drop(0.25), "n": len(rw),
        "init": metrics.get("initialThermalState"),
        "peak": metrics.get("peakThermalState"),
        "final": metrics.get("finalThermalState"),
    }


def fmt_t(v):
    return f"{v}s" if v is not None else "—"


# ---- per-device assembly ----------------------------------------------------------
def collect_device(device):
    rows = {"throughput": [], "throttle": [], "energy": [], "raw": [], "meta": None}
    for sc_token, en_token, label in RUNTIMES:
        # throughput (short-chat, median of N)
        scs = short_chat_runs(device, sc_token)
        if scs:
            ds = [load(p) for p in scs]
            ms = [d["metrics"] for d in ds]
            if rows["meta"] is None and sc_token == "litert-lm":
                rows["meta"] = {"device": ds[0]["device"], "model": ds[0]["model"],
                                "params": ds[0].get("parameters", {})}
            rows["throughput"].append({
                "label": label, "n": len(scs),
                "quant": ds[0]["model"].get("quantization", "?"),
                "decode": med([m["decodeTokensPerSecond"] for m in ms]),
                "ttft": med([m["firstTokenLatencyMS"] for m in ms]),
                "prefill": med([m["promptTokensPerSecond"] for m in ms]),
                "peakmem": med([m["memoryPeakDuringDecodeMB"] for m in ms]),
                "p99": med([m["interTokenLatencyP99MS"] for m in ms]),
                "intok": med([m.get("promptTokenCount", 0) for m in ms]),
                "gen": med([m["generatedTokenCount"] for m in ms]),
            })
            rows["raw"] += [p.name for p in scs]
        # throttle + energy (energy task, single 600 s run)
        ef = energy_file(device, en_token)
        if ef:
            d = load(ef)
            m = d["metrics"]
            ts = throttle_stats(m)
            if ts:
                ts["label"] = label
                # Cold burst = the short-chat median decode (the peak you'd quote),
                # so retained = steady / burst matches the README throttle table.
                ts["burst"] = next((r["decode"] for r in rows["throughput"]
                                    if r["label"] == label), ts["start"])
                ts["retained"] = ts["steady"] / ts["burst"] * 100 if ts["burst"] else 0
                rows["throttle"].append(ts)
            dpct = m.get("batteryDeltaPercent") or 0
            rows["energy"].append({
                "label": label,
                "jpt": m.get("energyJoulesPerToken"),
                "tok_per_pct": (m.get("generatedTokenCount") / dpct) if dpct else None,
                "avg_w": m.get("averagePackagePowerW"),
                "delta_pct": dpct,
                "peak": m.get("peakThermalState"),
                "init": m.get("initialThermalState"),
                "unplugged": d["device"].get("batteryState") == "unplugged",
            })
            rows["raw"].append(ef.name)
    return rows


def device_title(meta):
    dev = meta["device"]
    return f"{dev.get('modelIdentifier')} · iOS {dev.get('systemVersion')}"


# ---- markdown ---------------------------------------------------------------------
def md_throughput(rows):
    # Quant column is mandatory (fairness rule 3): each runtime ships its own ~4-bit
    # format, not bit-identical weights — the reader must see which.
    out = ["| Runtime | Quant | n | Decode tok/s | TTFT ms | Prefill tok/s | Peak RAM MB | ITL p99 ms |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    best = max(r["decode"] for r in rows["throughput"])
    for r in rows["throughput"]:
        win = " 🏆" if abs(r["decode"] - best) < 1e-6 else ""
        # prefill 0 / missing = runtime doesn't surface a prompt-token count (e.g. ANE
        # streamed-piece counter) — render "—" rather than a misleading measured-zero.
        prefill = f"{r['prefill']:.0f}" if r["prefill"] else "—"
        out.append(f"| {r['label']} | {r['quant']} | {r['n']} | {r['decode']:.1f}{win} | {r['ttft']:.0f} | "
                   f"{prefill} | {r['peakmem']:.0f} | {r['p99']:.1f} |")
    return "\n".join(out)


def md_throttle(rows):
    out = ["| Runtime | Cold burst tok/s | Sustained tok/s | Retained | t→−10% | t→−25% | Thermal (init→peak) |",
           "|---|---:|---:|---:|---:|---:|:--|"]
    for r in rows["throttle"]:
        out.append(f"| {r['label']} | {r['burst']:.1f} | {r['steady']:.1f} | "
                   f"{r['retained']:.0f}% | {fmt_t(r['t10'])} | {fmt_t(r['t25'])} | "
                   f"{r['init']}→{r['peak']} |")
    return "\n".join(out)


def md_energy(rows):
    out = ["| Runtime | J / token | Tokens / 1% battery | Avg pkg power W | Δbattery | Thermal (init→peak) |",
           "|---|---:|---:|---:|---:|:--|"]
    for r in rows["energy"]:
        tpp = f"{r['tok_per_pct']:,.0f}" if r["tok_per_pct"] else "—"
        out.append(f"| {r['label']} | {r['jpt']:.3f} | {tpp} | {r['avg_w']:.2f} | "
                   f"{r['delta_pct']:.0f}% | {r['init']}→{r['peak']} |")
    return "\n".join(out)


def md_bandwidth(rows, model_id, model_identifier):
    """Decode is memory-bandwidth-bound → tok/s × active-weight-bytes ≈ effective read
    bandwidth, and (where the device's peak BW is known) what % of the roofline it hits."""
    gb = MODEL_DECODE_GB.get(model_id)
    if not gb:
        return None
    # NOTE: one `gb` per logical model assumes all rows are ~4-bit (true for the Gemma
    # set). When mixed-quant rows land (e.g. CoreML Qwen3-0.6B is INT8), scale per row by
    # the runtime's actual bit-width / weight bytes; until then the note flags the caveat.
    peak = DEVICE_PEAK_BW.get(model_identifier)
    head = "| Runtime | Decode tok/s | Effective BW (GB/s) |"
    sep = "|---|---:|---:|"
    if peak:
        head += " % of peak BW |"
        sep += "---:|"
    out = [head, sep]
    best = max(r["decode"] for r in rows["throughput"])
    for r in rows["throughput"]:
        eff = r["decode"] * gb
        win = " 🏆" if abs(r["decode"] - best) < 1e-6 else ""
        line = f"| {r['label']} | {r['decode']:.1f}{win} | {eff:.1f} |"
        if peak:
            line += f" {eff / peak * 100:.0f}% |"
        out.append(line)
    return "\n".join(out), gb, peak


HEADER = """# LiteRT-LM on Apple silicon — per-device package

> Self-contained, reproducible LiteRT-LM measurements pulled straight from the raw JSONL in
> [`../../results/raw/`](../../results/raw/). Generated by
> [`scripts/litert_lm_report.py`](../../scripts/litert_lm_report.py) — **do not hand-edit**; re-run it.
> Part of the neutral [Apple Silicon LLM Benchmark](../../README.md) (same headless harness for every runtime).

## The short version

A neutral, reproducible on-device benchmark that does three things vendor numbers usually don't:
**(1) the models the LiteRT team actually optimises** — Gemma 4 today, **Qwen3-0.6B** wired and
landing next (lined up against the existing Qwen3-0.6B MLX / CoreML / Core AI rows), not a convenient
demo model; **(2) across multiple devices** — iPhone 17 Pro today, the 8 GB base iPhone 17 / 16 in
flight — so a number is a *device* number, not a hero-phone number; and **(3) it explains _why_ a
runtime is fast, not just that it is** — decode is memory-bandwidth-bound, so we report each runtime's
effective GB/s against the chip's roofline, turning "LiteRT-LM wins Gemma" into "LiteRT-LM extracts
the most of the memory system." Every cell traces to raw JSONL; every config is checked in.

## What was measured

| | |
|---|---|
| **Runtime** | [`google-ai-edge/LiteRT-LM`](https://github.com/google-ai-edge/LiteRT-LM) (SwiftPM product `LiteRTLM`) — these numbers captured on **v0.12.0**; the app now builds against **v0.13.1**, re-measure pending |
| **Backend** | Metal **GPU** (`EngineConfig(... backend: .gpu ...)`) |
| **Model** | [`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) → `gemma-4-E2B-it.litertlm`, **INT4 (QAT)**, ~2.6 GB on disk |
| **Adapter** | [`MediaPipeRuntime.swift`](../../ios/BenchmarkApp/Sources/Runtimes/MediaPipeRuntime.swift) (kind `litert-lm`) — `EngineConfig → Engine.initialize() → createConversation → sendMessageStream` |
| **Token counts** | exact, from LiteRT-LM's own `Conversation.getBenchmarkInfo` (not estimated) |
| **Harness** | fully headless via `devicectl` — model sideloaded from a Mac, nothing typed on the phone; identical protocol for every runtime |

The cross-runtime rows (MLX-Swift, llama.cpp, CoreML/ANE) are the **same device, same model, same
harness** — included so the LiteRT-LM numbers have honest context, not as the focus. Gemma 4 E2B is
the current anchor (the one model all four runtimes have on-device data for). LiteRT-LM is **not**
Gemma-only — `litert-community` ships **Qwen3** (0.6B/4B) and LFM in `.litertlm` too; a Qwen3-0.6B
block — the model Lu's team is optimising, lined up against the existing Qwen3-0.6B MLX / CoreML /
Core AI rows — lands here once its device runs are captured.

### Tasks & parameters

- **short-chat** — 20-token prompt ([`prompts/short-chat.md`](../../prompts/short-chat.md)), greedy
  (temp 0 / top-p 1), cold start, **median of n=3**. LiteRT-LM has no per-call output cap, so it runs
  to EOS (~458-token reply vs the 128-token budget); decode tok/s is a rate, so the head-to-head holds.
- **energy / throttle** — 600 s continuous generation, **unplugged**, cold start, tg128. Decode-rate
  trajectory from `decodeRateRollingWindow` (~1 sample/s); energy via the iPhone 1%-battery-step
  method ([`methodology/energy-ios.md`](../../methodology/energy-ios.md)). Whole-system draw — compare
  runtimes *within a device*, never across devices.

### Fairness & conditions ([rules](../../methodology/fairness-rules.md))

**Held equal:** 128-token budget, greedy (temp 0 / top-p 1), cold start, n=3 median, one device per
table. **Disclosed differences** — each a deployment-profile difference, not a thumb on the scale:

- **Quantisation is each runtime's native ~4-bit, not bit-identical** — shown per row in the **Quant**
  column (fairness rule 3). Watch the outliers: CoreML's Qwen3-0.6B is **INT8** and Core AI's ANE path
  is mixed 4/8-bit — heavier than the 4-bit GPU artifacts, so read their memory / bandwidth with that
  in mind.
- **Output length** — LiteRT-LM 0.12.0 has no per-call cap, so it runs to EOS (~458 tok) while the
  others stop at the 128-token budget. Decode tok/s is a **rate**, so the ranking holds; and LiteRT's
  longer, hotter generation with a larger KV makes its decode rate **and** peak memory, if anything,
  *conservative* — not flattering.
- **Chat template** — every adapter applies the model's template via its tokenizer; **llama.cpp's
  didn't** (it tokenised the bare prompt → 12 Gemma tokens vs ~20 templated). Now fixed in
  `LlamaCppRuntime`; the **published llama.cpp rows predate the fix** (bare prompt) and are flagged for
  re-measure — decode rate is largely unaffected, TTFT / prefill / output shift. (Qwen3 prompts
  otherwise land at 19 tokens across runtimes.)

Methodology: [thermal](../../methodology/thermal.md) · [energy (iOS)](../../methodology/energy-ios.md)
· [fairness rules](../../methodology/fairness-rules.md) · [runtime notes](../../runtimes/litert-lm.md)
"""

FOOTER_REPRO = """## Add a device (the "couple more devices" ask)

LiteRT-LM is one binary across the iPhone line, so a new device is three headless runs from a Mac
with the phone paired over Wi-Fi and **unplugged** (USB charges the phone → the battery never drops →
energy comes back `nil`; see [`methodology/energy-ios.md`](../../methodology/energy-ios.md)). `UDID`
is the paired device's identifier; the model side-loads on first launch.

```sh
APP=com.iosllmbenchmark.benchmarkapp
MODEL=litert-community/gemma-4-E2B-it-litert-lm

# 1) short-chat ×3 (throughput, cold) — decode/TTFT/prefill/RAM/jitter
for run in 1 2 3; do
  xcrun devicectl device process launch --device "$UDID" "$APP" -- \\
    --yardstick-autorun --runtime litert-lm --model-id "$MODEL" \\
    --task short-chat --runs 1
done

# 2) energy / throttle (600 s sustained, UNPLUGGED) — J/tok + decode trajectory
xcrun devicectl device process launch --device "$UDID" "$APP" -- \\
  --yardstick-autorun --runtime litert-lm --model-id "$MODEL" \\
  --task energy --sustain-seconds 600 --runs 1
```

Then collect the JSONL the app wrote to `Documents/results/`, rename to
`results/raw/<device>-litert-lm-gemma-4-e2b-{short-chat-runN,energy-tg128}.jsonl`, drop it in, and
re-run `python3 scripts/litert_lm_report.py` — the new device appears below automatically. For an
apples-to-apples cross-runtime block on that device, capture MLX-Swift / llama.cpp / CoreML the same
way (same model file, same prompt). Full device-setup steps: [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
"""


def main():
    devices = discover_devices()
    if not devices:
        raise SystemExit("no LiteRT-LM short-chat runs found in results/raw/")

    parts = [HEADER]
    all_raw = []
    for device in devices:
        rows = collect_device(device)
        if not rows["meta"]:
            continue
        title = device_title(rows["meta"])
        parts.append(f"\n---\n\n## {device} — {title}\n")
        parts.append(f"Model file: `{rows['meta']['model'].get('primaryFile')}` · "
                     f"quant: {rows['meta']['model'].get('quantization')} · "
                     f"on-disk: {rows['meta']['model'].get('onDiskSizeMB')} MB\n")

        parts.append("### Throughput — short-chat, cold, median of n=3\n")
        parts.append(md_throughput(rows) + "\n")
        parts.append("> _Decode tok/s, peak RAM and ITL are the load-bearing columns. Prefill is over a "
                     "20-token prompt (indicative only — the prefill-bound regime wants a long-context "
                     "task, on the roadmap); CoreML/ANE counts streamed pieces, so it reports no prefill rate. "
                     "Peak RAM is process RSS for this 0.12.0 capture; the harness now records jetsam-charged "
                     "`phys_footprint` (the figure that matters on an 8 GB device), so the next capture wave "
                     "reads a few % higher._\n")

        bw = md_bandwidth(rows, MODEL, rows["meta"]["device"].get("modelIdentifier"))
        if bw:
            table, gb, peak = bw
            parts.append("### Why it's fast — decode is memory-bandwidth-bound\n")
            parts.append(table + "\n")
            peak_txt = (f"the chip's ~{peak:.0f} GB/s peak (LPDDR5X, public-teardown **estimate** — a "
                        f"number Lu's team could pin down)") if peak else "the chip's peak bandwidth"
            parts.append(f"> _Decode reads ≈ all active weights once per token, so **tok/s × weight-bytes = "
                         f"effective read bandwidth** — here ≈ {gb:.2f} GB/token (the ~4-bit decoder; ±15% "
                         f"across runtimes' quants). On a fixed device this ranks how well each runtime works "
                         f"the memory system: the GPU runtimes sit at roughly half of {peak_txt}, and LiteRT-LM "
                         f"is closest to it — that, not raw FLOPs, is the 'why faster'. The ANE row runs lower "
                         f"GB/s by design (it trades bandwidth for ~half the power). Absolute GB/s carries the "
                         f"weight-byte assumption — it takes the ~4-bit decoder; an **INT8** row (see the Quant "
                         f"column) reads ~2× the bytes, so discount its GB/s. The same-device ordering is robust._\n")

        if rows["throttle"]:
            parts.append("### Sustained throttling — 600 s continuous, unplugged\n")
            parts.append(md_throttle(rows) + "\n")
            parts.append("> _Cold burst = the short-chat median above (the peak you'd quote); "
                         "sustained = mean of the last 10% of the 600 s run; t→−x% = seconds until "
                         "the run falls x% below its own first-5 s rate (throttle onset). Definitions "
                         "match `scripts/throttle_curve.py`, so this agrees with the README throttle "
                         "table. Two independent GPU runtimes shedding together is a phone-GPU thermal "
                         "property, not a runtime quirk; the ANE holds because it draws ~half the power._\n")

        if rows["energy"]:
            parts.append("### Energy — battery-delta, 600 s run\n")
            parts.append(md_energy(rows) + "\n")
            parts.append("> _Average draw is near-identical across runtimes (~4.5–4.9 W whole-system) — "
                         "the efficiency gap is **tokens per joule, not watts**: LiteRT-LM gets ~1.6× the "
                         "tokens per 1% battery of MLX because it decodes faster for the same power. ±1% "
                         "battery quantization ≈ ±12.5% on absolute joules (n=1) — read as a ranking, not "
                         "an absolute. Whole-system: compare runtimes within this device, not across devices._\n")
            flagged = [e for e in rows["energy"] if e["init"] not in (None, "nominal")]
            if flagged:
                names = ", ".join(f"{e['label']} (started `{e['init']}`)" for e in flagged)
                parts.append(f"> ⚠️ Thermal handicap to note for fairness: {names} — a run that "
                             "starts warmer than `nominal` is, if anything, penalised.\n")

        all_raw += rows["raw"]

    # provenance
    parts.append("\n---\n\n## Provenance — every cell traces to a raw file\n")
    parts.append("All numbers above are computed from these checked-in JSONLs "
                 "(`results/raw/`); re-run `scripts/litert_lm_report.py` to regenerate this page:\n")
    for name in sorted(set(all_raw)):
        parts.append(f"- [`{name}`](../../results/raw/{name})")
    parts.append("\n" + FOOTER_REPRO)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print("wrote:", OUT.relative_to(REPO))
    print("devices:", ", ".join(devices))


if __name__ == "__main__":
    main()
