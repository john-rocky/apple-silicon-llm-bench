#!/usr/bin/env python3
"""Compare decode-throttle curves across runtimes from yardstick energy-task JSONs.

Reads each runtime's result JSON (which carries `metrics.decodeRateRollingWindow`,
one tok/s sample per ~second), and reports — per runtime — the starting rate,
the steady-state (last-10%) rate, the % drop, the time to cross 10/25% drop,
and the thermal transition. Also draws an ASCII sparkline per curve so the
"ANE stays flat, GPU throttles" story is visible at a glance.

Usage:
    python3 throttle_curve.py <result1.json> <result2.json> ...
"""
import sys, json

BARS = "▁▂▃▄▅▆▇█"


def spark(xs, lo, hi):
    if hi <= lo:
        return "─" * len(xs)
    out = []
    for x in xs:
        i = int((x - lo) / (hi - lo) * (len(BARS) - 1) + 0.5)
        out.append(BARS[max(0, min(len(BARS) - 1, i))])
    return "".join(out)


def downsample(xs, n=60):
    if len(xs) <= n:
        return xs
    step = len(xs) / n
    return [xs[int(i * step)] for i in range(n)]


def time_to_drop(rw, start, frac):
    """seconds (≈ sample index) until rate falls to (1-frac)*start."""
    target = start * (1 - frac)
    for i, v in enumerate(rw):
        if v <= target:
            return i
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: throttle_curve.py <result.json> ...")
    rows = []
    glo, ghi = 1e9, 0
    for path in sys.argv[1:]:
        d = json.load(open(path))
        m = d["metrics"]
        rw = [x for x in m.get("decodeRateRollingWindow", []) if x > 0]
        if not rw:
            print(f"(skip, no rolling window) {path}")
            continue
        start = sum(rw[:5]) / min(5, len(rw))
        tail = rw[max(0, int(len(rw) * 0.9)):]
        steady = sum(tail) / len(tail)
        drop = (1 - steady / start) * 100 if start else 0
        rows.append({
            "rt": d.get("runtime"), "rw": rw, "start": start, "steady": steady,
            "drop": drop, "n": len(rw),
            "t10": time_to_drop(rw, start, 0.10),
            "t25": time_to_drop(rw, start, 0.25),
            "peak": m.get("peakThermalState"), "init": m.get("initialThermalState"),
            "max": m.get("maxTokens") or d.get("parameters", {}).get("maxTokens"),
        })
        glo, ghi = min(glo, min(rw)), max(ghi, max(rw))

    print(f"\n{'runtime':<12} {'start':>7} {'steady':>7} {'drop%':>6} "
          f"{'t-10%':>6} {'t-25%':>6} {'len':>5} {'thermal(init→peak)':>20}")
    print("-" * 78)
    for r in rows:
        t10 = f"{r['t10']}s" if r['t10'] is not None else "—"
        t25 = f"{r['t25']}s" if r['t25'] is not None else "—"
        print(f"{r['rt']:<12} {r['start']:>7.1f} {r['steady']:>7.1f} {r['drop']:>6.1f} "
              f"{t10:>6} {t25:>6} {r['n']:>5} {(str(r['init'])+'→'+str(r['peak'])):>20}")

    print(f"\nthrottle curves (tok/s over time, scaled {glo:.0f}–{ghi:.0f}):")
    for r in rows:
        print(f"  {r['rt']:<12} {spark(downsample(r['rw']), glo, ghi)}  "
              f"{r['start']:.0f}→{r['steady']:.0f}")


if __name__ == "__main__":
    main()
