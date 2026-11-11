# MacBook Air — Apple M3 (16 GB)

Secondary Mac reference. Useful precisely because it is the *low* end of
"runnable" — the fanless / 16 GB combination is where many on-device LLM
assumptions break.

| Field | Value |
|-------|-------|
| Chip | Apple M3 |
| GPU cores | 10 |
| Neural Engine | 16-core ANE |
| RAM | 16 GB |
| macOS version tested | macOS 26 |
| Storage class | Internal SSD |
| Cooling | Passive / fanless |
| Power | Plugged in, "Automatic" power mode |

## Notes

- Fanless: sustained-decode runs *will* throttle eventually; the
  `sustained` task is designed specifically to surface this.
- 16 GB RAM: 4-bit 7B models fit but with little headroom; FP16 LLMs
  are out of reach. Choose Q4 / Q6 quantizations.
- Useful comparison vs. the M4 Max: same architecture family, very
  different thermal and memory budgets.

## Build

For now, run via the iOS BenchmarkApp on a connected iPhone, or wait
for the Phase-2 macOS app target (tracked in
[`../docs/project.md`](../docs/project.md#roadmap)).

## Results

See the runtime/model rows in [`../RESULTS.md`](../RESULTS.md) filtered
to `MacBook Air M3`.
