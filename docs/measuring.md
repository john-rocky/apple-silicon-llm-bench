# What gets measured

Per `(runtime, model, device, build)` tuple:

- **Speed** — TTFT, prefill `tok/s`, decode `tok/s`, sustained-decode drift over 512+ tokens.
- **Memory** — baseline, peak during decode, after-generation.
- **Thermal** — initial / peak / final state across the run.
- **Jitter** — inter-token latency `p50` / `p95` / `p99` ms, captured from the gap between consecutive `.chunk` events. Surfaces the worst-case stall a chat UI will perceive even when the average decode rate looks smooth.
- **Energy** — joules per token. iOS uses the 1%-battery-step API; Mac uses `scripts/measure_energy.py` (wraps `powermetrics`, see "Optional: capture Mac energy" below).
- **Lifecycle** — survives background → foreground, cancellation latency, streaming.
- **Quality** — one GSM8K score per build today (n=100, one harness for every build; the [iPhone 17 Pro Gemma 4 E2B table](results-iphone.md)). WER / CER for ASR, perplexity / MMLU for LLM and byte-identical comparison vs Python references are on the roadmap (Phase 4).

Methodology lives under [`methodology/`](../methodology/). The numbers we publish follow [`methodology/fairness-rules.md`](../methodology/fairness-rules.md).

### Optional: capture Mac energy with `powermetrics`

```sh
YARDSTICK_BIN=.build/release/yardstick python3 scripts/measure_energy.py run \
     --task short-chat --runtime mlx-swift \
     --model mlx-community/gemma-4-e2b-it-4bit \
     --output results/raw/<device>-<runtime>-<model>-<task>-energy.jsonl
```

Don't run it with `sudo`. It asks for your password once, for `powermetrics`.

The wrapper starts `powermetrics` in the background, runs `yardstick`,
stops `powermetrics`, then patches the JSONL with `energyJoules`,
`averagePackagePowerW`, and `energyJoulesPerToken`. Numbers are
whole-system — run on an idle desktop and use them to compare
runtimes on the same Mac, not Macs to each other.

### Optional: import iPhone / iPad runs

The iOS app's **History → ••• → Export all (JSONL)** sheet hands you a
single newline-delimited file. AirDrop it to your Mac, then:

```sh
python3 scripts/import_ios_export.py <file> --device <label>
python3 scripts/render_results.py
```

`<label>` is a key of `DEVICE_DISPLAY` in `scripts/bench_common.py`, such as `iphone17pro`. The import script takes one file at a time.

The import script splits the bundle into one
`results/raw/<device>-<runtime>-<model>-<task>-runN.jsonl` per row,
re-keying the device label so `render_results.py` recognises it.
