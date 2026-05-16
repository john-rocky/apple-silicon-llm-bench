---
name: Add a benchmark row
about: Volunteer to run the Yardstick harness on a device / runtime / model we don't have yet
title: "[bench] <device> + <runtime> + <model>"
labels: ["bench", "wanted"]
---

## What I'm offering to run

- **Device:** (e.g. iPhone 17 Pro, MacBook Air M3 16 GB, M2 Mac Studio)
- **OS:** (e.g. iOS 26.4, macOS 26.0)
- **Runtime(s):** (mlx-swift / llama.cpp / coreml-llm / executorch / anemll / litert-lm)
- **Model(s):** (HF repo ids — verify they're in the [`ModelCatalog`](../ios/BenchmarkApp/Sources/Models/ModelCatalog.swift) for your runtime, or PR them in)
- **Task(s):** short-chat / long-context / sustained / lifecycle (default = short-chat)

## Why this row is useful

(Optional — what does this row let readers compare? "First M2 Studio row" / "First Qwen 3.5 9B run on iPhone Pro" / etc.)

## Status

- [ ] Ran the runs locally (`n ≥ 3` for short-chat)
- [ ] JSONLs are in `results/raw/` with the `<device>-<runtime>-<model>-<task>-runN.jsonl` filename convention
- [ ] Added a `devices/<device>.md` page if my device wasn't represented yet
- [ ] Added my device to `scripts/render_results.py::DEVICE_DISPLAY` if needed
- [ ] `python scripts/render_results.py --check` is clean
- [ ] Opening a PR

## Notes / blockers

(Anything that bit you — `Failed to load`, throttling, etc. Failed runs are signal — please leave them in the PR.)
