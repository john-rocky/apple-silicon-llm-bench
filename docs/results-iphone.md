# iPhone 17 Pro results

Real LLM inference on a phone — on-device, no server. iPhone 17 Pro, short-chat (128 tokens), thermal-nominal. **The winning runtime depends on what you optimize for — speed, memory, or quality.** Gemma rows are the **2026-07-28/30 fairness re-capture** (warm protocol: runs 2–3 median of n≥6 launches, run 1 dropped as cold, ctx forced 2048); per-number dates and n are stated in the table.

> **2026-07-28/30 fairness re-capture.** Every Gemma comparison below is now like-for-like:
> one warm protocol, one prefill instrument for every runtime (a harness bug had discarded
> LiteRT-LM's task-path prefill counters on capped runs — fixed in the r3 harness and proven
> on device), both MLX rows in one session, and memory on the median charged footprint.
> Full records with per-number n and number-trace audits:
> [`results/raw/2026-07-29-gemma4-e2b-protocol/`](../results/raw/2026-07-29-gemma4-e2b-protocol/README.md) (iPhone)
> and [`results/raw/2026-07-28-gemma4-e2b-protocol-mac/`](../results/raw/2026-07-28-gemma4-e2b-protocol-mac/README.md) (Mac).
> **Energy re-published on a rebuilt instrument**: the old J/token figures are retired —
> iOS 27's battery gauge reports in 5% steps and start/end deltas swing ×2 between identical
> runs ([evidence](../results/raw/2026-07-30-gemma4-e2b-protocol/README.md)). The ◊ column now
> measures between gauge transitions (tick-window, audited); on it **LiteRT leads iPhone
> energy** (LiteRT 0.147 vs MLX 0.182) while the Mac's GPU-only powermetrics basis has MLX first —
> the energy verdict genuinely differs by OS.

### Gemma 4 E2B — every runtime at its best *available* build (2026-07-18 + Cactus 2026-07-20, iOS 27.0)

![Gemma 4 E2B on iPhone 17 Pro — decode, peak memory, GSM8K per runtime at its best available build](charts/iphone_gemma4_bestavailable.png)

The earlier version of this table had each runtime on a different checkpoint quality class (MLX and llama.cpp on PTQ, LiteRT on QAT) — it measured who had the better checkpoint, not the better runtime. This one states the build per row, the capture session per number, and adds **GSM8K n=100** (measured on M4 Max with one identical harness for every row — greedy, thinking-off, same extractor). Decode/ITL are the warm protocol; memory is the median charged footprint over the same warm session:

| Runtime | Build | Decode tok/s | ITL p50 | Mem MB (median fp) | GSM8K | J/tok ◊ |
|---|---|---:|---:|---:|---:|---:|
| 🔴 LiteRT-LM | wNa8o8 QAT (official) | **61.1** 🏆 (A2 7/28, n=8) | **16.3 ms** | **497** 🏆 | 86.0% | **0.147** 🏆 ◊ |
| 🌵 Cactus ¶ | CQ4 **uncalibrated** (their pre-07-09 build) | 50.6 (7/20 cold ×3 — warm not re-measured) | 19.6 ms (7/20) | 1,061 (7/20, peak basis) | 87.0% | 0.222 ◊ |
| 🌵 Cactus | **CQ4 as shipped** (`cactus run` default) | 50.0 (C 7/29, n=8) | 19.8 ms | 632 | **3.0%** | 0.226 ◊ |
| 🟣 MLX-Swift | PTQ 4-bit | 49.1 (A2 7/28, n=6) | 20.5 ms | 3,010 | 84.0% | 0.182 ◊ |
| 🍎 Core AI ‡ | own int4 (from official QAT q4_0) | 47.1 (C 7/29, n=8) | 21.4 ms | 755 † | 88.0% | — (structural ◊) |
| 🔵 llama.cpp | Q4_K_M (PTQ) | 38.8 (7/27, n=10) | 24.4 ms (7/27) | 191 † | 76.0% | 0.260 ◊ |
| 🟣 MLX-Swift | QAT OptiQ int4 | 36.0 (A2 7/28, n=6) | 27.9 ms | 4,592 | **91.0%** 🏆 | 0.231 ◊ |
| 🔵 llama.cpp | **official QAT q4_0** | **unloadable** | — | — | — | — |

> ◊ **Tick-window instrument (r4, audited).** iOS 27's battery gauge reports in 5% steps, so start/end deltas swing ×2 between identical runs ([evidence, all 12 retired results](../results/raw/2026-07-30-gemma4-e2b-protocol/README.md)); J/tok here is instead measured **between gauge transitions** (2 complete ticks per row, transition timestamps in the raw JSONs, nominal-start enforced, unplugged). Steps aren't equidistant, so a row carries ±~10% — the Cactus-uncal 0.222 / Cactus-shipped 0.226 / MLX-OptiQ 0.231 trio is one **unresolved cluster**; LiteRT < MLX < cluster < llama.cpp are resolved. Core AI is structurally excluded: the equalized per-call budget (2048) exceeds its iOS KV cap (1,024) and the app is jetsammed at setup.
> † mmap'd weights: clean pages aren't charged to `phys_footprint`, so these numbers are not comparable with runtimes that wire their weights — llama.cpp's 191 MB hides 2.9 GB of mmap'd GGUF that shows up in residency (3.1 GB). LiteRT-LM is the only runtime under a gigabyte on **both** the footprint and resident columns, which is the honest form of the memory claim.
> ‡ **Patched engine (reference)**: Apple ships no Gemma-4 bundle and `EngineOptions.staticInputBuffers` is a local engine patch — but the *path* is Apple's standard `EngineFactory`. Its TTFT is the honest cost of S=1 unbatched prefill (Gemma-4's per-layer embeddings force it).
> ¶ **Cactus row = the build they demoted, because it is their best usable.** On 2026-07-09 Cactus replaced its default CQ4 and renamed the original to `-uncalibrated`. The two are **speed-identical** but the shipped default is reasoning-dead: **GSM8K 3.0%** on the same one-harness protocol, while the demoted build scores **87.0%** — the QAT-class band. Their newer mixed-precision prebuilts (cq3.26/cq2.54) probe at 4%/16% (n=25). Same best-usable rule that puts MLX on OptiQ; the uncalibrated zip is a manual download from the same HF repo. Engine = their Metal GPU default via `cactus_init`/`cactus_complete`, cloud-handoff + telemetry forced off, exact engine-reported token counts. Cross-session ratios go through a same-session LiteRT anchor: at matched conditions Cactus-shipped decodes at **0.84× LiteRT** (block C, 2026-07-29: 50.0 vs 59.4 — re-confirming the 0.83× the 7/20 control measured).

- **The decode+memory upset survives the fairness fix — and now a 7th row.** LiteRT-LM still beats every loadable build on decode and every wired-memory build on footprint (2.2–9.5×). At **matched quality** (LiteRT 86.0 vs MLX-PTQ 84.0) it is 1.14× faster with 6× less memory. **Cactus arrives as the clear #2 on decode** — at matched same-session conditions it runs 0.83× LiteRT's rate (¶, anchor-adjusted) at *higher* GSM8K (87.0 vs 86.0), paying 2.2× the memory and 2.6× the energy.
- **Quality still goes to MLX-OptiQ: 91.0%** — +5 pts over the wNa8o8 build LiteRT ships, at 0.66× its decode. No runtime is Pareto-dominant once quality is on the table: **speed/memory/energy → LiteRT-LM, quality → MLX-OptiQ, balance → Core AI or Cactus-uncalibrated** (Cactus: 0.83× LiteRT's decode at +1 GSM8K pt; Core AI: +2 pts at 0.65×).
- **The Cactus finding is an artifact-lineage story, not an engine story** (¶): the engine is genuinely fast (second-fastest decode measured here), and the build its own CLI ships scores 3.0% on GSM8K while the build it demoted scores 87.0%. "Which file did the runtime hand you" is worth 84 points — the sharpest case yet for stating the build per row.
- **Google's official QAT GGUF does not load** — llama.cpp aborts on a vocab defect ("empty token at index 237922", reproduced through the latest release b10064; the third-party Q4_K_M loads fine, so it is that file's conversion). The official-QAT row *is* the measurement: shipping an artifact ≠ shipping a usable artifact. llama.cpp's usable best is also the table's quality floor (76.0%).
- **PTQ→QAT re-measured with stored reports: 84.0 → 91.0 (+7 pts)** — supersedes the earlier "78 → 87" claim from the defective-harness era.
- **Energy (battery-delta, 600 s sustained, unplugged): wNa8o8 wins on-device energy too** — 24 % more tokens on the same 5 % of battery than MLX-PTQ (0.122 vs 0.151 J/tok), *reversing* the Mac result where MLX owns the energy Pareto — and it throttles less (76 % vs 64 % of burst rate retained). Core AI jetsams under the standard deep protocol (its known depth wall — the failed run stays on record per fairness rule #4); measured via a shallow-rep variant (`--max-tokens 192`, depth kept under the wall) it lands at **0.352 J/tok** — a favorable-bias lower bound that still spends ~2.9× LiteRT's energy per token. **llama.cpp is the energy floor at 0.483 J/tok**: it pulls ~2× any other build's average power (9.8 W), hits the thermal ceiling hardest, and keeps the least of its burst rate under sustained load (54 %, vs LiteRT 76 / OptiQ 67 / MLX-PTQ 64 / Cactus 57 / Core AI 56) — the same 1.9× llama-vs-MLX gap the Mac shows, amplified by the phone's thermal loop. **Cactus lands at 0.322 J/tok** (2026-07-20 capture, 600 s standard deep protocol, 10 % battery in one window at 9.2 W — the second-highest draw; sustained 28.7 tok/s = 57 % of burst; entered the window at `nominal` where the 07-19 runs entered at `fair`, a slightly favorable regime — disclosed): its near-LiteRT burst speed costs 2.6× LiteRT's energy per token. The spread on one phone (0.122 → 0.483, 4.0×) is this study's cleanest evidence that *runtime engineering, not silicon, sets the on-device energy bill*. A later audit found that the LiteRT-LM and MLX runs above started at thermal state `fair`, and later runs found that the battery gauge reports in 5% steps, so these J/tok figures don't rank the runtimes ([audit](../methodology/CAMPAIGN-gemma4-e2b-full-recapture.md), [later runs](../results/raw/2026-07-30-gemma4-e2b-protocol/README.md)).

- **The upset — Gemma 4 E2B (re-verified under the fair protocol):** Google's **LiteRT-LM** (wNa8o8 QAT, GPU, its native `.litertlm`) beats MLX-Swift on decode (61.1 vs 49.1 warm), uses ~6× less memory (497 vs 3,010 MB median footprint — and it is the only runtime under 1 GB on the resident column too), **and now wins prefill on the unified instrument** (3,513 vs 3,274 tok/s at p=1024, same task, same session — the old cross-instrument prefill ranking is retired). The purpose-built runtime wins on its own format — and on the audited tick-window instrument it also leads iPhone energy (0.147 vs 0.182 J/tok; the Mac's GPU-only basis flips to MLX).

### Qwen 3.5 2B (pre-refresh numbers — Debug builds, iOS 26.4.2, 2026-05-28)

| Runtime | Decode tok/s | Peak MB |
|---|---:|---:|
| 🟣 MLX-Swift | **61.2** 🏆 | 1,279 |
| 🔵 llama.cpp | 39.1 | 1,479 |
| 🟠 CoreML/ANE | 27.9 | **241** 🏆 |

> ⚠️ Debug-build captures ([fairness-rules #7](../methodology/fairness-rules.md)); a Release re-capture is pending. The CoreML/ANE runtime is **off by default** in current builds (the author's own library, kept out of the neutral default; its chunked-MLKV 241 MB footprint stands as the memory reference). No LiteRT-LM row at this size — a Qwen3-0.6B `.litertlm` match is wired and pending.

- **Counting:** MLX / llama.cpp / LiteRT-LM report exact tokenizer tokens (LiteRT-LM via `getBenchmarkInfo`); CoreML/ANE counts streamed pieces (≈ tokens). Since the r2 harness every runtime is capped at the same 128-token budget; decode tok/s is a rate, so the head-to-head holds.
- **Fully automated, side-loaded** via `devicectl` headless mode — nothing typed on the phone, same methodology as the desktop rows.
- **Coming next:** Apple Foundation Models, chart refresh for the new Gemma table, more models and more iPhones / iPads. [One row is a great PR](../CONTRIBUTING.md).

> **How the LiteRT-LM row was measured (updated 2026-07-30):** `google-ai-edge/LiteRT-LM` running `litert-community/gemma-4-E2B-it.litertlm` (wNa8o8 QAT) on the Metal **GPU** backend, via the in-tree [`MediaPipeRuntime`](../ios/BenchmarkApp/Sources/Runtimes/MediaPipeRuntime.swift) adapter — same headless harness + prompt as every other row. Since the r2 harness the row is capped at the same 128-token budget as every runtime and timed on the harness wall-clock column; since r3 the engine's own prefill/decode counters (`Conversation.getBenchmarkInfo`) are kept on capped runs too, so counts are exact, not estimated. Memory = jetsam-charged `phys_footprint`. First-ever load builds device caches (~112 s once); cold loads after that are 1.4–3.6 s. LiteRT-LM is vendored as a **local SwiftPM package** (`scripts/bootstrap.sh` clones it with `GIT_LFS_SKIP_SMUDGE=1`; the released package trips SwiftPM's unsafe-flags rule via its `-all_load`).
>
> **How the CoreML/ANE rows were measured:** `john-rocky/CoreML-LLM` on the Neural Engine (`computeUnits: .cpuAndNeuralEngine`) — Gemma 4 E2B via the chunked `.mlmodelc` path, Qwen 3.5 2B via `Qwen35MLKVGenerator` (chunked MLKV, hence the 241 MB). Decode counts streamed pieces (≈ tokens); first-load ANE compilation makes its load time high (and it's the lowest-throughput runtime — the ANE trades speed for memory).
>
> Decode tok/s is the headline number; the full per-run audit (prefill, TTFT, inter-token jitter, memory) lives in [`RESULTS.md`](../RESULTS.md). The 2026-07-18 Gemma-4-E2B session's raw per-run JSONs and audit trail (including the thermally-excluded captures) live in [`results/raw/2026-07-18-gemma4-bestquant/`](../results/raw/2026-07-18-gemma4-bestquant/SUMMARY.txt) pending the RESULTS.md importer's extension to that format.
