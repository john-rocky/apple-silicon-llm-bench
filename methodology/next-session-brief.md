# NEXT-SESSION BRIEF — start here (written 2026-06-24)

Self-contained handoff for the next session(s). Two work streams: **(A) comprehensive on-device bench** (needs
the iPhone), **(B) Core AI export** to close every blank (no device needed until the measure step). Read the
"DON'T REPEAT" list first — this session burned a lot of cycles on three wrong conclusions.

## What this project is
Cross-framework LLM benchmark — **Core AI vs MLX vs LiteRT-LM**, on **Mac M4 Max + iPhone 17 Pro**, decode tok/s
+ memory + (next) energy. Report for the Google LiteRT team. Public repo: github.com/john-rocky/apple-silicon-llm-bench.
Headline finding (holds Mac + on-device): **Core AI ≈ MLX ≫ LiteRT-LM; on iPhone the ANE beats MLX** (DeepSeek-R1
ANE 83 vs MLX 73). Report: `docs/litert-community-vs-mlx-coreai.md`.

## ⚠ DON'T REPEAT (this session's corrected mistakes — all verified)
1. **`externalize_embedder` is NOT a LiteRT bug.** Refuted: a 1.7B extemb build loads 5/5; issue #2645 was filed,
   then RETRACTED + apologized. Don't re-file.
2. **The iPhone 3B "OOM" was OUR app missing memory entitlements**, not LiteRT and not a mmap-loader limit.
   `increased-memory-limit` + `extended-virtual-addressing` (both required) fixed it: Ministral-3B 3/3, Llama-3.2-3B
   5/5. The entitlements are in `ios/BenchmarkApp/BenchmarkApp.entitlements` + `project.yml`; the App-ID
   capabilities are registered (added via Xcode GUI — CLI `-allowProvisioningUpdates` can't register them).
   **Any new build must keep these or ≳2 GB models will falsely OOM.**
3. **Phi-4-mini iPhone "OOM" is the int8 quant (3.6 GB), not Phi/LiteRT.** MLX runs Phi int4 (~2 GB) fine; an int4
   LiteRT Phi would load. (int4 Phi convert is blocked by a litert_torch/transformers incompat — separate issue.)
4. **The Mac crashes from running iOS bundles on the Mac** (e.g. litert-mac-verify on an iOS artifact). Exports +
   coreai-build compiles are safe. Verify on-device, never run iOS bundles on the Mac.
5. **Verify before concluding/filing.** Every "this is a real limit" claim here was wrong until isolated. Isolate
   with a controlled test (small model, on/off, multiple cold runs, read the actual device log).

## Current state (done)
- 10-model matrix measured (Mac 26/30, iPhone filled per the report). Core AI iPhone complete for qwen2/qwen3
  (incl. VibeThinker ANE 71.5, filled this session).
- Comprehensive-bench harness verified: speed + memory (`phys_footprint`) + energy task all exist.
- All side-load bundles staged on Mac (Core AI 14/14, LiteRT-local 8/8).

---

## STREAM A — Comprehensive bench (needs iPhone 17 Pro connected)
Goal: speed + memory + **energy** across every comparable model × runtime. Full guide:
**`methodology/comprehensive-bench-runbook.md`**. Matrix: `results/raw/2026-06-25-comprehensive/manifest.tsv`.
Driver: `scripts/comprehensive_bench.sh`.

1. Confirm the app on device is the entitlement build (rebuild via Xcode if unsure — see DON'T-REPEAT #2).
2. `scripts/comprehensive_bench.sh stage`   (USB; side-loads all bundles)
3. `scripts/comprehensive_bench.sh speed`   (plugged OK; short-chat 3× → decode/TTFT/memory)
4. Energy (UNPLUGGED, selective, battery-limited): `… energy <model-id> <runtime>` then unplug, wait, `… collect`.
   **The money shot = Core AI ANE vs GPU J/token on the same model** (`qwen3-1.7b-ane` vs `-gpu`).

## STREAM B — Core AI export to fill every blank (split into two separate sessions)
6 export classes/shims → all 10 models become 3-way Core AI (Mac + iPhone); only 3 MLX cells stay permanently
blank (Ministral iPhone-MLX hard block; VibeThinker/OLMo MLX = no repo). Run as **two independent sessions:**

1. **Export session** → `methodology/coreai-export-session.md` — self-contained, works only in
   `~/code/coreai/coreai-models` (no device, no bench knowledge). Writes the classes, produces the bundles.
   Order: Ministral (config shim) → Gemma3-iOS, Llama-iOS (macOS done) → Phi3, OLMo-2, SmolLM3 (new archs; SmolLM3
   has a NoPE quirk).
2. **Gap-fill session** → `methodology/gap-fill-session.md` — runs after, with the iPhone. Wires the bundles into
   the app (catalog + bundleSpec + manifest), rebuilds (keep entitlements), side-loads, measures, updates the report.

---

## Repo map (key files)
- `docs/litert-community-vs-mlx-coreai.md` — the report (Mac + iPhone tables, all corrected).
- `results/raw/2026-06-24-coreai-iphone/extemb-isolation/SUMMARY.txt` — the full debugging trail (extemb refute →
  entitlement root cause → Phi quant).
- `methodology/comprehensive-bench-runbook.md` · `coreai-export-todo.md` — the two detailed playbooks.
- `~/Downloads/meeting/` — Doc paste blocks, the correction comment for Shuangfeng, the retracted-bug note.
- Memory: `litert-community-crossframework-bench.md` (status + the corrected findings).
