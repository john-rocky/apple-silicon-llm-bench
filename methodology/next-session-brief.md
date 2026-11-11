# NEXT-SESSION BRIEF — start here (written 2026-06-24)

> **2026-06-25 UPDATE — Core AI matrix CLOSED (both streams done).** Mac Core AI **9/10** (added OLMo-2 384.4 /
> SmolLM3 192.9 / Ministral-3 186.0; only Phi partial-RoPE blocked). iPhone Core AI **16/20** measured — 6 new models:
> Gemma3-1B GPU 103.6, OLMo-2 ANE 95.6 / GPU 86.1, Llama-3.2-3B ANE 24.2 / GPU 19.3, SmolLM3 ANE 23.0 / GPU 20.5,
> Ministral-3B GPU 17.6. **New findings (corrected the brief):** (1) **Phi partial-RoPE is a wall on Mac *and* iPhone,
> GPU *and* ANE** (`Array index out of range` / `RoPE freqs [48]≠[64]`) — not just iPhone-ANE; (2) **at 3B on iPhone MLX
> beats Core AI** (Llama/SmolLM3); Core AI wins ≤1.7B-qwen + 1B; (3) **ANE > Core AI's own GPU throughout.** Report (both
> copies) + raw JSONL updated & committed.
>
> **⚠ BUILD-SIGNING GOTCHA (new, saves a session):** the canonical App ID `com.iosllmbenchmark.benchmarkapp` is
> "not available" to re-register (taken by another team) → automatic signing can't mint a capabilities profile, and
> the GUI hits the same wall. **Workaround that worked, fully CLI, no repo changes:** build with `CODE_SIGN_STYLE=Automatic`
> + a temporary `PRODUCT_BUNDLE_IDENTIFIER` override to an existing MFN25KNUGJ App ID that already carries both memory
> caps (e.g. `com.example.CoreMLLLMChat`, cached profile in `~/Library/Developer/Xcode/UserData/Provisioning Profiles/`),
> install, then drive devicectl with that bundle id (a scratch copy of `comprehensive_bench.sh` with `APP=` swapped).
> The pbxproj now carries `CODE_SIGN_ENTITLEMENTS` so a normal `DEVELOPMENT_TEAM=… -allowProvisioningUpdates` build also wires the caps.

Self-contained handoff for the next session(s). Two work streams: **(A) comprehensive on-device bench** (needs
the iPhone), **(B) Core AI export** to close every blank (no device needed until the measure step). Read the
"DON'T REPEAT" list first — this session burned a lot of cycles on three wrong conclusions.

## What this project is
Cross-framework LLM benchmark — **Core AI vs MLX vs LiteRT-LM**, on **Mac M4 Max + iPhone 17 Pro**, decode tok/s
+ memory + (next) energy. Public repo: github.com/john-rocky/apple-silicon-llm-bench.
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

## STREAM B — Core AI export DONE (2026-06-25); only the bench-continuation remains
6 models exported + already wired (catalog + bundleSpec + manifest, all bundles verified). **Mac Core AI now
10/10** (export session added macOS classes phi3/olmo2/smollm3/ministral, all parity PASS). **iPhone: GPU for all
6, ANE for Llama/OLMo-2/SmolLM3.** So Stream B reduces to one session:

→ **Gap-fill / bench-continuation** → `methodology/gap-fill-session.md` — **rebuild (keep entitlements) →
  side-load → measure (9 iPhone cells + the Mac column) → update the report.** Wiring is in place.

**6 permanently-blank cells, each with a documented reason** (do NOT chase — confirmed walls; see gap-fill doc):
3× iPhone-ANE (Phi = compiler SIGSEGV on partial-rotary; Ministral = multimodal-fp8 loader; Gemma3 =
dual-RoPE/sliding not expressible on the standard pipeline) + 3× MLX (Ministral iPhone arch block; VibeThinker /
OLMo-2 no repo). Export playbook for any future arch: `methodology/coreai-export-session.md`.

---

## Repo map (key files)
- `docs/litert-community-vs-mlx-coreai.md` — the report (Mac + iPhone tables, all corrected).
- `results/raw/2026-06-24-coreai-iphone/extemb-isolation/SUMMARY.txt` — the full debugging trail (extemb refute →
  entitlement root cause → Phi quant).
- `methodology/comprehensive-bench-runbook.md` · `coreai-export-todo.md` — the two detailed playbooks.
- `~/Downloads/meeting/` — report text, the correction note, the retracted-bug note.
- Memory: `litert-community-crossframework-bench.md` (status + the corrected findings).
