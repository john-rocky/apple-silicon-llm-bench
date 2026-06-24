# Gap-fill session — wire the new Core AI bundles into the bench + measure (run AFTER the export session)

**Your job:** the export session produced Core AI bundles for up to 6 models in
`~/code/coreai/coreai-models/exports/` (see its hand-off checklist). Wire each into the benchmark app, side-load,
measure on iPhone (+ Mac), and fill the blanks in the report. You work in this repo
(`~/Downloads/ios-llm-benchmark`). **iPhone 17 Pro must be connected.**

## Prereqs (verify before starting)
- For each model to fill: `exports/<name>_gpu/` and `exports/<name>_ane_pure4bit/` each have a `*.h18p.aimodelc`
  + `tokenizer/` + `metadata.json`. (Ministral/Gemma3/Llama may be GPU+ANE or, where only the iOS class was
  written, both; Mac-only macOS bundles are fine for the Mac column.)
- The app build has the **memory entitlements** (`increased-memory-limit` + `extended-virtual-addressing`).

## ⚠ Critical (don't repeat past mistakes)
- **Keep the memory entitlements.** Without them ≳2 GB models *falsely* OOM (this cost a whole session). They're
  in `ios/BenchmarkApp/BenchmarkApp.entitlements` + `project.yml`; the App-ID capabilities are registered. If a
  CLI `xcodebuild -allowProvisioningUpdates` fails on provisioning ("No Accounts" / capability not in profile),
  rebuild once from **Xcode GUI** (it has the account), then CLI builds work off the cached profile.
- **Never run an iOS bundle on the Mac** — measure on-device only.
- **ANE cold-load compiles on-device** (slow first run, warms after). Run 3× cold and take the median.

## Per model — wire it (3 edits, mirror the existing qwen3 / deepseek entries)
1. **`ios/BenchmarkApp/Sources/Models/ModelCatalog.swift`** — add a `ModelInfo` for `core-ai/<model>-ane` and
   `core-ai/<model>-gpu` (`hfRepoId: ""`; copy a qwen3/deepseek core-ai entry as the template).
2. **`ios/BenchmarkApp/Sources/Runtimes/CoreAIRuntime.swift`** — add bundleSpec cases (before `default: return nil`):
   ```
   case "core-ai/<model>-ane": return ("<name>_ane", "static-shape")
   case "core-ai/<model>-gpu": return ("<name>_gpu", "coreai-pipelined")
   ```
   (device folder = `<name>_ane` / `<name>_gpu`; the ANE *source* dir is `exports/<name>_ane_pure4bit`.)
3. **`results/raw/2026-06-25-comprehensive/manifest.tsv`** — add side-load rows:
   `<Family>\t<params>\tcore-ai\tcore-ai/<model>-ane\tEX/<name>_ane_pure4bit\tCoreAIModels/<name>_ane` (and `-gpu`).

## Build → side-load → measure
1. **Rebuild** (keep entitlements): `xcodebuild -project ios/BenchmarkApp/BenchmarkApp.xcodeproj -scheme BenchmarkApp
   -configuration Release -destination 'generic/platform=iOS' -allowProvisioningUpdates DEVELOPMENT_TEAM=MFN25KNUGJ
   -jobs 6 -derivedDataPath ~/bench-dd build`, then install. (Or Xcode GUI if provisioning balks.)
2. **Side-load:** `scripts/comprehensive_bench.sh stage` (picks up the new manifest rows) — or per-bundle
   `xcrun devicectl device copy to … --source exports/<name>_… --destination Documents/CoreAIModels/<name>_…`.
3. **Measure iPhone:** `scripts/comprehensive_bench.sh speed <Family>` → short-chat 3× cold per (ane/gpu). Record
   decode_tok_s / ttft_ms / peak_mb. (Energy later, per the comprehensive runbook — optional here.)
4. **Measure Mac column:** bench the **macOS** Core AI bundle with coreai-models' `llm-benchmark` (allowed on Mac).

## Update the report + raw data
- `docs/litert-community-vs-mlx-coreai.md` **and** `~/code/litertlm-convert/reports/litert-community-vs-mlx-coreai.md`
  (kept in sync): fill the Core AI cells in the Mac + iPhone tables for each newly-measured model; remove the
  `✗ no class` / `untested` notes for those rows.
- `results/raw/2026-06-24-coreai-iphone/results.jsonl` (or a new dated dir): append the measured rows.
- **Commit + push.** Per `~/.claude/CLAUDE.md`: no "claude" in committer or message; don't commit CoreML/model
  files or build files; keep the repo minimal.

## Done when
All 6 fillable models show Core AI numbers (Mac + iPhone) in the report. The only remaining blanks are the 3
permanent MLX cells (Ministral iPhone-MLX = MLX-Swift arch hard block; VibeThinker & OLMo-2 MLX = no
mlx-community repo). Update `methodology/next-session-brief.md` status if you close the matrix.
