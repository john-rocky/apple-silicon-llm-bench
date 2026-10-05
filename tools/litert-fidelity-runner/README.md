# litert-fidelity-runner

Mac runner for the backend-fidelity suite (`methodology/fidelity.md`). The executable is
still named `litert-mac-verify`: it is the yardstick source in `tools/litert-mac-verify/`
plus the `--suite` batch mode. It lives in its own directory because that yardstick is
pinned to the exact wrapper revision behind the published GSM8K rows and must not move.

```bash
./setup.sh    # clones swift-litert-lm @ 84b5df7, applies wrapper.patch, builds
.build/release/litert-mac-verify <model.litertlm> \
  --suite ../../evaldata/fidelity_suite_v1.jsonl --out out.jsonl \
  --backend cpu|gpu --tier smoke|core|full --reps N --runtime-version v0.17.1 \
  [--suite-max-num-tokens M] [--activation f32] [--resume] [--timeout S]
```

- The LiteRT-LM version is whatever `wrapper.patch` pins (v0.17.1). `--runtime-version` only
  labels the records; change the pins and the label together.
- `--activation f32` and `--suite-max-num-tokens M` change the recorded backend label
  (`gpu-f32`, `gpu-ctx8192`), so the scorer treats them as separate arms.
- `maxNumTokens` is total context for this runtime. Exit code 3 means one generation timed
  out; rerun with `--resume`.
- `wrapper.patch` adds one non-upstream global to the vendored Swift wrapper
  (`liteRTLMActivationDataTypeOverride`), which calls
  `litert_lm_engine_settings_set_activation_data_type`.
