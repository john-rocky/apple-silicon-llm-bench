#!/bin/bash
# Fetch the Swift wrapper this runner builds against and apply wrapper.patch
# (xcframework pins -> LiteRT-LM v0.17.1, plus the activation-data-type override).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d swift-litert-lm ]; then
  git clone https://github.com/john-rocky/swift-litert-lm.git
  git -C swift-litert-lm checkout --detach 84b5df7
  git -C swift-litert-lm apply ../wrapper.patch
fi
swift build -c release
echo "built: $(pwd)/.build/release/litert-mac-verify"
