#!/bin/bash
# fetch_runner.sh <work dir> — fetch and build the LiteRT-LM #3913 copy-fidelity runner (PowerBeef's
# research/issue-3913 at commit 39d5a014, LiteRT-LM v0.18.0 Swift package) into <work dir>/LiteRT-LM.
#
# The extracted tree MUST sit under a directory named exactly "LiteRT-LM": runner/Package.swift depends on
# .package(path: "../../..") and names the product's package "LiteRT-LM"; SwiftPM takes a path dependency's
# identity from the directory name, so any other name fails with "unknown package 'LiteRT-LM'".
# The archive's own sha256 is not stable (GitHub regenerates it); the eight case files are checked instead
# (cases.SHA256SUMS next to this script). `swift build` runs through quiet_wait.py when that tool exists.
set -euo pipefail
W=${1:?usage: fetch_runner.sh <work dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=39d5a014fab329ca25c41b7bd4df76c7dacea490
URL=https://github.com/PowerBeef/agello-litert-lm/archive/$COMMIT.tar.gz
QW=$HOME/code/standup/tools/quiet/quiet_wait.py
mkdir -p "$W/LiteRT-LM" "$W/dl" "$W/logs"
cd "$W"
if [ ! -f "LiteRT-LM/research/issue-3913/runner/Package.swift" ]; then
  [ -s "dl/runner-$COMMIT.tar.gz" ] || curl -sSL -o "dl/runner-$COMMIT.tar.gz" "$URL"
  top=$(tar -tzf "dl/runner-$COMMIT.tar.gz" | head -1 | cut -d/ -f1)
  tar -xzf "dl/runner-$COMMIT.tar.gz" -C LiteRT-LM --strip-components 1 \
    "$top/Package.swift" "$top/swift" "$top/research/issue-3913"
fi
(cd LiteRT-LM/research/issue-3913/cases && shasum -a 256 -c "$HERE/cases.SHA256SUMS")
echo "runner source: $COMMIT (research/issue-3913, $(find LiteRT-LM/research/issue-3913 -type f | wc -l | tr -d ' ') files)"
build() {
  (cd LiteRT-LM/research/issue-3913/runner && swift --version && swift build -c release)
}
if [ -x "$QW" ]; then
  echo "building through quiet_wait.py (waits while another lane holds the Mac measurement window)"
  "$QW" --max-wait 21600 -- bash -c "$(declare -f build); cd '$W' && build" 2>&1 | tee "$W/logs/build_runner.out"
else
  build 2>&1 | tee "$W/logs/build_runner.out"
fi
R=$W/LiteRT-LM/research/issue-3913/runner/.build/release/runner
[ -x "$R" ] || { echo "runner binary missing: $R" >&2; exit 1; }
echo "runner: $R"
shasum -a 256 "$R"
find "$W/LiteRT-LM/research/issue-3913/runner/.build" -name 'libCLiteRTLM_mac.dylib' -path '*artifacts*' -exec shasum -a 256 {} \;
grep -n "checksum" "$W/LiteRT-LM/Package.swift" | head -2
