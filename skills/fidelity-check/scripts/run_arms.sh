#!/bin/bash
# run_arms.sh <work dir> <label> <arm>... — run the #3913 copy-fidelity cases, one runner process per arm.
#
#   arm = <cases>:<backend>:<act>    cases ∈ {long3, short, baseline, long, cross4096, xl}, backend ∈ {gpu, cpu},
#                                    act ∈ {fp16, fp32}  (fp16 = the runtime's default activation type; on the GPU
#                                    that is fp16 for text models; on the cpu the engine computes in fp32 — label it fp32)
#   label = <model>-<file>, e.g. E2B-ropecomposite; the bundle must be <work dir>/models/<label>.litertlm
#           (a symlink is fine: the runner writes its cache into <work dir>/models/cache)
# Output: <work dir>/raw/<cases>__<label>__<backend>-<act>.jsonl (one JSON line per case, the runner's stdout) and
#         <work dir>/logs/<same>.log (the runner's stderr = the engine log). Existing complete records are skipped.
# All arms of one call run inside ONE Mac measurement window (quiet_hold.py, when that tool exists) — keep a call
# at <= 15 min of work. Watchdog 900 s per arm. Window times are appended to <work dir>/logs/holds.txt.
set -uo pipefail
W=${1:?usage: run_arms.sh <work dir> <label> <cases>:<backend>:<act>...}; shift
LABEL=${1:?label}; shift
HOLD=$HOME/code/standup/tools/quiet/quiet_hold.py
if [ "${FIDELITY_INNER:-}" != "1" ] && [ -x "$HOLD" ]; then
  FIDELITY_INNER=1 exec "$HOLD" "fidelity-check-$LABEL" -- "$0" "$W" "$LABEL" "$@"
fi
RUNNER=$W/LiteRT-LM/research/issue-3913/runner/.build/release/runner
CASES=$W/LiteRT-LM/research/issue-3913/cases
MODEL=$W/models/$LABEL.litertlm
TIMEOUT=${FIDELITY_TIMEOUT:-900}
mkdir -p "$W/raw" "$W/logs"
[ -x "$RUNNER" ] || { echo "runner missing: $RUNNER (run fetch_runner.sh first)" >&2; exit 2; }
[ -e "$MODEL" ] || { echo "bundle missing: $MODEL (symlink the bundle there)" >&2; exit 2; }
echo "$(date '+%F %T') taken  $LABEL: $*" >> "$W/logs/holds.txt"
rc_all=0
for arm in "$@"; do
  IFS=: read -r cases backend act <<< "$arm"
  cj=$CASES/$cases.json
  [ -f "$cj" ] || { echo "no such case set: $cases"; rc_all=2; continue; }
  name=${cases}__${LABEL}__${backend}-${act}
  out=$W/raw/$name.jsonl; log=$W/logs/$name.log
  want=$(grep -o '"id"' "$cj" | wc -l | tr -d " ")
  if [ -s "$out" ] && [ "$(wc -l < "$out" | tr -d ' ')" = "$want" ]; then echo "skip $name (complete)"; continue; fi
  rm -f "$out.partial"
  t0=$(date +%s)
  echo "-- $(date '+%T') $name ($want cases)"
  "$RUNNER" "$MODEL" "$backend" "$act" "$cj" > "$out.partial" 2> "$log" &
  pid=$!
  # the watchdog gets no stdio: an orphaned `sleep` holding the caller's pipe open made `run_arms.sh | tail` hang 15 min (blind run 2026-10-18)
  ( sleep "$TIMEOUT"; if kill -0 $pid 2>/dev/null; then echo "WATCHDOG: killed runner pid $pid after $TIMEOUT s" >> "$log"; kill $pid; fi ) >/dev/null 2>&1 </dev/null &
  wd=$!
  wait $pid; rc=$?
  pkill -P $wd 2>/dev/null; kill $wd 2>/dev/null; wait $wd 2>/dev/null   # the sleep child too, by parent pid (never by name)
  got=$(wc -l < "$out.partial" | tr -d ' ')
  if [ $rc -eq 0 ] && [ "$got" = "$want" ]; then
    mv "$out.partial" "$out"
    echo "-- $(date '+%T') $name ok: $got lines in $(( $(date +%s) - t0 )) s, exact $(grep -c '"exact":true' "$out")/$got"
  else
    mv "$out.partial" "$out.failed-$(date '+%H%M%S')"
    echo "-- $(date '+%T') $name FAILED rc=$rc lines=$got/$want (see $log)"
    rc_all=1
  fi
done
echo "$(date '+%F %T') released $LABEL rc=$rc_all" >> "$W/logs/holds.txt"
exit $rc_all
