#!/bin/bash
# usage: queue.sh NAME ZONE JOB [JOB ...]   Runs oracle jobs one after another on one TPU VM and fetches each result.
# JOB prefixes: "wait:" the job is already running (just wait and fetch); "resume:" already on the VM (start without pushing).
set -u
NAME=$1; ZONE=$2; shift 2
SP=$SCRATCH
cd "$SP/wt/kbk" || exit 1
export PROJECT=gen-lang-client-0956375059 ACCEL=v6e-1 ZONE NAME
LOG="$SP/jobs/queue_r3a.log"
state() { gcloud compute tpus tpu-vm ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --command "pgrep -u oracle -f 'run.sh|tpu_oracle' >/dev/null && echo busy || echo idle" 2>/dev/null | grep -E "^(busy|idle)$" | tail -1; }
# two explicit "idle" answers in a row before the chip counts as free: a dropped ssh must never start a second job on a busy chip
wait_idle() { n=0; while [ $n -lt 2 ]; do s=$(state); if [ "$s" = idle ]; then n=$((n+1)); else n=0; fi; sleep 25; done; }
capped() { "$@" >/dev/null 2>&1 & p=$!; ( sleep "$CAP"; kill $p 2>/dev/null ) & w=$!; wait $p 2>/dev/null; kill $w 2>/dev/null; }
summ() { python3 -I -c "
import json,sys,collections
v=[json.loads(l) for l in open(sys.argv[1])]; v=[x for x in v if 'verdict' in x]
print(len(v), dict(collections.Counter(x['verdict'] for x in v)))" "oracle/jobs/$1/results/$1.jsonl" 2>&1 | tail -1; }
for spec in "$@"; do
  j=${spec#*:}; mode=${spec%%:*}; [ "$mode" = "$spec" ] && mode=new
  wait_idle_first=1
  if [ "$mode" != wait ]; then
    wait_idle
    # a pack that is still being built: wait up to 8 minutes for it, then skip the job instead of starting nothing
    if [ "$mode" = new ]; then t=0; while [ ! -d "$SP/jobs/$j" ] && [ $t -lt 32 ]; do sleep 15; t=$((t+1)); done
      [ -d "$SP/jobs/$j" ] || { echo "SKIP $NAME $j $(date -u +%H:%MZ) no pack" >> "$LOG"; continue; }; fi
    [ "$mode" = new ] && { CAP=180 capped bash oracle/vm.sh push "$SP/jobs/$j"; }
    CAP=20 capped bash oracle/vm.sh run "$j"
    echo "STARTED $NAME $j $(date -u +%H:%MZ)" >> "$LOG"; sleep 50
    [ "$(state)" = busy ] || echo "WARNING $NAME $j not running 50 s after start" >> "$LOG"
  fi
  wait_idle; CAP=150 capped bash oracle/vm.sh fetch "$j"
  echo "DONE $NAME $j $(date -u +%H:%MZ) $(summ "$j")" >> "$LOG"
done
echo "QUEUE EMPTY $NAME $(date -u +%H:%MZ)" >> "$LOG"
