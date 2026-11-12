#!/usr/bin/env bash
# bash v7r_pipeline.sh SET N JOBBASE VMTAG   e.g. fix2 2 kimi-kn2 k2 | r3 6 kimi-r3 r3
# CPU step 1 (waits for generation) -> TPU on N v6e-1 VMs (fleet_run_n.sh: guard $12 each, VM deleted when the round ends).
SET=$1; N=$2; JOB=$3; T=$4; OURS=${5:-$(seq -s, 1 $N)}
W=/Users/hrudaynara/Projects/gemma-work; F=$W/tpu/fleet
log() { echo "$(date -u +%H:%M:%SZ) v7r-$SET $*"; }
log "waiting for generation, then CPU step 1"
cd $W/v7 && nice -n 10 python3 -u step1_v7r.py $SET $N > out/$SET/step1.log 2>&1
log "$(tail -1 $W/v7/out/$SET/step1.log)"
for k in ${OURS//,/ }; do [ -s $F/$JOB-s$k.items.jsonl ] && (cd $F && nohup bash fleet_run_n.sh $k $JOB $T >> fleet-$JOB.log 2>&1 < /dev/null &); sleep 5; done
log "LAUNCHED $JOB TPU verification on our fleet: groups $OURS of $N"
sleep 30; while pgrep -f "fleet_run_n.sh [0-9]+ $JOB " >/dev/null; do sleep 60; done
log "PIPELINE DONE $SET"
