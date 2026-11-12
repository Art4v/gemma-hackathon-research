#!/usr/bin/env bash
# v7 fixes: wait for Kimi -> CPU step 1 -> TPU verification on 2 VMs (kimi-kn1, fixed tooling, guards $12) -> done.
W=/Users/hrudaynara/Projects/gemma-work; F=$W/tpu/fleet
log() { echo "$(date -u +%H:%M:%SZ) fixpipe $*"; }
log "waiting for the Kimi fixes, then CPU step 1"
cd $W/v7 && nice -n 10 python3 -u fix_step1.py 2 > out/fix1/step1.log 2>&1
log "$(tail -1 $W/v7/out/fix1/step1.log)"
for k in 1 2; do [ -s $F/kimi-kn1-s$k.items.jsonl ] && (cd $F && nohup bash fleet_run_n.sh $k kimi-kn1 kf >> fleet.log 2>&1 < /dev/null &); sleep 4; done
log "LAUNCHED kimi-kn1 TPU verification"
sleep 30; while pgrep -f "fleet_run_n.sh [0-9]+ kimi-kn1 " >/dev/null; do sleep 60; done
log "FIX PIPELINE DONE"
