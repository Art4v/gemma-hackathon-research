#!/usr/bin/env bash
# Approved by Hruday (Nov 11 ~04:37Z): start TPU round 2 when Kimi generation and CPU scoring are both finished.
SC=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad
K=$SC/kimi_pilot/out; F=$SC/tpu/fleet
log() { echo "$(date -u +%H:%M:%SZ) round2 $*"; }
log "waiting for generation (both shards) and CPU scoring to finish"
until grep -qE "FULL GEN DONE|CAP REACHED" $K/full_global_s0.log && grep -qE "FULL GEN DONE|CAP REACHED" $K/full_us_s1.log && grep -q "CPU SCORING DONE" $K/score_cpu.log; do sleep 60; done
log "generation and scoring done; building items"
python3 $F/build_round2.py || { log "BUILD FAILED"; exit 1; }
for k in 1 2 3 4 5 6 7 8; do
  [ -s $F/kimi-k31-r2-s$k.items.jsonl ] || { log "s$k empty, skipped"; continue; }
  (cd $F && nohup bash fleet_run_r2.sh $k kimi-k31-r2 >> fleet.log 2>&1 < /dev/null &); sleep 4
done
log "LAUNCHED 8 round-2 VMs"
