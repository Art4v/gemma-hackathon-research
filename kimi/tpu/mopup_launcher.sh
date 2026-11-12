#!/usr/bin/env bash
# Approved by Hruday (Nov 11 ~06:10Z): after round 2 ends, one mop-up pass over every unjudged Kimi step-1 survivor, 3 VMs, same caps.
F=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/tpu/fleet
log() { echo "$(date -u +%H:%M:%SZ) mopup $*"; }
log "waiting for every round-2 crashloop to end and every r2 VM to be deleted"
until ! pgrep -f "crashloop.py kimi-k31-r2" >/dev/null && ! pgrep -f "fleet_run_r2.sh" >/dev/null; do sleep 60; done
log "round 2 finished; building mop-up items"
python3 $F/build_mopup.py 3 || { log "BUILD FAILED"; exit 1; }
for k in 1 2 3; do
  [ -s $F/kimi-k31-mop-s$k.items.jsonl ] || { log "s$k empty, skipped"; continue; }
  (cd $F && nohup bash fleet_run_mop.sh $k kimi-k31-mop >> fleet.log 2>&1 < /dev/null &); sleep 4
done
log "LAUNCHED mop-up VMs"
