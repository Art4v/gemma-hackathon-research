#!/usr/bin/env bash
# Approved scope (repair rounds incl. their TPU verification): after every repair-round-2 VM finishes, judge the unjudged survivors.
F=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/tpu/fleet
log() { echo "$(date -u +%H:%M:%SZ) a2mop $*"; }
log "waiting for every repair-round-2 VM to finish"
until ! pgrep -f "fleet_run_a2.sh" >/dev/null && ! pgrep -f "crashloop.py kimi-a2-" >/dev/null; do sleep 60; done
python3 $F/build_mopup_rep.py 2 3 || { log "BUILD FAILED"; exit 1; }
for k in 1 2 3; do
  [ -s $F/kimi-a2mop-s$k.items.jsonl ] || { log "s$k empty, skipped"; continue; }
  (cd $F && nohup bash fleet_run_a2mop.sh $k kimi-a2mop >> fleet.log 2>&1 < /dev/null &); sleep 4
done
log "LAUNCHED"
