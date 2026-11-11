#!/usr/bin/env bash
# One fleet VM K: start its guard (RATE 3.24 = the dearest v6e zone, CAP_USD 12, IDLE_MIN 15), hunt a v6e-1 (on-demand), run the
# oracle on items K through crashloop.py (crash recovery, results under ops1/repo/oracle/jobs/<job>-p*/results), then delete the VM
# as soon as the round ends, without waiting for the guard.   bash fleet_run.sh K JOBBASE
set -u
K=$1; JOB=${2:-kimi-k31-r1}-s$K
SC=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad
F=$SC/tpu/fleet; D=$F/mopvm$K; O=$SC/tpu/ops1
PROJECT=project-dfbc8c9e-7988-418a-856; NAME=oracle-v6e1-mop$K-hruday
log() { echo "$(date -u +%H:%M:%SZ) mopvm$K $*"; }
(cd $D && RATE=3.24 CAP_USD=12 IDLE_MIN=15 nohup bash guard.sh >> guard.out 2>&1 < /dev/null &)
log "guard started; hunting $NAME"
(cd $D && bash hunt_up.sh > up.log 2>&1)
grep -q UP_DONE $D/up.log || { log "HUNT FAILED: $(tail -1 $D/hunt.log)"; kill $(cat $D/guard.pid 2>/dev/null) 2>/dev/null; exit 1; }
Z=$(cat $D/zone.txt); log "UP in $Z"
cp $F/$JOB.items.jsonl $O/
(cd $O && PROJECT=$PROJECT NAME=$NAME ZONE=$Z ACCEL=v6e-1 SHARDS=1 python3 -u crashloop.py $JOB $JOB.items.jsonl > $F/crashloop-$JOB.log 2>&1)
log "crashloop ended: $(grep -vE 'MallocStack|^remote:' $F/crashloop-$JOB.log | tail -1)"
(cd $D && ZONE=$Z NAME=$NAME PROJECT=$PROJECT bash vm8.sh delete 2>&1 | tail -1)
rm -f $D/zone.txt $D/created_at.txt; kill $(cat $D/guard.pid 2>/dev/null) 2>/dev/null
log "VM DELETED"
