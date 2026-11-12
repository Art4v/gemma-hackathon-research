#!/usr/bin/env bash
# One fleet VM for the NEW Kimi problems: bash fleet_run_n.sh K JOBBASE VMTAG   (e.g. 3 kimi-n1 n1  ->  VM oracle-v6e1-n1f3-hruday)
# Same steps as fleet_run_a2.sh: guard (RATE 3.24, CAP_USD 12, IDLE_MIN 15), hunt a v6e-1 (fixed template: bootstrap retried, oracle
# user verified, VM deleted if setup fails), crashloop.py, delete the VM as soon as the round ends.
set -u
K=$1; JOB=$2-s$K; T=$3
SC=/Users/hrudaynara/Projects/gemma-work
F=$SC/tpu/fleet; D=$F/${T}vm$K; O=$SC/tpu/ops1
PROJECT=project-dfbc8c9e-7988-418a-856; NAME=oracle-v6e1-${T}f$K-hruday
log() { echo "$(date -u +%H:%M:%SZ) ${T}vm$K $*"; }
mkdir -p $D
for f in grab_v6e.sh vm8.sh hunt_up.sh guard.sh; do sed "s/oracle-v6e1-a2m1-hruday/$NAME/" $F/a2mvm1/$f > $D/$f; done
grep -q "$NAME" $D/guard.sh && grep -q "$NAME" $D/hunt_up.sh || { log "TEMPLATE FAILED"; exit 1; }
(cd $D && RATE=3.24 CAP_USD=12 IDLE_MIN=15 nohup bash guard.sh >> guard.out 2>&1 < /dev/null &)
log "guard started; hunting $NAME"
(cd $D && bash hunt_up.sh > up.log 2>&1)
grep -q UP_DONE $D/up.log || { log "HUNT FAILED: $(tail -1 $D/hunt.log 2>/dev/null) $(tail -1 $D/up.log)"; kill $(cat $D/guard.pid 2>/dev/null) 2>/dev/null; exit 1; }
Z=$(cat $D/zone.txt); log "UP in $Z"
cp $F/$JOB.items.jsonl $O/
(cd $O && PROJECT=$PROJECT NAME=$NAME ZONE=$Z ACCEL=v6e-1 SHARDS=1 python3 -u crashloop.py $JOB $JOB.items.jsonl > $F/crashloop-$JOB.log 2>&1)
log "crashloop ended: $(grep -vE 'MallocStack|^remote:' $F/crashloop-$JOB.log | tail -1)"
(cd $D && ZONE=$Z NAME=$NAME PROJECT=$PROJECT bash vm8.sh delete 2>&1 | tail -1)
rm -f $D/zone.txt $D/created_at.txt; kill $(cat $D/guard.pid 2>/dev/null) 2>/dev/null
log "VM DELETED"
