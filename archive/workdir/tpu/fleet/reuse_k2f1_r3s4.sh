#!/usr/bin/env bash
# Reuse oracle-v6e1-k2f1-hruday (us-east5-a, already bootstrapped) for round-3 group 4 after its kimi-kn2-s1 job ends, then delete it.
# The VM's guard (k2vm1, $12 cap, idle 15 min) stays on as the safety net until the delete.
set -u
F=/Users/hrudaynara/Projects/gemma-work/tpu/fleet; O=/Users/hrudaynara/Projects/gemma-work/tpu/ops1
PROJECT=project-dfbc8c9e-7988-418a-856; NAME=oracle-v6e1-k2f1-hruday; Z=us-east5-a; JOB=kimi-r3-s4
log() { echo "$(date -u +%H:%M:%SZ) reuse-k2f1 $*"; }
while kill -0 43981 2>/dev/null; do sleep 15; done
log "kimi-kn2-s1 ended: $(grep -vE 'MallocStack|^remote:' $F/crashloop-kimi-kn2-s1.log | tail -1)"
cp $F/$JOB.items.jsonl $O/
log "running $JOB on $NAME"
(cd $O && PROJECT=$PROJECT NAME=$NAME ZONE=$Z ACCEL=v6e-1 SHARDS=1 python3 -u crashloop.py $JOB $JOB.items.jsonl > $F/crashloop-$JOB.log 2>&1)
log "crashloop ended: $(grep -vE 'MallocStack|^remote:' $F/crashloop-$JOB.log | tail -1)"
(cd $F/k2vm1 && ZONE=$Z NAME=$NAME PROJECT=$PROJECT bash vm8.sh delete 2>&1 | tail -1)
rm -f $F/k2vm1/zone.txt $F/k2vm1/created_at.txt; kill $(cat $F/k2vm1/guard.pid 2>/dev/null) 31195 2>/dev/null
log "VM DELETED"
