#!/usr/bin/env bash
# bash chain_vm.sh WAITPID VMDIR JOB [JOB...]: wait for WAITPID (a running crashloop on this VM) to end, then run each JOB's items on the same
# VM (already bootstrapped), then delete the VM and stop its guard. NAME and ZONE come from VMDIR's guard.sh / zone.txt.
set -u
P=$1; D=$2; shift 2
F=/Users/hrudaynara/Projects/gemma-work/tpu/fleet; O=/Users/hrudaynara/Projects/gemma-work/tpu/ops1; PROJECT=project-dfbc8c9e-7988-418a-856
NAME=$(grep -o 'oracle-v6e1-[a-z0-9]*-hruday' $F/$D/guard.sh | head -1); Z=$(cat $F/$D/zone.txt)
log() { echo "$(date -u +%H:%M:%SZ) chain-$D $*"; }
while kill -0 $P 2>/dev/null; do sleep 15; done
log "previous job ended on $NAME ($Z)"
for JOB in "$@"; do
  cp $F/$JOB.items.jsonl $O/; log "running $JOB ($(wc -l < $F/$JOB.items.jsonl) items)"
  (cd $O && PROJECT=$PROJECT NAME=$NAME ZONE=$Z ACCEL=v6e-1 SHARDS=1 python3 -u crashloop.py $JOB $JOB.items.jsonl > $F/crashloop-$JOB.log 2>&1)
  log "$JOB ended: $(grep -vE 'MallocStack|^remote:' $F/crashloop-$JOB.log | tail -1)"
done
(cd $F/$D && ZONE=$Z NAME=$NAME PROJECT=$PROJECT bash vm8.sh delete 2>&1 | tail -1)
rm -f $F/$D/zone.txt $F/$D/created_at.txt; kill $(cat $F/$D/guard.pid 2>/dev/null) 2>/dev/null
log "VM DELETED"
