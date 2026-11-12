#!/usr/bin/env bash
# Shape-shift robustness run: 3 turns (generate on Modal -> CPU step 1 -> TPU v6e-1 -> ingest), VM deleted at the end.
set -u
R=/Users/hrudaynara/Projects/gemma-work/robust; W=/Users/hrudaynara/Projects/gemma-work; F=$W/tpu/fleet; O=$W/tpu/ops1
PYV=$W/venv/bin/python; PROJECT=project-dfbc8c9e-7988-418a-856; NAME=oracle-v6e1-robust-hruday; V=$F/robustvm
export MODAL_PROFILE=newacct
log() { echo "$(date -u +%H:%M:%SZ) robust $*"; }
mkdir -p $V; for f in grab_v6e.sh vm8.sh hunt_up.sh guard.sh; do sed "s/oracle-v6e1-a2m1-hruday/$NAME/" $F/a2mvm1/$f > $V/$f; done
grep -q "$NAME" $V/guard.sh || { log "TEMPLATE FAILED"; exit 1; }
for T in 0 1 2; do
  (cd $W && BUDGET_CAP=39 python3 budget_guard.py check newacct 2>&1 | grep -v MallocStack) || { log "Modal at cap; stopping"; break; }
  log "turn $T: generating"; (cd $R && $PYV robust.py gen 2>&1 | grep -vE MallocStack | tail -2) || { log "GEN/DIGEST FAILED"; exit 3; }
  (cd $R && $PYV robust.py step1 $T 2>&1 | tail -1)
  n=$(grep -c . $O/robust-shift-t$T.items.jsonl 2>/dev/null || echo 0)
  if [ "$n" -gt 0 ]; then
    if ! gcloud compute tpus tpu-vm list --project=$PROJECT --zone=- --format="value(name)" 2>/dev/null | grep -q $NAME; then
      log "hunting a v6e-1"; (cd $V && rm -f zone.txt created_at.txt && bash hunt_up.sh 2>&1 | grep -v MallocStack > up.log)
      grep -q UP_DONE $V/up.log || { log "TPU hunt failed; stopping"; break; }
      (cd $V && RATE=3.24 CAP_USD=8 IDLE_MIN=15 nohup bash guard.sh >> guard.out 2>&1 < /dev/null &)
      log "UP in $(cat $V/zone.txt)"
    fi
    job=robust-shift-t$T
    (cd $O && PROJECT=$PROJECT NAME=$NAME ZONE=$(cat $V/zone.txt) ACCEL=v6e-1 SHARDS=1 python3 -u crashloop.py $job $job.items.jsonl 2>&1 | grep -vE "MallocStack|^remote:" | tail -2)
    (cd $R && $PYV robust.py ingest $T 2>&1 | tail -1)
  fi
  log "TURN $T DONE"
done
(cd $V && ZONE=$(cat zone.txt 2>/dev/null) NAME=$NAME PROJECT=$PROJECT bash vm8.sh delete 2>&1 | tail -1; kill $(cat guard.pid 2>/dev/null) 2>/dev/null; rm -f zone.txt created_at.txt)
log "ROBUST DONE"; (cd $R && $PYV robust.py report)
