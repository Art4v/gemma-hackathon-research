#!/usr/bin/env bash
# Overnight pipeline for the NEW Kimi problems (e1 re-admits, e2 kbkd, e3 scale variants), Hruday's go of Nov 11:
#  1 wait for Kimi first attempts + CPU step 1 to finish      4 Kimi repair round 2 (cap: $200 minus everything spent on new problems)
#  2 TPU oracle on 8 v6e-1 VMs (kimi-n1)                       5 CPU step 1 on the repairs, TPU on 8 VMs (kimi-n2), mop-up
#  3 mop-up VMs for lost batches (kimi-n1mop), alongside 4     6 assemble v6
# Every VM: own guard (CAP 12 USD, IDLE 15 min), deleted when its round ends.   nohup bash pipeline_n.sh >> fleet.log &
set -u
SC=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad
F=$SC/tpu/fleet; K=$SC/kimi_pilot; NEWP=new/new_problems.jsonl,scale/scale_problems.jsonl
log() { echo "$(date -u +%H:%M:%SZ) pipeN $*"; }
gen_done() { for f in "$@"; do grep -qE "FULL GEN DONE|REPAIR GEN DONE|CAP REACHED" $f 2>/dev/null || return 1; done; return 0; }
fleet() {  # fleet R N JOBTAG VMTAG
  python3 $F/build_items_n.py $1 $2 $3 || { log "BUILD FAILED $3"; return 1; }
  for k in $(seq 1 $2); do
    [ -s $F/$3-s$k.items.jsonl ] || { log "$3 s$k empty, skipped"; continue; }
    (cd $F && nohup bash fleet_run_n.sh $k $3 $4 >> fleet.log 2>&1 < /dev/null &); sleep 4
  done
  log "LAUNCHED $3"
}
wait_fleet() { sleep 30; while pgrep -f "fleet_run_n.sh [0-9]+ $1 " >/dev/null; do sleep 60; done; log "fleet $1 finished"; }
count_items() { cat $F/$1-s*.items.jsonl 2>/dev/null | wc -l | tr -d ' '; }

log "waiting for first-attempt generation and CPU step 1"
until gen_done $K/out/new/gen_global_s0.log $K/out/new/gen_us_s1.log $K/out/new/gen_scale_global_s0.log $K/out/new/gen_scale_us_s1.log \
      && grep -q "CPU SCORING DONE" $K/out/new/score_cpu.log; do sleep 60; done
log "first attempts generated and step-1 scored; top-up pass for failed requests (throttling etc.), same per-shard ledgers and caps"
for sh in 0 1; do PR=$([ $sh = 0 ] && echo global || echo us)
  (cd $K && PROBLEMS=new/new_problems.jsonl OUTDIR=out/new TAG=new PROFILE=$PR SHARD=$sh NSHARDS=2 CAP_USD=48 WORKERS=4 venv/bin/python -u run_full.py > out/new/topup_new_${PR}_s$sh.log 2>&1)
  (cd $K && PROBLEMS=scale/scale_problems.jsonl OUTDIR=out/new TAG=scale PROFILE=$PR SHARD=$sh NSHARDS=2 CAP_USD=50 WORKERS=4 venv/bin/python -u run_full.py > out/new/topup_scale_${PR}_s$sh.log 2>&1)
done
(cd $K && OUT=out/new/cpu_scored.jsonl GEN_GLOB="$K/out/new/k31_*.jsonl" LOG_GLOB="$K/out/new/gen_*.log" PROBLEMS=$NEWP python3 -u score_cpu.py >> out/new/score_cpu.log 2>&1)
log "first attempts final: $(grep -h "to generate" $K/out/new/topup_*.log | tr '\n' ' ') | step-1 rows $(wc -l < $K/out/new/cpu_scored.jsonl)"
fleet 1 8 kimi-n1 n1; wait_fleet kimi-n1

python3 $F/build_items_n.py 1 3 kimi-n1mop
if [ "$(count_items kimi-n1mop)" -ge 5 ]; then fleet 1 3 kimi-n1mop n1m; else log "n1 mop-up: under 5 items, skipped"; fi

SPENT=$(cd $K && python3 -c "import glob,json; print(round(sum(json.loads(l)['cost_usd'] for f in glob.glob('out/new/ledger_*.jsonl') for l in open(f)),2))")
CAP=$(python3 -c "print(max(0, round((200 - $SPENT) / 2 - 2, 2)))")
log "new-problem Kimi spend so far \$$SPENT; repair round 2 cap \$$CAP per shard (x2)"
if python3 -c "import sys; sys.exit(0 if $CAP >= 5 else 1)"; then
  mkdir -p $K/out/new/rep2
  for sh in 0 1; do PR=$([ $sh = 0 ] && echo global || echo us)
    (cd $K && ROUND=2 PROFILE=$PR SHARD=$sh NSHARDS=2 CAP_USD=$CAP WORKERS=12 nohup venv/bin/python -u run_repair_n.py > out/new/rep2/gen_${PR}_s$sh.log 2>&1 < /dev/null &)
  done
  sleep 20
  (cd $K && OUT=out/new/rep2/cpu_scored.jsonl GEN_GLOB="$K/out/new/rep2/k31_rep2_*.jsonl" LOG_GLOB="$K/out/new/rep2/gen_*.log" PROBLEMS=$NEWP \
     nohup python3 -u score_cpu.py > out/new/rep2/score_cpu.log 2>&1 < /dev/null &)
  log "repair round 2 generating: $(head -1 $K/out/new/rep2/gen_global_s0.log) | $(head -1 $K/out/new/rep2/gen_us_s1.log)"
  until gen_done $K/out/new/rep2/gen_global_s0.log $K/out/new/rep2/gen_us_s1.log && grep -q "CPU SCORING DONE" $K/out/new/rep2/score_cpu.log; do sleep 60; done
  log "repairs generated and step-1 scored: $(wc -l < $K/out/new/rep2/cpu_scored.jsonl) rows; $(tail -1 $K/out/new/rep2/gen_global_s0.log) | $(tail -1 $K/out/new/rep2/gen_us_s1.log)"
  wait_fleet kimi-n1mop
  fleet 2 8 kimi-n2 n2; wait_fleet kimi-n2
  python3 $F/build_items_n.py 2 3 kimi-n2mop
  if [ "$(count_items kimi-n2mop)" -ge 5 ]; then fleet 2 3 kimi-n2mop n2m; wait_fleet kimi-n2mop; else log "n2 mop-up: under 5 items, skipped"; fi
else
  log "repair round 2 skipped: cap too small"; wait_fleet kimi-n1mop
fi
(cd $K && python3 assemble_v6.py > v6/assemble.log 2>&1)
log "PIPELINE N DONE: $(head -1 $K/v6/assemble.log | cut -c1-200)"
