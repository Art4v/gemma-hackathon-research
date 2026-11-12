#!/usr/bin/env bash
# Wait for the first round-3 run (8 shards) to drain, then top up throttled/unsent requests at 24 concurrent (4 shards x 6 workers) until
# every target has an ok row or the round-3 ceiling (TOTAL_CAP, all r3 ledgers) is reached; then touch out/r3/GEN_FINAL (releases step 1 -> TPU).
V=/Users/hrudaynara/Projects/gemma-work/v7; PYV=/Users/hrudaynara/Projects/gemma-work/venv/bin/python; TOTAL_CAP=${TOTAL_CAP:-120}
log() { echo "$(date -u +%H:%M:%SZ) r3watch $*"; }
cd $V
while [ "$(grep -l 'R3 GEN DONE' out/r3_gen_s*.log 2>/dev/null | wc -l)" -lt 8 ]; do sleep 30; done
for t in 1 2 3 4; do
  read REM SPENT <<< "$($PYV - <<'PY'
import json, glob, zlib
gap = json.load(open("r3/gap.json")); R = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}
want = {p for p, g in gap.items() if g["chain"] and g["chain"][-1][1] in R}
done = {json.loads(l)["problem_id"] for f in glob.glob("out/r3/k31_r3_*.jsonl") for l in open(f) if json.loads(l).get("ok")}
spent = sum(json.loads(l)["cost_usd"] for f in glob.glob("out/r3/ledger_*.jsonl") for l in open(f))
print(len(want - done), round(spent, 2))
PY
)"
  log "after pass $((t-1)): remaining $REM, round-3 spend \$$SPENT"
  [ "$REM" -eq 0 ] && break
  LEFT=$($PYV -c "print(max(0.0, $TOTAL_CAP - $SPENT))"); PER=$($PYV -c "print(round($LEFT / 4, 2))")
  $PYV -c "import sys; sys.exit(0 if $PER >= 2 else 1)" || { log "ceiling reached (\$$SPENT of \$$TOTAL_CAP); stopping top-ups"; break; }
  log "top-up $t: 4 shards x 6 workers, cap \$$PER each"
  for sh in 0 1 2 3; do PR=$([ $((sh%2)) = 0 ] && echo global || echo us); PROFILE=$PR SHARD=$sh NSHARDS=4 CAP_USD=$PER WORKERS=6 TAG=_t$t \
    nohup $PYV -u run_r3.py > out/r3_top${t}_s$sh.log 2>&1 < /dev/null & done
  sleep 20
  while [ "$(grep -l 'R3 GEN DONE' out/r3_top${t}_s*.log 2>/dev/null | wc -l)" -lt 4 ]; do sleep 30; done
done
touch out/r3/GEN_FINAL; log "GEN_FINAL"
