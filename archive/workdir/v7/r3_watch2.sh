#!/usr/bin/env bash
# Round-3 completion: whenever no run_r3.py is running, count what is still missing; top it up on the global profile (us is throttled)
# at 4 salted shards x 6 workers, at most 3 more passes and within TOTAL_CAP across all r3 ledgers; then touch out/r3/GEN_FINAL.
V=/Users/hrudaynara/Projects/gemma-work/v7; PYV=/Users/hrudaynara/Projects/gemma-work/venv/bin/python; TOTAL_CAP=${TOTAL_CAP:-120}
log() { echo "$(date -u +%H:%M:%SZ) r3watch2 $*"; }
cd $V
for t in 2 3 4 5; do
  sleep 20; while pgrep -f "run_r3.py" >/dev/null; do sleep 30; done
  read REM SPENT <<< "$($PYV - <<'PY'
import json, glob
gap = json.load(open("r3/gap.json")); R = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}
want = {p for p, g in gap.items() if g["chain"] and g["chain"][-1][1] in R}
done = {json.loads(l)["problem_id"] for f in glob.glob("out/r3/k31_r3_*.jsonl") for l in open(f) if json.loads(l).get("ok")}
spent = sum(json.loads(l)["cost_usd"] for f in glob.glob("out/r3/ledger_*.jsonl") for l in open(f))
print(len(want - done), round(spent, 2))
PY
)"
  log "remaining $REM, round-3 spend \$$SPENT"
  [ "$REM" -eq 0 ] && break
  [ "$t" -eq 5 ] && { log "stopping after 3 top-up passes with $REM left"; break; }
  PER=$($PYV -c "print(round(max(0.0, $TOTAL_CAP - $SPENT) / 4, 2))")
  $PYV -c "import sys; sys.exit(0 if $PER >= 2 else 1)" || { log "ceiling reached (\$$SPENT of \$$TOTAL_CAP)"; break; }
  log "top-up $t: 4 global shards x 6 workers, cap \$$PER each"
  for sh in 0 1 2 3; do PROFILE=global SHARD=$sh NSHARDS=4 SALT=t$t CAP_USD=$PER WORKERS=6 TAG=_t${t}_s$sh \
    nohup $PYV -u run_r3.py > out/r3_top${t}_s$sh.log 2>&1 < /dev/null & done
done
touch out/r3/GEN_FINAL; log "GEN_FINAL"
