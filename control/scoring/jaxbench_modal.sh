#!/usr/bin/env bash
# JAXBench lora arm per control/PROTOCOL.md (bare prompt, greedy, 3 turns), generalized from the v4/v5 runner (Nov 11) for any adapter, generated on the SAME
# backend as the base bare/docs and v1 arms: Modal A10G, Ollama 0.40.14 (control/modal_gemma.py, app gemma-bench, profile newacct).
# Guards: EXPECT_DIGEST is the digest the v4 model got in the Modal volume (its model layer verified = the v4 GGUF sha); after every
# generate, EVERY row must carry that digest and model name, else the run stops. Modal spend is checked before every turn (cap $27).
set -u
# Nov 12 pre-flight (Karthik's audit 6813236025): digest must be a full, new digest; fresh run dirs unless RESUME=1; every step's exit code checked; row counts per turn.
SC=${SC:-/Users/hrudaynara/Projects/gemma-work}
ROOT=/Users/hrudaynara/Projects/gemma-hackathon-research
PY=/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python
MODEL=${MODEL:?set MODEL, e.g. gemma4-e4b-lora:v7}; PREFIX=${PREFIX:?set PREFIX, e.g. lora-v7-modal}; WORKERS=${WORKERS:-6}
: "${EXPECT_DIGEST:?set EXPECT_DIGEST to the new model's digest, from the create script's output}"
echo "$EXPECT_DIGEST" | grep -Eq '^[0-9a-f]{12,}$' || { echo "EXPECT_DIGEST must be >= 12 hex chars"; exit 2; }
for old in ee6656371218 0f764d3f6798 641ddde330ed 2a99a2f97822; do case "$EXPECT_DIGEST" in $old*) echo "EXPECT_DIGEST $EXPECT_DIGEST is an OLD model (base/v4/v5/mac)"; exit 2;; esac; done
BASE_DIGEST=ee6656371218
export MODAL_PROFILE=${MODAL_PROFILE:-newacct}; BUDGET_CAP=${BUDGET_CAP:-27}; GCP_PROJECT=${GCP_PROJECT:-project-dfbc8c9e-7988-418a-856}
S=$SC/tpu; O=$S/ops1; V=$S/fleet/evalvm1; mkdir -p $V; for f in grab_v6e.sh vm8.sh hunt_up.sh guard.sh; do sed "s/oracle-v6e1-a2m1-hruday/oracle-v6e1-evalf1-hruday/; s/project-dfbc8c9e-7988-418a-856/$GCP_PROJECT/" $S/fleet/a2mvm1/$f > $V/$f; done
RUN=$ROOT/control/runs/$(echo "$MODEL" | sed 's/[^A-Za-z0-9._-]/_/g')__modal
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
check_rows() {  # every row: served by MODEL with EXPECT_DIGEST, never the base digest
  $PY - "$RUN/bare.jsonl" "$MODEL" "$EXPECT_DIGEST" "$BASE_DIGEST" <<'EOF'
import json, sys
p, model, want, base = sys.argv[1:]
rows = [json.loads(l) for l in open(p) if l.strip()]
bad = [r["id"] if "id" in r else (r["task"], r["turn"]) for r in rows
       if (r.get("backend") or {}).get("model") != model or not str((r.get("backend") or {}).get("digest", "")).startswith(want)
       or str((r.get("backend") or {}).get("digest", "")).startswith(base)]
print(f"rows {len(rows)}, wrong model/digest {len(bad)}", bad[:5])
sys.exit(1 if bad or not rows else 0)
EOF
}
cd "$ROOT"
if [ "${RESUME:-0}" != 1 ]; then  # fresh run: no leftovers that bench.py would resume or the merge would pick up
  left=$(ls -d "$RUN" control/runs/cpu_checks/${PREFIX}-t* "$O"/repo/oracle/jobs/control-${PREFIX}-t* 2>/dev/null | head -3)
  [ -n "$left" ] && { log "NOT FRESH: $left (set RESUME=1 to continue a run deliberately)"; exit 4; }
fi
for turn in 0 1 2; do
  (cd $SC && BUDGET_CAP=$BUDGET_CAP python3 budget_guard.py check $MODAL_PROFILE 2>&1 | grep -v MallocStack) || { log "Modal at cap; stopping"; break; }
  log "turn $turn: generating on Modal ($WORKERS workers)"
  $PY control/bench.py --backend modal --model "$MODEL" generate --arm bare --workers "$WORKERS" > "$SC/scoring_kit/gen_${PREFIX}_t$turn.log" 2>&1
  rc=$?; grep -vE MallocStack "$SC/scoring_kit/gen_${PREFIX}_t$turn.log" | tail -3; [ $rc -eq 0 ] || { log "GENERATE FAILED (rc $rc); stopping"; exit 5; }
  nt=$($PY -c "import json; print(sum(1 for l in open('$RUN/bare.jsonl') if l.strip() and json.loads(l).get('turn') == $turn))")
  log "turn $turn: $nt rows generated"; [ "$turn" != 0 ] || [ "$nt" -eq 64 ] || { log "TURN 0 HAS $nt ROWS, NOT 64; stopping"; exit 5; }
  check_rows || { log "DIGEST CHECK FAILED: not the v4 model; stopping"; exit 3; }
  tag="$PREFIX-t$turn"
  log "turn $turn: CPU step 1 ($tag)"
  $PY control/cpu_check.py --backend modal --model "$MODEL" --tag "$tag" --py "$PY" --ingest > "$SC/scoring_kit/cpu_${tag}.log" 2>&1
  rc=$?; grep -vE MallocStack "$SC/scoring_kit/cpu_${tag}.log" | tail -4; [ $rc -eq 0 ] || { log "CPU STEP 1 FAILED (rc $rc); stopping"; exit 6; }
  items=control/runs/cpu_checks/${tag}_tpu_items.jsonl
  n=0; [ -f "$items" ] && n=$(grep -c . "$items")
  log "turn $turn: $n attempt(s) passed step 1 -> TPU"
  if [ "$n" -gt 0 ]; then
    if ! gcloud compute tpus tpu-vm list --project=$GCP_PROJECT --zone=- --format="value(name)" 2>/dev/null | grep -q oracle-v6e1-evalf1-hruday; then
      log "hunting a v6e-1"; (cd $V && rm -f zone.txt created_at.txt && bash hunt_up.sh 2>&1 | grep -v MallocStack > up.log)
      grep -q UP_DONE $V/up.log || { log "TPU hunt failed; stopping after CPU step 1"; break; }
      (cd $V && RATE=3.24 CAP_USD=12 IDLE_MIN=15 nohup bash guard.sh >> guard.out 2>&1 < /dev/null &)
    fi
    job="control-$PREFIX-t$turn"
    cp "$items" "$O/$job.items.jsonl"
    (cd "$O" && export PROJECT=$GCP_PROJECT ACCEL=v6e-1 NAME=oracle-v6e1-evalf1-hruday ZONE=$(cat $V/zone.txt) SHARDS=1 && \
      python3 -u crashloop.py "$job" "$job.items.jsonl" 2>&1 | grep -vE "MallocStack|^remote:" | tail -3)
    # Nov 11: ingest each task's FINAL verdict once. Halt-cascade victims (harness_oom with "halted unexpectedly" / RuntimeUnexpectedCoreHalt)
    # are not verdicts: crashloop re-runs them in a later part, and bench.py treats harness_oom as terminal. Parts are read in numeric order.
    merged="$O/$job.final.jsonl"
    python3 - "$O/repo/oracle/jobs" "$job" "$merged" <<'PYEOF'
import glob, json, re, sys
root, job, out = sys.argv[1:]
parts = sorted(glob.glob(f"{root}/{job}-[px]*/results/*.jsonl"), key=lambda p: (p.split(f"{job}-")[1][0], int(re.search(r"-[px](\d+)", p).group(1))))
halt = lambda r: r.get("verdict") == "harness_oom" and any(h in (r.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))
meta, final = None, {}
for p in parts:
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("meta"): meta = meta or r; continue
        if halt(r) and r["id"] in final: continue
        if not halt(r) or r["id"] not in final: final[r["id"]] = r
open(out, "w").write((json.dumps(meta) + "\n" if meta else "") + "".join(json.dumps(r) + "\n" for r in final.values()))
print(f"{len(final)} final verdicts from {len(parts)} parts")
PYEOF
    $PY control/bench.py --backend modal --model "$MODEL" ingest "$merged" > "$SC/scoring_kit/ingest_${job}.log" 2>&1
    rc=$?; grep -vE MallocStack "$SC/scoring_kit/ingest_${job}.log" | tail -1; [ $rc -eq 0 ] || { log "INGEST FAILED (rc $rc); stopping"; exit 7; }
  fi
  log "TURN $turn DONE"
done
log "JAXBENCH DONE ($MODEL on Modal)"; (cd $V && ZONE=$(cat zone.txt 2>/dev/null) NAME=oracle-v6e1-evalf1-hruday PROJECT=$GCP_PROJECT bash vm8.sh delete 2>&1 | tail -1; kill $(cat guard.pid 2>/dev/null) 2>/dev/null; rm -f zone.txt created_at.txt)
$PY control/bench.py --backend modal --model "$MODEL" summary 2>/dev/null | tail -25 || true
# post-run: tasks with fewer than 3 turns must have reached a terminal verdict (a Modal error can leave one short silently; v5 needed a catch-up)
$PY -c "import json, collections; rows=[json.loads(l) for l in open('$RUN/bare.jsonl') if l.strip()]; t=collections.defaultdict(set)
[t[r['task']].add(r['turn']) for r in rows]; print(len(t), 'tasks; turns per task:', dict(collections.Counter(len(v) for v in t.values())), '(tasks with < 3 turns must have a terminal verdict; check bench summary)')"
