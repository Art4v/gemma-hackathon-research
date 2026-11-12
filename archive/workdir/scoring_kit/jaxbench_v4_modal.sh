#!/usr/bin/env bash
# JAXBench lora arm per control/PROTOCOL.md (bare prompt, greedy, 3 turns), for v4 (gemma4-e4b-lora:v4final-e2) generated on the SAME
# backend as the base bare/docs and v1 arms: Modal A10G, Ollama 0.40.14 (control/modal_gemma.py, app gemma-bench, profile newacct).
# Guards: EXPECT_DIGEST is the digest the v4 model got in the Modal volume (its model layer verified = the v4 GGUF sha); after every
# generate, EVERY row must carry that digest and model name, else the run stops. Modal spend is checked before every turn (cap $27).
set -u
SC=/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad
ROOT=/Users/hrudaynara/Projects/gemma-hackathon-research
PY=/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python
MODEL=${MODEL:-gemma4-e4b-lora:v4final-e2}; PREFIX=${PREFIX:-lora-v4final-e2-modal}; WORKERS=${WORKERS:-6}
: "${EXPECT_DIGEST:?set EXPECT_DIGEST to the v4 digest in the Modal volume}"
BASE_DIGEST=ee6656371218
export MODAL_PROFILE=newacct
S=$SC/tpu; O=$S/ops1
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
for turn in 0 1 2; do
  (cd $SC && BUDGET_CAP=27 python3 budget_guard.py check newacct 2>&1 | grep -v MallocStack) || { log "Modal at cap; stopping"; break; }
  log "turn $turn: generating on Modal ($WORKERS workers)"
  $PY control/bench.py --backend modal --model "$MODEL" generate --arm bare --workers "$WORKERS" 2>&1 | grep -vE MallocStack | tail -3
  check_rows || { log "DIGEST CHECK FAILED: not the v4 model; stopping"; exit 3; }
  tag="$PREFIX-t$turn"
  log "turn $turn: CPU step 1 ($tag)"
  $PY control/cpu_check.py --backend modal --model "$MODEL" --tag "$tag" --py "$PY" --ingest 2>&1 | grep -vE MallocStack | tail -4
  items=control/runs/cpu_checks/${tag}_tpu_items.jsonl
  n=0; [ -f "$items" ] && n=$(grep -c . "$items")
  log "turn $turn: $n attempt(s) passed step 1 -> TPU"
  if [ "$n" -gt 0 ]; then
    if ! gcloud compute tpus tpu-vm list --project=project-dfbc8c9e-7988-418a-856 --zone=- --format="value(name)" 2>/dev/null | grep -q oracle-v6e1-hruday; then
      log "hunting a v6e-1"; (cd $S/v1 && rm -f zone.txt created_at.txt && bash hunt_up.sh 2>&1 | grep -v MallocStack > up.log)
      grep -q UP_DONE $S/v1/up.log || { log "TPU hunt failed; stopping after CPU step 1"; break; }
      (cd $S/v1 && nohup bash guard.sh >> guard.out 2>&1 < /dev/null &)
    fi
    job="control-$PREFIX-t$turn"
    cp "$items" "$O/$job.items.jsonl"
    (cd "$O" && export PROJECT=project-dfbc8c9e-7988-418a-856 ACCEL=v6e-1 NAME=oracle-v6e1-hruday ZONE=$(cat $S/v1/zone.txt) SHARDS=1 && \
      python3 -u crashloop.py "$job" "$job.items.jsonl" 2>&1 | grep -vE "MallocStack|^remote:" | tail -3)
    for r in "$O"/repo/oracle/jobs/"$job"-p*/results/*.jsonl; do
      [ -f "$r" ] && $PY control/bench.py --backend modal --model "$MODEL" ingest "$r" 2>&1 | grep -vE MallocStack | tail -1
    done
  fi
  log "TURN $turn DONE"
done
log "JAXBENCH DONE ($MODEL on Modal)"
$PY control/bench.py --backend modal --model "$MODEL" summary 2>/dev/null | tail -25 || true
