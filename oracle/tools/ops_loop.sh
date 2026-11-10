#!/usr/bin/env bash
# Operator loop: run oracle jobs back to back on the v6e, publish each one's results as soon as it finishes.
#   bash ops_loop.sh JOB [JOB ...]     (each JOB already packed in repo/oracle/jobs/ and pushed to the VM; the first may be running)
# Per job: wait until run.sh stops -> fetch -> collect.py -> commit oracle/results/<job>/ on a branch from origin/${BASE:-dataset-gen} ->
# push oracle-results/<job> -> start the next job. A job that stops short (fewer rows than items: a compiler crash kills the oracle
# process) is published as-is and the loop STOPS, so the crashing item can be split out by hand.
# ponytail: sequential, one chip; add a shard-aware variant only if we move to a v6e-8.
set -u
D=$(cd "$(dirname "$0")" && pwd); R=$D/repo
# RUNNER: oracle/vm.sh (a GCP TPU VM; needs ZONE) or oracle/colab.sh (a Colab session; NAME = the session name)
RUNNER=${RUNNER:-oracle/vm.sh}
export PROJECT=${PROJECT:?set PROJECT: no default, so a run can never touch the wrong account} ZONE=${ZONE:-} NAME=${NAME:-oracle-v6e}
[ "$RUNNER" = oracle/vm.sh ] && [ -z "$ZONE" ] && { echo "set ZONE for oracle/vm.sh"; exit 1; }
cd "$R"

status() { bash $RUNNER status "$1" 2>&1 | grep -vE "^(SSH|WARNING|Propagating|Using|Existing)"; }
# vm.sh prints "running" on its own line; colab.sh prints "rows written: N; running" on one line
is_running() { echo "$1" | grep -qE "(^|; )running"; }
on_vm_kill() {
  [ "$RUNNER" = oracle/vm.sh ] || { echo "$(date +%T) $j: all rows written, still running on Colab; fetching anyway"; return; }
  gcloud compute tpus tpu-vm ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command='sudo pkill -u oracle -f run.sh; sudo pkill -u oracle -f tpu_oracle.py' >/dev/null 2>&1; echo "$(date +%T) KILLED hung oracle process for $j (all rows written)"; }
items() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['n_items'])" "oracle/jobs/$1/MANIFEST.json"; }

publish() {  # $1 job
  local j=$1 wt; wt=$(mktemp -d)
  git fetch -q origin "+refs/heads/${BASE:-dataset-gen}:refs/remotes/origin/${BASE:-dataset-gen}"
  git worktree add -q --detach "$wt" origin/${BASE:-dataset-gen}
  mkdir -p "$wt/oracle/results/$j"
  cp oracle/jobs/$j/results/$j.jsonl oracle/jobs/$j/MANIFEST.json oracle/jobs/$j/summary.json "$wt/oracle/results/$j/" 2>/dev/null
  cp oracle/jobs/$j/run.log "$wt/oracle/results/$j/" 2>/dev/null
  local n; n=$(grep -c . "$wt/oracle/results/$j/$j.jsonl" 2>/dev/null || echo 0)
  ( cd "$wt" && git add "oracle/results/$j" && git -c user.name=LordKarV -c user.email=kkvattem@gmail.com commit -q \
      -m "oracle results: $j ($n rows, TPU ${ACCEL:-v6e-1} $ZONE, project $PROJECT)" && git push -q origin "HEAD:refs/heads/oracle-results/$j" )
  local rc=$?
  git worktree remove --force "$wt"
  return $rc
}

for j in "$@"; do
  s=$(status "$j")
  if echo "$s" | grep -q "not running" && echo "$s" | grep -q "rows written: 0"; then
    echo "$(date +%T) START $j"; bash $RUNNER run "$j" >/dev/null 2>&1 &
    # run.sh's file and JAX checks take a while before the oracle process exists: wait for it, or give up after 5 min
    for _ in $(seq 30); do sleep 10; s=$(status "$j"); is_running "$s" && break; echo "$s" | grep -qE "rows written: [1-9]" && break; done
    { is_running "$s" || echo "$s" | grep -qE "rows written: [1-9]"; } || { echo "$(date +%T) NOSTART $j: never started, stopping"; exit 3; }
  fi
  n=$(items "$j")
  # hang guard: the oracle process sometimes hangs on exit after its last row (Nov 9, ~10 min idle); kill it once every row is written
  while s=$(status "$j"); is_running "$s"; do
    echo "$s" | grep -qE "rows written: $n(;|\$)" && { sleep 60; is_running "$(status "$j")" && on_vm_kill; break; }
    sleep 60
  done
  bash $RUNNER fetch "$j" >/dev/null 2>&1
  python3 -I -B oracle/collect.py "oracle/jobs/$j" > "oracle/jobs/$j/collect.out" 2>&1
  rows=$(grep -c . "oracle/jobs/$j/results/$j.jsonl" 2>/dev/null || echo 0); n=$(items "$j")
  verdicts=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('verdicts'))" "oracle/jobs/$j/summary.json" 2>/dev/null)
  if [ "$rows" -eq 0 ]; then echo "$(date +%T) EMPTY $j: 0 rows, not publishing, stopping"; exit 4; fi
  if publish "$j"; then echo "$(date +%T) DONE $j rows=$rows/$n verdicts=$verdicts pushed oracle-results/$j"; else echo "$(date +%T) PUSH FAILED $j"; fi
  if [ "$rows" -lt "$n" ]; then echo "$(date +%T) SHORT $j: $rows of $n rows, stopping (crash?)"; exit 2; fi
done
echo "$(date +%T) QUEUE EMPTY"
