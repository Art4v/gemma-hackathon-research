#!/usr/bin/env bash
# Karthik's hunter (one create in flight, on-demand), then our vm8.sh bootstrap / security check / one-process-per-chip check.
set -uo pipefail
cd "$(dirname "$0")"
export PROJECT=project-dfbc8c9e-7988-418a-856 NAME=oracle-v6e1-a2m1-hruday ACCEL=v6e-1
out=$(MAX_ROUNDS=${MAX_ROUNDS:-30} ZONES="us-east5-b europe-west4-a asia-south1-b southamerica-west1-a us-central1-a us-central1-c us-south1-a us-south1-c us-west1-c us-east5-a us-central1-b asia-northeast1-b asia-southeast1-b asia-east1-c asia-south1-c southamerica-east1-c" bash grab_v6e.sh | tee -a hunt.log | tail -1)
echo "$out"
z=$(echo "$out" | sed -n 's/.*WON zone=\([^ ]*\).*/\1/p')
[ -n "$z" ] || { echo "HUNT FAILED"; exit 1; }
echo "$z" > zone.txt; date -u +%s > created_at.txt
export ZONE=$z
# Nov 11: on a fresh VM the first SSH can race Google's key propagation ("Permission denied (publickey)") and bootstrap fails
# silently; retry until it reports done, verify the oracle user, and delete the VM if it never comes up (no VM may bill unguarded).
ok=0
for i in 1 2 3 4 5; do
  b=$(bash vm8.sh bootstrap 2>&1 | tail -3); echo "$b"
  echo "$b" | grep -q "bootstrap done" && { ok=1; break; }
  echo "bootstrap attempt $i failed; retrying in 30 s"; sleep 30
done
if [ $ok = 1 ] && bash vm8.sh check 2>&1 | tail -6 | tee /dev/stderr | grep -q . && gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --command="id oracle" >/dev/null 2>&1; then
  :
else
  echo "BOOTSTRAP FAILED on $NAME in $z; deleting it"; bash vm8.sh delete 2>&1 | tail -1; rm -f zone.txt created_at.txt; exit 1
fi
echo "== chipcheck"; bash vm8.sh chipcheck 2>&1 | tail -10
echo "UP_DONE zone=$z"
