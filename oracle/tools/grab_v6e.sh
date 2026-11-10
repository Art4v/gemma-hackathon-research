#!/usr/bin/env bash
# Hunt for one on-demand TPU v6e VM (default v6e-8) across every zone that offers v6e, round after round, until one is granted.
# Prints one line per attempt with Google's own reason, then "WON zone=... name=... accel=..." and exits 0.
#   PROJECT=<project id> [ACCEL=v6e-8] [NAME=oracle-v6e8] [MAX_ROUNDS=40] [ZONES="z1 z2"] bash oracle/tools/grab_v6e.sh
# Sequential on purpose: only one create is ever in flight, so a hunt can never leave two VMs billing.
# On-demand only: spot v6e was preempted mid-job twice (Nov 8-9). The VM bills from WON until `vm.sh delete`.
# Use the result with: ZONE=<won zone> ACCEL=$ACCEL NAME=$NAME bash oracle/vm.sh bootstrap / check, and SHARDS=8 for crashloop.py.
set -u
P=${PROJECT:?set PROJECT: no default, so a hunt can never land on the wrong account}
ACCEL=${ACCEL:-v6e-8}
NAME=${NAME:-oracle-v6e8}
MAX_ROUNDS=${MAX_ROUNDS:-40}
SCOPES=https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/devstorage.read_only   # same as vm.sh create
# zones that list v6e (Nov 2026); europe-west4-a and us-east5-b first, where capacity came up most often for us
ZONES=${ZONES:-"europe-west4-a us-east5-b us-east5-a us-east5-c us-east1-d us-central1-a us-central1-b us-central1-c us-south1-a us-south1-c us-east4-a us-west1-c asia-northeast1-b asia-east1-c asia-south1-b asia-south1-c asia-southeast1-b southamerica-east1-c southamerica-west1-a"}

for r in $(seq 1 "$MAX_ROUNDS"); do
  echo "round $r $(date -u +%H:%M:%SZ) accel=$ACCEL"
  for z in $ZONES; do
    if out=$(gcloud compute tpus tpu-vm create "$NAME" --project="$P" --zone="$z" --accelerator-type="$ACCEL" \
               --version=v2-alpha-tpuv6e --scopes="$SCOPES" --quiet 2>&1); then
      echo "WON zone=$z name=$NAME accel=$ACCEL project=$P"; exit 0
    fi
    # Google's reason is the JSON "message" field; fall back to the last line
    why=$(echo "$out" | sed -n 's/.*"message": "\(.*\)".*/\1/p' | tr -d '\\' | head -1); [ -n "$why" ] || why=$(echo "$out" | tail -1)
    echo "  $z: $(echo "$why" | cut -c1-110)"
    # a create that half-succeeded blocks every later create with the same name: stop and let a human look
    echo "$out" | grep -qi 'already exists' && { echo "STOP: $NAME already exists in $z; check it before hunting again"; exit 2; }
  done
  sleep 60
done
echo "GAVE UP after $MAX_ROUNDS rounds"; exit 1
