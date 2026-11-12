#!/usr/bin/env bash
# Hard spending guard for our TPU VM: delete it when estimated spend passes CAP_USD, or after IDLE_MIN with no oracle job running.
# Estimated spend = uptime x RATE (v6e on-demand ~$2.97/chip-h, x8 chips, a conservative figure). Prints one line per action.
cd "$(dirname "$0")"
exec 9>guard.lock; flock -n 9 2>/dev/null || { [ -f guard.pid ] && kill -0 "$(cat guard.pid)" 2>/dev/null && { echo "guard already running"; exit 0; }; }; echo $$ > guard.pid
export PROJECT=project-dfbc8c9e-7988-418a-856 NAME=oracle-v6e1-a2m1-hruday
RATE=${RATE:-2.97}; CAP_USD=${CAP_USD:-40}; IDLE_MIN=${IDLE_MIN:-40}
idle=0; seen_job=${SEEN_JOB:-1}   # count idle time from creation: short jobs can finish between 5-minute checks (Nov 10: a v6e-1 idled 4h50m)
while true; do
  if [ -f zone.txt ] && [ -f created_at.txt ]; then
    Z=$(cat zone.txt); now=$(date -u +%s); up=$(( now - $(cat created_at.txt) ))
    spend=$(python3 -c "print(round($up/3600*$RATE,2))")
    if gcloud compute tpus tpu-vm describe $NAME --zone=$Z --project=$PROJECT --format="value(state)" >/dev/null 2>&1; then
      over=$(python3 -c "print(int($up/3600*$RATE >= $CAP_USD))")
      busy=$(gcloud compute tpus tpu-vm ssh $NAME --zone=$Z --project=$PROJECT --command="pgrep -u oracle -f 'tpu_oracle.py|run.sh' >/dev/null && echo 1 || echo 0" 2>/dev/null | tail -1)
      [ "$busy" = 1 ] && { idle=0; seen_job=1; } || idle=$((idle + 5))
      if [ "$over" = 1 ] || { [ "$seen_job" = 1 ] && [ $idle -ge $IDLE_MIN ]; }; then
        why=$([ "$over" = 1 ] && echo "spend cap \$$CAP_USD reached (est \$$spend)" || echo "idle ${idle} min after jobs")
        echo "$(date -u +%H:%M) GUARD DELETE $NAME in $Z: $why"
        ZONE=$Z bash vm8.sh delete 2>&1 | tail -2; rm -f zone.txt created_at.txt; seen_job=0; idle=0
      else
        echo "$(date -u +%H:%M) up $((up/60)) min, est \$$spend of \$$CAP_USD, busy=$busy idle=${idle}m" >> guard.status
      fi
    fi
  fi
  sleep 300
done
