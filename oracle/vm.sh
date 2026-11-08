#!/usr/bin/env bash
# One TPU v6e VM for oracle jobs, set up exactly as QUEUE.md's "VM security" recipe (session 2, 2026-11-07). Run from a machine with
# gcloud, logged in (gcloud auth login) to a project that has v6e quota in ZONE.
#
#   export PROJECT=<project id> ZONE=<a zone with v6e> [NAME=oracle-v6e] [SPOT=1]
#   bash oracle/vm.sh create              # the VM: v6e-1, logging + read-only storage scopes and nothing broader
#   bash oracle/vm.sh bootstrap           # the no-sudo oracle user, its metadata block (IPv4 + IPv6), Python 3.12 + jax[tpu]==0.10.2 for it
#   bash oracle/vm.sh check               # as that user: the token is unreachable, no sudo, no admin group, and the TPU is visible
#   bash oracle/vm.sh push DIR [DIR ...]  # job directories (make_pack.py / verify_refs.py build) to /home/oracle/jobs/
#   bash oracle/vm.sh run JOB             # JOB's run.sh in the background as oracle; it survives the ssh session ending
#   bash oracle/vm.sh status JOB          # the last log lines and the rows written so far
#   bash oracle/vm.sh fetch JOB           # results/, shards/, logs/ and run.log back into oracle/jobs/JOB/
#   bash oracle/vm.sh delete              # delete the VM, then list what is left in ZONE (should be nothing)
#
# The VM bills from create to delete, idle or not. run.sh is resumable: after a spot preemption, create, bootstrap, push and run again.
set -euo pipefail
: "${PROJECT:?export PROJECT=<project id>}" "${ZONE:?export ZONE=<zone with v6e>}"
NAME=${NAME:-oracle-v6e}
G=(gcloud compute tpus tpu-vm --project="$PROJECT" --zone="$ZONE")
OH=/home/oracle
PY=$OH/venv/bin/python
# libtpu can't ask the blocked metadata server for the topology, so it comes from the environment (QUEUE.md step 4)
TPU_ENV="TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1"
on_vm() { "${G[@]}" ssh "$NAME" --command="$1"; }
job_name() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "bad job name: $1" >&2; exit 1; }; echo "$1"; }

case "${1:-}" in
create)
  "${G[@]}" create "$NAME" --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e \
    --scopes=https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/devstorage.read_only ${SPOT:+--spot}
  ;;
bootstrap)
  on_vm 'set -euo pipefail
    id oracle >/dev/null 2>&1 || sudo useradd -m -s /bin/bash oracle
    U=$(id -u oracle)
    sudo iptables  -C OUTPUT -m owner --uid-owner "$U" -d 169.254.169.254 -j REJECT 2>/dev/null || sudo iptables  -I OUTPUT 1 -m owner --uid-owner "$U" -d 169.254.169.254 -j REJECT
    sudo ip6tables -C OUTPUT -m owner --uid-owner "$U" -d fd20:ce::254     -j REJECT 2>/dev/null || sudo ip6tables -I OUTPUT 1 -m owner --uid-owner "$U" -d fd20:ce::254     -j REJECT
    sudo chmod 750 /home/oracle
    sudo -u oracle -H bash -c "cd ~ && mkdir -p jobs && curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null && ~/.local/bin/uv venv -q --python 3.12 ~/venv && ~/.local/bin/uv pip install -q --python ~/venv/bin/python \"jax[tpu]==0.10.2\""
    echo bootstrap done'
  ;;
check)
  on_vm "sudo -u oracle -H env $TPU_ENV bash -c '
    for u in http://169.254.169.254 http://metadata.google.internal \"http://[fd20:ce::254]\"; do
      c=\$(curl -s -m 4 -o /dev/null -w \"%{http_code}\" -H \"Metadata-Flavor: Google\" \"\$u/computeMetadata/v1/instance/service-accounts/default/token\" || true)
      echo \"token at \$u: \$c (must not be 200)\"; done
    sudo -n true 2>/dev/null && echo \"sudo: YES (must be no)\" || echo \"sudo: no\"
    echo \"groups: \$(id -Gn)\"
    $PY -c \"import jax; print(\\\"jax\\\", jax.__version__, jax.devices())\"'"
  ;;
push)
  shift; [ $# -ge 1 ] || { echo "push DIR [DIR ...]" >&2; exit 1; }
  on_vm 'rm -rf /tmp/push && mkdir -p /tmp/push'
  for d in "$@"; do
    j=$(job_name "$(basename "$d")"); [ -f "$d/run.sh" ] || { echo "$d: no run.sh, not a job directory" >&2; exit 1; }
    "${G[@]}" scp --recurse "$d" "$NAME:/tmp/push/"
    on_vm "sudo rm -rf $OH/jobs/$j && sudo cp -r /tmp/push/$j $OH/jobs/ && sudo chown -R oracle:oracle $OH/jobs/$j && echo pushed $j"
  done
  ;;
run)
  j=$(job_name "${2:?run JOB}")
  on_vm "sudo -u oracle -H bash -c 'cd $OH/jobs/$j && PYTHON=$PY setsid nohup env $TPU_ENV bash run.sh > run.log 2>&1 < /dev/null & echo started $j'"
  ;;
status)
  j=$(job_name "${2:?status JOB}")
  on_vm "sudo tail -n 5 $OH/jobs/$j/run.log 2>/dev/null; sudo bash -c 'cat $OH/jobs/$j/shards/*.jsonl 2>/dev/null | grep -vc \"\\\"meta\\\": true\"' | xargs echo rows written:; pgrep -u oracle -f run.sh >/dev/null && echo running || echo 'not running'"
  ;;
fetch)
  j=$(job_name "${2:?fetch JOB}")
  on_vm "sudo tar -C $OH/jobs/$j -czf /tmp/$j-results.tgz results shards logs run.log && sudo chmod 644 /tmp/$j-results.tgz"
  mkdir -p "oracle/jobs/$j"
  "${G[@]}" scp "$NAME:/tmp/$j-results.tgz" "oracle/jobs/$j/"
  tar -C "oracle/jobs/$j" -xzf "oracle/jobs/$j/$j-results.tgz" && rm "oracle/jobs/$j/$j-results.tgz"
  echo "fetched into oracle/jobs/$j/results/"
  ;;
delete)
  "${G[@]}" delete "$NAME" --quiet
  echo "TPU VMs left in $ZONE:"; "${G[@]}" list
  ;;
*)
  sed -n '2,17p' "$0"; exit 1
  ;;
esac
