#!/usr/bin/env bash
# TPU oracle job control-tpu3 (overnight, control-tpu3): 1 items in 1 parallel shards, one TPU chip per shard.
# oracle_sha c9a07d9298a1: the exact oracle code that judges this job (every result row carries it).
#   bash run.sh [--cpu-smoke]    -> results/control-tpu3.jsonl (logs/shard<i>.log while it runs)
# Resumable: rerun after any stop and only the ids missing from shards/<i>of1.jsonl run. Keep the shard count: an id's shard is
# its position mod 1. --cpu-smoke runs Pallas in interpret mode on CPU (plumbing only). PYTHON overrides python3.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
SMOKE=""
[ "${1:-}" = "--cpu-smoke" ] && SMOKE="--cpu-smoke"
"$PY" - <<'EOF'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path("MANIFEST.json").read_text())
bad = [p for p, h in m["files"].items() if not pathlib.Path(p).is_file() or hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() != h]
sys.exit(f"refusing to run: changed or missing since the pack was built: {bad}" if bad else 0)
EOF
export ORACLE_COMMIT=963c0e3 PYTHONDONTWRITEBYTECODE=1
VM_CHECKS=1
if [ "${ORACLE_ON_COLAB:-}" = 1 ]; then  # Colab runs as root and holds no credentials of ours; approved Nov 8 (oracle/colab.sh)
  echo "ORACLE_ON_COLAB=1: Cloud VM security checks skipped (no project credentials in a Colab session); file check done above" >&2
  VM_CHECKS=""
fi
if [ -z "$SMOKE" ] && [ -n "$VM_CHECKS" ]; then  # model-written code must not reach the VM's service-account token (QUEUE.md, "VM security")
  command -v curl >/dev/null || { echo "refusing to run: no curl to verify the metadata block" >&2; exit 1; }
  for u in http://169.254.169.254 http://metadata.google.internal "http://[fd20:ce::254]"; do
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' \
          "$u/computeMetadata/v1/instance/service-accounts/default/token" 2>/dev/null)" = 200 ]; then
      echo "refusing to run: the metadata token is reachable at $u; block it first (QUEUE.md, VM security)" >&2; exit 1; fi
  done
  # the block must hold against the job's own code: no sudo, no Docker socket, no admin group
  if sudo -n true 2>/dev/null; then
    echo "refusing to run: this user has passwordless sudo and could lift the metadata block; run jobs as a no-sudo user (QUEUE.md)" >&2; exit 1; fi
  for s in ${RUNSH_SOCKETS:-/var/run/docker.sock /run/docker.sock /run/containerd/containerd.sock}; do  # empty falls back to the list
    if [ -w "$s" ]; then echo "refusing to run: this user can write the container socket $s, which is root (QUEUE.md)" >&2; exit 1; fi
  done
  if id -Gn | tr ' ' '\n' | grep -qxE 'docker|sudo|adm|google-sudoers|wheel|lxd|disk|root'; then
    echo "refusing to run: this user is in an admin group ($(id -Gn)); run jobs as a no-sudo user (QUEUE.md)" >&2; exit 1; fi
fi
if [ -z "$SMOKE" ]; then  # pin JAX once here, not in N shards at once (tpu_oracle.init would pip install in each)
  PIN=$(sed -n 's/^JAX_PIN = "\([^"]*\)".*/\1/p' tpu_check.py)
  "$PY" -c "import jax, sys; sys.exit(jax.__version__ != '$PIN')" 2>/dev/null || "$PY" -m pip install -q "jax[tpu]==$PIN"
fi
mkdir -p shards logs results
N=1
for i in $(seq 0 $((N - 1))); do
  CHIP=""
  # one process per chip via libtpu's env vars; not yet run on a real multi-chip VM. N=1 uses the default (all chips).
  [ -z "$SMOKE" ] && [ "$N" -gt 1 ] && CHIP="TPU_VISIBLE_CHIPS=$i TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_PORT=$((8476 + i))"
  env $CHIP "$PY" tpu_oracle.py pack.json "shards/${i}of$N.jsonl" --shard "$i/$N" --seeds 3 --speed-bar 0.7 --stack "${STACK:-cloud-tpu-vm}" $SMOKE \
    > "logs/shard$i.log" 2>&1 &
  pids="${pids:-} $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
cat shards/*of$N.jsonl > "results/control-tpu3.jsonl"
echo "results/control-tpu3.jsonl: $(grep -vc '"meta": true' "results/control-tpu3.jsonl" || true) rows of 1$([ $fail = 0 ] || echo '; a shard failed, see logs/, rerun to resume')"
exit $fail
