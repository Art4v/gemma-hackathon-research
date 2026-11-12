#!/usr/bin/env bash
# Oracle jobs on a Colab TPU v6e session, through the colab CLI (logged in already; compute units bought with `colab pay`).
# Approved Nov 8: on Colab the oracle runs directly. run.sh's Cloud VM security checks are skipped (ORACLE_ON_COLAB=1: Colab runs as
# root and a session holds no credentials of ours; never mount Drive or log in inside the session). Its file check and JAX pin still run.
#
#   [NAME=oracle-colab] [TPU=v6e1]          TPU= (empty) gives a free CPU session, for testing the plumbing with SMOKE=1
#   bash oracle/colab.sh start              # the session (billable from here; check the rate with `colab usage`)
#   bash oracle/colab.sh setup              # jax[tpu]==0.10.2 in the session's Python, and the chip it sees
#   bash oracle/colab.sh push DIR [DIR ...] # job directories (make_pack.py / verify_refs.py build) to /content/jobs/
#   bash oracle/colab.sh run JOB            # JOB's run.sh in the background; it keeps running between commands. SMOKE=1: --cpu-smoke
#   bash oracle/colab.sh status JOB         # the last log lines, rows written so far, running or not
#   bash oracle/colab.sh fetch JOB          # results/, shards/, logs/ and run.log back into oracle/jobs/JOB/
#   bash oracle/colab.sh stop               # release the session (always, when done: an idle session still burns compute units)
set -euo pipefail
NAME=${NAME:-oracle-colab}
TPU=${TPU-v6e1}
J=/content/jobs
here=$(cd "$(dirname "$0")/.." && pwd)

py_on() {  # run Python source (stdin) in the session's kernel; $1 = timeout seconds
  local f; f=$(mktemp -t colabexec).py; cat > "$f"
  colab exec -s "$NAME" -f "$f" --timeout "$1"; local rc=$?; rm -f "$f"; return $rc
}
job_name() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "bad job name: $1" >&2; exit 1; }; echo "$1"; }

case "${1:-}" in
start)
  colab new -s "$NAME" ${TPU:+--tpu "$TPU"}
  colab status -s "$NAME"
  ;;
setup)
  spec='jax==0.10.2'; [ -n "$TPU" ] && spec='jax[tpu]==0.10.2'
  py_on 1800 <<EOF
import subprocess, sys
# jaxtyping and pydantic: JAXBench's 64p reference imports them (requirements.txt pins the same versions)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "$spec", "jaxtyping==0.3.11", "pydantic==2.14.3"], check=True)
print(subprocess.run([sys.executable, "-c", "import jax; print('jax', jax.__version__, jax.devices())"], capture_output=True, text=True, check=True).stdout)
EOF
  ;;
push)
  shift; [ $# -ge 1 ] || { echo "push DIR [DIR ...]" >&2; exit 1; }
  for d in "$@"; do
    j=$(job_name "$(basename "$d")"); [ -f "$d/run.sh" ] || { echo "$d: no run.sh, not a job directory" >&2; exit 1; }
    t=$(mktemp -d); tar -C "$(dirname "$d")" -czf "$t/$j.tgz" "$j"
    py_on 60 <<< "import os; os.makedirs('$J', exist_ok=True)"
    colab upload -s "$NAME" "$t/$j.tgz" "$J/$j.tgz"; rm -rf "$t"
    py_on 300 <<EOF
import shutil, subprocess
shutil.rmtree("$J/$j", ignore_errors=True)
subprocess.run(["tar", "-C", "$J", "-xzf", "$J/$j.tgz"], check=True)
print("pushed $j")
EOF
  done
  ;;
run)
  j=$(job_name "${2:?run JOB}")
  py_on 60 <<EOF
import os, subprocess, sys
env = {**os.environ, "ORACLE_ON_COLAB": "1", "PYTHON": sys.executable, "STACK": "colab-tpu"}
args = ["bash", "run.sh"] + (["--cpu-smoke"] if "${SMOKE:-}" == "1" else [])
p = subprocess.Popen(args, cwd="$J/$j", env=env, stdout=open("$J/$j/run.log", "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
open("$J/$j/run.pid", "w").write(str(p.pid))
print("started $j, pid", p.pid)
EOF
  ;;
status)
  j=$(job_name "${2:?status JOB}")
  py_on 60 <<EOF
import glob, os
d = "$J/$j"
log = open(d + "/run.log").read().splitlines() if os.path.exists(d + "/run.log") else []
print("\n".join(log[-5:]))
rows = sum(1 for f in glob.glob(d + "/shards/*.jsonl") for l in open(f) if '"meta": true' not in l)
pid = int(open(d + "/run.pid").read()) if os.path.exists(d + "/run.pid") else None
alive = pid is not None and os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z"
print(f"rows written: {rows}; {'running' if alive else 'not running'}")
EOF
  ;;
fetch)
  j=$(job_name "${2:?fetch JOB}")
  py_on 300 <<EOF
import subprocess
subprocess.run(["tar", "-C", "$J/$j", "-czf", "/content/$j-results.tgz", "results", "shards", "logs", "run.log"], check=True)
print("packed")
EOF
  mkdir -p "$here/oracle/jobs/$j"
  colab download -s "$NAME" "/content/$j-results.tgz" "$here/oracle/jobs/$j/$j-results.tgz"
  tar -C "$here/oracle/jobs/$j" -xzf "$here/oracle/jobs/$j/$j-results.tgz" && rm "$here/oracle/jobs/$j/$j-results.tgz"
  echo "fetched into oracle/jobs/$j/results/"
  ;;
stop)
  colab stop -s "$NAME"
  colab sessions
  ;;
*)
  sed -n '2,15p' "$0"; exit 1
  ;;
esac
