"""Build a self-contained TPU oracle job directory: everything a Cloud TPU VM with no git checkout needs to run and reproduce the job.

  python oracle/make_pack.py ITEMS.jsonl OUT_DIR --job-id ID --owner NAME --step STEP [--seeds 3] [--speed-bar 0.8] [--shards 1]
                             [--reference-only] [--sensitivity [--sensitivity-seed 1] [--sensitivity-bar X]] [--groups G1,G2]
  python oracle/make_pack.py oracle/manifest.json OUT_DIR ... --reference-only --groups final_exam [--bench benchmarks/jaxbench]

ITEMS.jsonl: one kernel per line, {id, task, ref_code, code} or {id, task, ref_path, code_path} (paths relative to the items file).
An item may carry "human_target": true (a human kernel, a speed target: tpu_oracle.py records is_kernel but does not enforce it).
Items of one task must share one reference. --reference-only: the code is not needed and ignored; the job is one item per task,
id "ref/<task>", and the oracle runs and times only the reference (tpu_oracle.py --reference-only). --sensitivity (implies
--reference-only) passes tpu_oracle.py's --sensitivity, --sensitivity-seed and --sensitivity-bar through and records them in the
manifest. ITEMS may instead be a JAXBench manifest (.json, reference-only): one item per task, its plain_ref or else its
human_kernel under --bench. --groups keeps only the tasks of those manifest groups, in the order given (final_exam first).

OUT_DIR gets:
  tpu_oracle.py, tpu_check.py, outside_ops.py, structure.py, keys.py  copied verbatim, in the oracle folder's layout so the
                                                                oracle's imports and its oracle_sha (a hash of exactly these four files) are unchanged
  pack.json      the oracle's input format; items in "dev", each carrying its reference's ref_sha256
  run.sh         N shards in parallel, resumable; checks every file against MANIFEST.json first; writes results/<job_id>.jsonl
  MANIFEST.json  job_id, owner, step, oracle_commit, oracle_sha, seeds, speed_bar, n_items, shards, reference_only, the sha256 of
                 every other file and of every item's code and reference
The pure parts import without JAX; nothing here runs any item's code.
"""
import argparse, hashlib, json, pathlib, re, shutil, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tpu_oracle import N_SEEDS, ROOT, SPEED_BAR  # noqa: E402  (pure at import: no JAX)

ORACLE_FILES = ("tpu_oracle.py", "tpu_check.py", "outside_ops.py", "structure.py", "keys.py")  # tpu_oracle._provenance hashes these, in order
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _text(row: dict, key: str, base: pathlib.Path) -> str | None:
    """row[key + '_code'] or the file at row[key + '_path']; exactly one of them, or neither (None)."""
    inline, path = row.get(f"{key}_code") if key == "ref" else row.get("code"), row.get(f"{key}_path")
    if inline is not None and path is not None:
        raise ValueError(f"item {row.get('id')!r}: give {key} inline or as a path, not both")
    if path is not None:
        return (base / path).read_text()
    return inline


def load_items(src, base: pathlib.Path | None = None, reference_only: bool = False) -> list:
    """Items as [{id, task, ref_code, code}], validated. `src` is a JSONL path (paths inside resolve against its directory) or a list
    of dicts (paths resolve against `base`)."""
    if isinstance(src, (str, pathlib.Path)):
        p = pathlib.Path(src)
        src, base = [json.loads(l) for l in p.read_text().splitlines() if l.strip()], p.parent
    base = pathlib.Path(base or ".")
    out, refs, seen = [], {}, set()
    for row in src:
        i, t = row.get("id"), row.get("task")
        if not (isinstance(i, str) and i and isinstance(t, str) and t):
            raise ValueError(f"every item needs a string id and task: {str(row)[:120]}")
        if i in seen:
            raise ValueError(f"duplicate id {i!r}")
        seen.add(i)
        ref, code = _text(row, "ref", base), None if reference_only else _text(row, "code", base)
        if ref is None or (code is None and not reference_only):
            raise ValueError(f"item {i!r}: needs {'a reference' if ref is None else 'code'} (ref_code/ref_path, code/code_path)")
        if refs.setdefault(t, ref) != ref:
            raise ValueError(f"item {i!r}: task {t!r} already has a different reference")
        if row.get("human_target", False) not in (True, False):
            raise ValueError(f"item {i!r}: human_target must be true or false")
        if row.get("human_target"):  # the flag switches off is_kernel (and may skip structure seeds): only JAXBench's own human kernels
            want = human_kernel_hashes().get(t)
            if want is None or code is None or hashlib.sha256(code.encode()).hexdigest() != want:
                raise ValueError(f"item {i!r}: human_target is only for task {t!r}'s JAXBench human kernel (manifest human_kernel_sha256)")
        out.append({"id": i, "task": t, "ref_code": ref, "code": code, **({"human_target": True} if row.get("human_target") else {})})
    if not out:
        raise ValueError("no items")
    return out


def human_kernel_hashes(manifest=None) -> dict:
    """{task: human_kernel_sha256} from the study's manifest: the only code that may carry human_target (Lebron James 6705030345)."""
    tasks = json.loads(pathlib.Path(manifest or ROOT / "manifest.json").read_text())["tasks"]
    return {t["task"]: t["human_kernel_sha256"] for t in tasks if t.get("human_kernel_sha256")}


def manifest_items(manifest, groups: list, bench) -> list:
    """Reference-only items from a JAXBench manifest: one per task in `groups` (every task when empty), grouped in that order."""
    tasks = json.loads(pathlib.Path(manifest).read_text())["tasks"]
    order = groups or list(dict.fromkeys(t.get("group") for t in tasks))
    rows = [{"id": f"ref/{t['task']}", "task": t["task"], "ref_path": t.get("plain_ref") or t["human_kernel"]}
            for g in order for t in tasks if t.get("group") == g]
    if not rows:
        raise ValueError(f"no manifest task in groups {groups}")
    return load_items(rows, bench, reference_only=True)


def in_groups(items: list, groups: list, manifest) -> list:
    """`items` restricted to the tasks of these manifest groups, grouped in the order given."""
    group = {t["task"]: t.get("group") for t in json.loads(pathlib.Path(manifest).read_text())["tasks"]}
    out = [i for g in groups for i in items if group.get(i["task"]) == g]
    if not out:
        raise ValueError(f"no item in groups {groups}")
    return out


def oracle_commit() -> str:
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    if p.returncode:
        return "unknown"
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *ORACLE_FILES], capture_output=True, text=True).stdout
    return p.stdout.strip() + ("-dirty" if dirty.strip() else "")  # a commit hides local edits; oracle_sha is exact either way


RUN_SH = r'''#!/usr/bin/env bash
# TPU oracle job __JOB__ (__OWNER__, __STEP__): __N_ITEMS__ items in __SHARDS__ parallel shards, one TPU chip per shard.
# oracle_sha __ORACLE_SHA__: pin this as build_rows --oracle-sha for rows judged by this job (never an older validated sha).
#   bash run.sh [--cpu-smoke]    -> results/__JOB__.jsonl (logs/shard<i>.log while it runs)
# Resumable: rerun after any stop and only the ids missing from shards/<i>of__SHARDS__.jsonl run. Keep the shard count: an id's shard is
# its position mod __SHARDS__. --cpu-smoke runs Pallas in interpret mode on CPU (plumbing only). PYTHON overrides python3.
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
export ORACLE_COMMIT=__COMMIT__ PYTHONDONTWRITEBYTECODE=1
if [ -z "$SMOKE" ]; then  # model-written code must not reach the VM's service-account token (QUEUE.md, "VM security")
  command -v curl >/dev/null || { echo "refusing to run: no curl to verify the metadata block" >&2; exit 1; }
  for u in http://169.254.169.254 http://metadata.google.internal "http://[fd20:ce::254]"; do
    if [ "$(curl -s -m 4 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' \
          "$u/computeMetadata/v1/instance/service-accounts/default/token" 2>/dev/null)" = 200 ]; then
      echo "refusing to run: the metadata token is reachable at $u; block it first (QUEUE.md, VM security)" >&2; exit 1; fi
  done
  # the block must hold against the job's own code: no sudo, no Docker socket, no admin group (Lebron James 6708367610)
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
N=__SHARDS__
for i in $(seq 0 $((N - 1))); do
  CHIP=""
  # ponytail: one process per chip via libtpu's env vars; not yet run on a real multi-chip VM. N=1 uses the default (all chips).
  [ -z "$SMOKE" ] && [ "$N" -gt 1 ] && CHIP="TPU_VISIBLE_CHIPS=$i TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_PORT=$((8476 + i))"
  env $CHIP "$PY" tpu_oracle.py pack.json "shards/${i}of$N.jsonl" --shard "$i/$N" __FLAGS__ --stack "${STACK:-cloud-tpu-vm}" $SMOKE \
    > "logs/shard$i.log" 2>&1 &
  pids="${pids:-} $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
cat shards/*of$N.jsonl > "results/__JOB__.jsonl"
echo "results/__JOB__.jsonl: $(grep -vc '"meta": true' "results/__JOB__.jsonl" || true) rows of __N_ITEMS__$([ $fail = 0 ] || echo '; a shard failed, see logs/, rerun to resume')"
exit $fail
'''


def build(items: list, out: pathlib.Path, *, job_id: str, owner: str, step: str, seeds: int, speed_bar: float, shards: int,
          reference_only: bool = False, sensitivity: dict | None = None, extra: dict | None = None) -> dict:
    """Write the job directory `out` (must not exist) and return its manifest. `extra` is merged into the manifest. `sensitivity`
    ({seed, bar}, bar None for the default rule) makes it a reference-only sensitivity job."""
    reference_only = reference_only or sensitivity is not None
    if not JOB_ID.fullmatch(job_id or ""):
        raise ValueError(f"job_id {job_id!r}: letters, digits, '.', '_', '-' only")
    if not (isinstance(seeds, int) and seeds >= 1 and isinstance(shards, int) and shards >= 1 and float(speed_bar) > 0):
        raise ValueError(f"need seeds >= 1, shards >= 1, speed_bar > 0 (got {seeds}, {shards}, {speed_bar})")
    out = pathlib.Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: a job directory is built once")
    refs = {i["task"]: i["ref_code"] for i in items}
    ref_sha = {t: sha256(c.encode()) for t, c in refs.items()}
    if reference_only:
        pack_items = [{"id": f"ref/{t}", "task": t, "ref_sha256": ref_sha[t]} for t in refs]
    else:
        pack_items = [{"id": i["id"], "task": i["task"], "code": i["code"], "ref_sha256": ref_sha[i["task"]],
                       **({"human_target": True} if i.get("human_target") else {})} for i in items]
    for rel in ORACLE_FILES:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, out / rel)
    (out / "pack.json").write_text(json.dumps({"dev_refs": refs, "train_refs": {}, "dev": pack_items, "train": []}))
    commit = oracle_commit()
    flags = f"--seeds {seeds} --speed-bar {speed_bar}" + (" --reference-only" if reference_only else "")
    if sensitivity is not None:
        flags += f" --sensitivity --sensitivity-seed {sensitivity['seed']}" + (
            f" --sensitivity-bar {sensitivity['bar']}" if sensitivity["bar"] is not None else "")
    oracle_sha = sha256(b"".join((out / p).read_bytes() for p in ORACLE_FILES))[:12]
    subs = {"__JOB__": job_id, "__OWNER__": owner, "__STEP__": str(step), "__N_ITEMS__": str(len(pack_items)), "__SHARDS__": str(shards),
            "__COMMIT__": commit, "__FLAGS__": flags, "__ORACLE_SHA__": oracle_sha}
    sh = RUN_SH
    for k, v in subs.items():
        sh = sh.replace(k, v)
    (out / "run.sh").write_text(sh)
    (out / "run.sh").chmod(0o755)
    m = {"job_id": job_id, "owner": owner, "step": step, "oracle_commit": commit,
         "oracle_sha": oracle_sha,
         "seeds": list(range(seeds)), "speed_bar": speed_bar, "n_items": len(pack_items), "shards": shards, "reference_only": reference_only,
         **({"sensitivity": sensitivity} if sensitivity is not None else {}), **(extra or {}),
         "files": {str(p.relative_to(out)): sha256(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file()},
         "items": [{"id": i["id"], "task": i["task"], "code_sha256": sha256(i["code"].encode()) if "code" in i else None,
                    "ref_sha256": i["ref_sha256"]} for i in pack_items]}
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--job-id", required=True); ap.add_argument("--owner", required=True); ap.add_argument("--step", required=True)
    ap.add_argument("--seeds", type=int, default=N_SEEDS); ap.add_argument("--speed-bar", type=float, default=SPEED_BAR)
    ap.add_argument("--shards", type=int, default=1, help="parallel shards, one per TPU chip (v6e-1: 1, v6e-8: 8)")
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="reference only: each floating input zeroed in turn (tpu_oracle.py)")
    ap.add_argument("--sensitivity-seed", type=int, default=1)
    ap.add_argument("--sensitivity-bar", type=float, default=None, help="default: max(2 * err_xla_default, 0.01) per task")
    ap.add_argument("--groups", default="", help="comma-separated manifest groups, in run order, e.g. final_exam")
    ap.add_argument("--manifest", default=str(ROOT / "manifest.json"), help="for --groups on an items file")
    ap.add_argument("--bench", default=str(ROOT.parent / "benchmarks/jaxbench"), help="where a manifest's reference paths resolve")
    a = ap.parse_args(argv)
    ro, groups = a.reference_only or a.sensitivity, [g for g in a.groups.split(",") if g]
    if a.items.endswith(".json"):
        if not ro:
            raise SystemExit("a manifest has no kernels: use --reference-only or --sensitivity")
        items = manifest_items(a.items, groups, a.bench)
    else:
        items = load_items(a.items, reference_only=ro)
        items = in_groups(items, groups, a.manifest) if groups else items
    sens = {"seed": a.sensitivity_seed, "bar": a.sensitivity_bar} if a.sensitivity else None
    m = build(items, pathlib.Path(a.out), job_id=a.job_id, owner=a.owner, step=a.step, seeds=a.seeds, speed_bar=a.speed_bar,
              shards=a.shards, reference_only=ro, sensitivity=sens, extra={"groups": groups} if groups else None)
    print(f"{a.out}: {m['n_items']} items, {m['shards']} shards, oracle {m['oracle_commit']} sha {m['oracle_sha']}")


if __name__ == "__main__":
    main()
