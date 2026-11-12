"""Check each level 2-only task's plain-JAX reference (control/refs) against Google's own kernel, on the TPU, before the benchmark uses it.

  python control/verify_refs.py build OUT_DIR --job-id ID --owner NAME        # a job directory, like make_pack.py's, on the Mac
  bash OUT_DIR/run.sh                                                         # on the v6e VM, as QUEUE.md says
  python control/verify_refs.py report OUT_DIR/results/ID.jsonl               # which references hold

The reference is the oracle's answer key, so a wrong reference would mark correct kernels wrong. Google's kernel is judged by the real
oracle (tpu_oracle.py, unchanged) with that reference: same seeds, same answer key, same tolerance. The one difference is that the kernel
check is skipped, because Google's kernels do legitimate setup outside pallas_call; each row says so (kernel_check_skipped, and
is_kernel_as_checked for what the check found). A reference holds iff Google's kernel comes out pass or correct_slow on every seed: then
the reference computes what the kernel computes. Anything else (wrong, a failure, a harness error) means the task's reference is not used
until it is fixed.
"""
import argparse, hashlib, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CORRECT = {"pass", "correct_slow"}


def run(argv) -> None:
    """On the VM: tpu_oracle.main with the kernel check recorded but not enforced."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import tpu_oracle as o
    real = o.kernel_check

    def recorded_only(jaxpr):
        r = real(jaxpr)
        return {**r, "is_kernel": True, "reason": "", "is_kernel_as_checked": r["is_kernel"],
                "kernel_check_reason": r["reason"], "kernel_check_skipped": True}
    o.kernel_check = recorded_only
    o.main(argv)


def build(out: pathlib.Path, job_id: str, owner: str) -> dict:
    sys.path.insert(0, str(ROOT / "oracle"))
    import make_pack
    rows = [{"id": f"verify/{p.stem}", "task": p.stem, "ref_code": p.read_text(),
             "code": (ROOT / f"benchmarks/jaxbench/level2/{p.stem}/baseline.py").read_text()} for p in sorted((HERE / "refs").glob("*.py"))]
    m = make_pack.build(make_pack.load_items(rows), out, job_id=job_id, owner=owner, step="verify-refs", seeds=3,
                        speed_bar=make_pack.SPEED_BAR, shards=1)
    (out / "verify_refs.py").write_text(pathlib.Path(__file__).read_text())
    sh = (out / "run.sh").read_text()
    assert sh.count('"$PY" tpu_oracle.py pack.json') == 1
    (out / "run.sh").write_text(sh.replace('"$PY" tpu_oracle.py pack.json', '"$PY" verify_refs.py run pack.json'))
    m["files"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name != "MANIFEST.json"}
    m["verify_refs"] = True
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


def report(paths) -> list:
    rows = [json.loads(l) for p in paths for l in pathlib.Path(p).read_text().splitlines() if l.startswith("{") and '"meta": true' not in l]
    out = []
    for r in rows:
        ok = r.get("verdict") in CORRECT and len(r.get("per_seed") or []) >= 3
        out.append({"task": r["task"], "holds": ok, "verdict": r.get("verdict"), "feedback": r.get("feedback", "")[:200],
                    "err_kernel": [s.get("err_kernel") for s in r.get("per_seed") or []]})
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        return run(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("out"); b.add_argument("--job-id", required=True); b.add_argument("--owner", required=True)
    r = sub.add_parser("report"); r.add_argument("results", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build(pathlib.Path(a.out), a.job_id, a.owner)
        print(f"{a.out}: {m['n_items']} references to check, oracle sha {m['oracle_sha']}")
    else:
        rows = report(a.results)
        for x in rows:
            print(f"{x['task']:40} {'HOLDS' if x['holds'] else 'FAILS'}  {x['verdict']}  {x['feedback'][:100]}")
        print(f"{sum(x['holds'] for x in rows)} of {len(rows)} references hold")


if __name__ == "__main__":
    main()
