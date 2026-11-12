"""Judge on the CPU what the CPU can decide, before any TPU time: the oracle's own step 1 (tpu_oracle.py --static-only) on every attempt
waiting for a verdict. Step 1 traces the kernel without running it (import, workload, the kernel check), so a failure there is the same
verdict and the same feedback line a TPU run gives at that step: those are recorded as the attempt's verdict (judged_on says CPU). The
attempts that pass step 1 go to the TPU, written as oracle items for make_pack.py; for each the TPU lowering is also tried on this CPU
(curation/lower_filter.py, sandboxed) and reported, for information only: the TPU judges them either way, so a retry always sees the
oracle's own feedback.

  python3 control/cpu_check.py --backend modal --tag t1 --py <a JAX 0.10.2 python> [--ingest]
    -> control/runs/cpu_checks/<tag>_static.jsonl (decided on CPU), <tag>_tpu_items.jsonl (for the TPU), <tag>_lowering.jsonl
"""
import argparse, collections, json, os, pathlib, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

JUDGED_ON = "cpu: oracle step 1 (tpu_oracle.py --static-only), the same verdict a TPU run gives at that step"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=bench.BACKENDS, default="modal"); ap.add_argument("--model", default=bench.MODEL)
    ap.add_argument("--tag", required=True); ap.add_argument("--py", required=True, help="a Python with JAX 0.10.2")
    ap.add_argument("--ingest", action="store_true", help="record the CPU-decided verdicts in the run")
    a = ap.parse_args(argv)
    bench.BACKEND = a.backend
    items = bench.items(a.model)
    if not items:
        raise SystemExit("no attempts are waiting for a verdict")
    out = ROOT / "control/runs/cpu_checks"; out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "pack.json").write_text(json.dumps({"refs": {i["task"]: i["ref_code"] for i in items},
                                                 "items": [{"id": i["id"], "task": i["task"], "code": i["code"]} for i in items]}))
        subprocess.run([a.py, str(ROOT / "oracle/tpu_oracle.py"), str(d / "pack.json"), str(d / "static.jsonl"), "--static-only"],
                       env={**os.environ, "JAX_PLATFORMS": "cpu"}, stdout=subprocess.DEVNULL, check=True)
        rows = [json.loads(l) for l in (d / "static.jsonl").read_text().splitlines() if '"meta": true' not in l]
        survivors = {r["id"] for r in rows if r["verdict"] == "static_ok"}
        decided = [{**r, "judged_on": JUDGED_ON} for r in rows if r["verdict"] != "static_ok"]
        (out / f"{a.tag}_static.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decided))
        tpu = [i for i in items if i["id"] in survivors]
        (out / f"{a.tag}_tpu_items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in tpu))
        lowering = []
        if tpu:
            (d / "lf_items.jsonl").write_text("".join(json.dumps({"task": i["task"], "ref": i["ref_code"]}) + "\n" for i in tpu))
            (d / "lf_rows.jsonl").write_text("".join(json.dumps({"id": i["id"], "task": i["task"], "code": i["code"]}) + "\n" for i in tpu))
            subprocess.run([sys.executable, str(ROOT / "curation/lower_filter.py"), "filter", "--py", a.py, "--items", str(d / "lf_items.jsonl"),
                            "--rows", str(d / "lf_rows.jsonl"), "--out", str(d / "lower.jsonl"), "--feedback-out", str(d / "lower_fb.jsonl")],
                           stdout=subprocess.DEVNULL, check=True)
            fb = {json.loads(l)["id"]: json.loads(l)["feedback"] for l in (d / "lower_fb.jsonl").read_text().splitlines()}
            lowering = [{**json.loads(l), "feedback": fb.get(json.loads(l)["id"], "")} for l in (d / "lower.jsonl").read_text().splitlines()]
        (out / f"{a.tag}_lowering.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lowering))
    if a.ingest and decided:
        bench.ingest(a.model, [out / f"{a.tag}_static.jsonl"])
    for arm in bench.ARMS:
        rs = [r for r in rows if r["id"].startswith(arm + "/")]
        print(f"{arm}: {len(rs)} checked -> {dict(collections.Counter(r['verdict'] for r in rs))}")
    print(f"{len(decided)} decided on CPU{' and ingested' if a.ingest else ''}; {len(tpu)} for the TPU -> {out / (a.tag + '_tpu_items.jsonl')}")
    for r in lowering:
        print(f"  TPU lowering on CPU: {r['id']}: {r['outcome']} {r['feedback'][:120]}")


if __name__ == "__main__":
    main()
