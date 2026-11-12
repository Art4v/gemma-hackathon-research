"""Free CPU gates for the Kimi K3.1 kernels, as control/cpu_check.py runs them: (1) the oracle's own step 1 (tpu_oracle.py --static-only,
oracle sha c9a07d9298a1, JAX 0.10.2 on CPU), then (2) for step-1 survivors, Pallas TPU-v6e lowering on CPU (curation/lower_filter.py).
Loops: scores every new ok row in out/k31_*.jsonl in batches, appends to out/cpu_scored.jsonl, and exits once generation is done and
everything is scored.   python3 score_cpu.py"""
import glob, json, os, pathlib, subprocess, sys, tempfile, time
P = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/Users/hrudaynara/Projects/gemma-hackathon-research")
PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"  # JAX 0.10.2, as bench/cpu_check use
OUT = pathlib.Path(os.environ.get("OUT", str(P / "out" / "cpu_scored.jsonl")))
GEN_GLOB = os.environ.get("GEN_GLOB", str(P / "out" / "k31_*.jsonl"))
LOG_GLOB = os.environ.get("LOG_GLOB", str(P / "out" / "full_*_s*.log"))
BATCH = 60

refs = {}
for pf_ in os.environ.get("PROBLEMS", str(P / "full_problems.jsonl")).split(","):
    for l in open(pf_):
        r = json.loads(l)
        refs[r["problem_id"]] = (r["task_name"], r["problem_file"])
ref_code, files = {}, {}
for pid, (task, pf) in refs.items():
    files.setdefault(pf, set()).add(pid)
for pf, pids in files.items():
    for l in open(pf):
        r = json.loads(l)
        if r["problem_id"] in pids and r["problem_id"] not in ref_code:
            ref_code[r["problem_id"]] = r["ref_code"]


def gen_rows():
    rows = {}
    for f in sorted(glob.glob(GEN_GLOB)):
        for l in open(f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d.get("ok") and d.get("code") and d["problem_id"] in ref_code:
                rows[d["problem_id"]] = d
    return rows


def gen_done():
    logs = glob.glob(LOG_GLOB)
    return logs and all(any(m in open(x).read() for m in ("FULL GEN DONE", "REPAIR GEN DONE", "CAP REACHED")) for x in logs)


def score(batch):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "pack.json").write_text(json.dumps({"refs": {r["task_name"]: ref_code[r["problem_id"]] for r in batch},
                                                 "items": [{"id": r["problem_id"], "task": r["task_name"], "code": r["code"]} for r in batch]}))
        subprocess.run([PY, str(ROOT / "oracle/tpu_oracle.py"), str(d / "pack.json"), str(d / "static.jsonl"), "--static-only"],
                       env={**os.environ, "JAX_PLATFORMS": "cpu"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        st = {}
        if (d / "static.jsonl").exists():
            for l in (d / "static.jsonl").read_text().splitlines():
                if '"meta": true' not in l:
                    x = json.loads(l); st[x["id"]] = x
        ok = [r for r in batch if st.get(r["problem_id"], {}).get("verdict") == "static_ok"]
        low = {}
        if ok:
            (d / "lf_items.jsonl").write_text("".join(json.dumps({"task": r["task_name"], "ref": ref_code[r["problem_id"]]}) + "\n" for r in ok))
            (d / "lf_rows.jsonl").write_text("".join(json.dumps({"id": r["problem_id"], "task": r["task_name"], "code": r["code"]}) + "\n" for r in ok))
            subprocess.run([sys.executable, str(ROOT / "curation/lower_filter.py"), "filter", "--py", PY, "--items", str(d / "lf_items.jsonl"),
                            "--rows", str(d / "lf_rows.jsonl"), "--out", str(d / "lower.jsonl"), "--feedback-out", str(d / "lower_fb.jsonl")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
            fb = {json.loads(l)["id"]: json.loads(l)["feedback"] for l in (d / "lower_fb.jsonl").read_text().splitlines()} if (d / "lower_fb.jsonl").exists() else {}
            if (d / "lower.jsonl").exists():
                for l in (d / "lower.jsonl").read_text().splitlines():
                    x = json.loads(l); low[x["id"]] = {"outcome": x.get("outcome"), "feedback": fb.get(x["id"], "")}
    out = []
    for r in batch:
        s = st.get(r["problem_id"], {"verdict": "harness_no_row", "feedback": "static check produced no row"})
        out.append({"problem_id": r["problem_id"], "task_name": r["task_name"], "source": r.get("source"), "static": s.get("verdict"),
                    "static_feedback": (s.get("feedback") or "")[:400], "lowering": low.get(r["problem_id"], {}).get("outcome"),
                    "lowering_feedback": (low.get(r["problem_id"], {}).get("feedback") or "")[:400], "oracle_sha": s.get("oracle_sha")})
    return out


while True:
    scored = {json.loads(l)["problem_id"] for l in open(OUT)} if OUT.exists() else set()
    rows = gen_rows()
    todo = [r for pid, r in rows.items() if pid not in scored]
    if not todo:
        if gen_done():
            print("CPU SCORING DONE", len(scored), flush=True); break
        time.sleep(120); continue
    for i in range(0, len(todo), BATCH):
        res = score(todo[i:i + BATCH])
        with OUT.open("a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in res))
        n = len(scored) + i + len(res)
        so = sum(1 for x in res if x["static"] == "static_ok"); lo = sum(1 for x in res if x["lowering"] not in (None, "reject"))
        print(f"scored {n}: batch {len(res)} -> static_ok {so}, lowering not rejected {lo}", flush=True)
