"""Shape-shift robustness test: the final adapter on the 8 matmul k-tasks it passes, with M/K/N changed to shapes absent from training
(refs/index.json; shape A = 2048x7168x9216, tile-friendly; shape B = 3072x7680x8704, not multiples of 1024). Runs control/bench.py's own
generate() (same prompt, greedy, seed rule, 3-turn repair loop, digest assertion) with its task list and run dir pointed here.

  python robust.py gen              generate the next turn for every task without a terminal verdict (Modal, 6 workers)
  python robust.py step1 T          oracle --static-only on turn T's rows: failures -> verdicts.jsonl, static_ok -> TPU items
  python robust.py ingest T         TPU final verdicts of job robust-shift-tT (halt victims skipped) -> verdicts.jsonl
  python robust.py report"""
import collections, glob, json, os, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
import bench  # noqa: E402

MODEL = "gemma4-e4b-lora:v7final"; DIGEST = "42b1250057a1a8c3"
RUN = HERE / "run"; RUN.mkdir(exist_ok=True)
IDX = json.load(open(HERE / "refs/index.json"))
REFS = {n: (HERE / f"refs/{n}.py").read_text() for n in IDX}
bench.tasks = lambda: {n: (REFS[n], f"robust/{n}") for n in sorted(REFS)}
bench.run_dir = lambda model: RUN
OPS = pathlib.Path("/Users/hrudaynara/Projects/gemma-work/tpu/ops1")
PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"
ORACLE = "/Users/hrudaynara/Projects/gemma-hackathon-research/oracle/tpu_oracle.py"
halt = lambda r: r.get("verdict") == "harness_oom" and any(h in (r.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))


def rows():
    return bench.read_jsonl(RUN / "bare.jsonl") if (RUN / "bare.jsonl").exists() else []


def add_verdicts(vs):
    with open(RUN / "verdicts.jsonl", "a") as f:
        for v in vs:
            f.write(json.dumps(v) + "\n")


cmd = sys.argv[1]
if cmd == "gen":
    n = bench.generate("bare", MODEL, chat_fn=bench.modal_chat(MODEL), workers=6)
    bad = [r["id"] for r in rows() if not str((r.get("backend") or {}).get("digest", "")).startswith(DIGEST) or r.get("model") != MODEL]
    print(f"{n} generated; rows {len(rows())}; wrong model/digest: {len(bad)}")
    sys.exit(1 if bad else 0)
elif cmd == "step1":
    T = int(sys.argv[2]); have = bench.verdicts(MODEL)
    todo = [r for r in rows() if r["turn"] == T and r["id"] not in have]
    items = [{"id": r["id"], "task": r["task"], "code": r["code"]} for r in todo]
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        (t / "pack.json").write_text(json.dumps({"refs": {r["task"]: REFS[r["task"]] for r in todo}, "items": items}))
        subprocess.run([PY, ORACLE, str(t / "pack.json"), str(t / "static.jsonl"), "--static-only"], env={**os.environ, "JAX_PLATFORMS": "cpu"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        st = {x["id"]: x for x in bench.read_jsonl(t / "static.jsonl") if not x.get("meta")}
    fails = [st.get(i["id"], {"id": i["id"], "verdict": "harness_no_row", "feedback": "static check produced no row"}) for i in items
             if st.get(i["id"], {}).get("verdict") != "static_ok"]
    add_verdicts(fails)
    ok = [{"id": i["id"], "task": i["task"], "ref_code": REFS[i["task"]], "code": i["code"]} for i in items if st.get(i["id"], {}).get("verdict") == "static_ok"]
    (OPS / f"robust-shift-t{T}.items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ok))
    print(f"turn {T}: {len(items)} checked, {len(ok)} static_ok -> TPU, {len(fails)} failed step 1 {dict(collections.Counter(f['verdict'] for f in fails))}")
elif cmd == "ingest":
    T = int(sys.argv[2]); final = {}
    for p in sorted(glob.glob(str(OPS / f"repo/oracle/jobs/robust-shift-t{T}-*/results/*.jsonl"))):
        for r in bench.read_jsonl(pathlib.Path(p)):
            if r.get("meta"):
                continue
            if not halt(r) or r["id"] not in final:
                final[r["id"]] = r
    have = bench.verdicts(MODEL)
    new = [v for i, v in final.items() if i not in have]
    add_verdicts(new)
    print(f"turn {T}: {len(new)} TPU verdicts ingested {dict(collections.Counter(v['verdict'] for v in new))}")
elif cmd == "report":
    vs = bench.verdicts(MODEL); by = collections.defaultdict(list)
    for r in rows():
        v = vs.get(r["id"]); by[r["task"]].append((r["turn"], v["verdict"] if v else "pending", (v or {}).get("speed_vs_xla")))
    for shape in ("shapeA", "shapeB"):
        ts = sorted(t for t in by if t.endswith(shape)); p = sum(any(x[1] == "pass" for x in by[t]) for t in ts)
        print(f"== {shape} {IDX[ts[0]]['M']}x{IDX[ts[0]]['K']}x{IDX[ts[0]]['N']}: pass within 3 = {p}/{len(ts)}")
        for t in ts:
            print(f"   {t[:48]:48} " + " | ".join(f"t{a}:{b}{'' if not c else ' %.2fx' % c}" for a, b, c in sorted(by[t])))
