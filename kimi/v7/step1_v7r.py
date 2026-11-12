"""CPU step 1 (tpu_oracle.py --static-only, oracle sha c9a07d9298a1, JAX 0.10.2, the dataset venv) for a v7 Kimi round, then TPU items for every
static_ok kernel, split N ways for the fleet.
  python3 step1_v7r.py fix2 N   waits for every out/fix2/gen_s*.log to say FIX2 GEN DONE; ids kimiKN2/<pid>; items tpu/fleet/kimi-kn2-s<K>.items.jsonl
  python3 step1_v7r.py r3 N     waits for out/r3/GEN_FINAL (touched once generation is complete, top-ups included); ids kimiR3/<pid>;
                                items tpu/fleet/kimi-r3-s<K>.items.jsonl
Outputs out/<set>/cpu_scored.jsonl."""
import glob, json, os, pathlib, subprocess, sys, tempfile, time, zlib
V7 = pathlib.Path(__file__).resolve().parent; F = pathlib.Path.home() / "Projects/gemma-work/tpu/fleet"
PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"
ORACLE = "/Users/hrudaynara/Projects/gemma-hackathon-research/oracle/tpu_oracle.py"
SET, N = sys.argv[1], int(sys.argv[2])
D = V7 / "out" / SET
if SET == "fix2":
    while not (logs := glob.glob(str(D / "gen_s*.log"))) or not all("FIX2 GEN DONE" in open(f).read() for f in logs):
        time.sleep(30)
    items = {}
    for f in ("kn_items_existing.jsonl", "kn_items_existing_8.jsonl", "kn_items_existing_p.jsonl"):
        for l in open(V7 / f):
            x = json.loads(l); items[x["id"]] = x
    gen = {}
    for f in glob.glob(str(D / "k31_fix2_*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d.get("ok") and d.get("code") and d["id"] in items:
                gen[d["problem_id"]] = {"task": d["task"], "ref": items[d["id"]]["ref_code"], "code": d["code"]}
    PFX, JOB = "kimiKN2/", "kimi-kn2"
else:
    while not (D / "GEN_FINAL").exists():
        time.sleep(30)
    gap = json.load(open(V7 / "r3/gap.json"))
    gen = {}
    for f in glob.glob(str(D / "k31_r3_*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d.get("ok") and d.get("code"):
                gen[d["problem_id"]] = {"task": gap[d["problem_id"]]["task"], "ref": gap[d["problem_id"]]["ref"], "code": d["code"]}
    PFX, JOB = "kimiR3/", "kimi-r3"
print(SET, len(gen), "kernels with code", flush=True)
rows = []; ids = sorted(gen)
for i in range(0, len(ids), 40):
    batch = ids[i:i + 40]
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        (t / "pack.json").write_text(json.dumps({"refs": {gen[x]["task"]: gen[x]["ref"] for x in batch},
                                                 "items": [{"id": PFX + x, "task": gen[x]["task"], "code": gen[x]["code"]} for x in batch]}))
        subprocess.run([PY, ORACLE, str(t / "pack.json"), str(t / "static.jsonl"), "--static-only"], env={**os.environ, "JAX_PLATFORMS": "cpu"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        st = {}
        if (t / "static.jsonl").exists():
            for l in (t / "static.jsonl").read_text().splitlines():
                if '"meta": true' not in l:
                    x = json.loads(l); st[x["id"]] = x
    for x in batch:
        s = st.get(PFX + x, {"verdict": "harness_no_row", "feedback": "static check produced no row"})
        rows.append({"id": PFX + x, "problem_id": x, "task": gen[x]["task"], "static": s.get("verdict"), "static_feedback": (s.get("feedback") or "")[:400],
                     "oracle_sha": s.get("oracle_sha")})
    print(f"step 1: {len(rows)}/{len(ids)} scored, static_ok so far {sum(r['static'] == 'static_ok' for r in rows)}", flush=True)
(D / "cpu_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
ok = [r for r in rows if r["static"] == "static_ok"]
groups = {k: [] for k in range(1, N + 1)}
for r in ok:
    groups[zlib.crc32(r["task"].encode()) % N + 1].append({"id": r["id"], "task": r["task"], "ref_code": gen[r["problem_id"]]["ref"],
                                                          "code": gen[r["problem_id"]]["code"]})
for k, g in groups.items():
    (F / f"{JOB}-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in g))
print(f"STEP1 DONE {SET}: {len(ok)} of {len(rows)} pass step 1 -> TPU items per VM", {k: len(g) for k, g in groups.items()}, flush=True)
