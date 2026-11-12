"""TPU items for Kimi repair round R: every repair that passed oracle step 1 on CPU (out/rep<R>/cpu_scored.jsonl), as {id, task, ref_code,
code} with id kimi31a<R>/<problem_id>, split 8 ways by task into fleet/kimi-a<R>-s<K>.items.jsonl.   python3 build_rep_items.py R"""
import glob, json, pathlib, sys, zlib
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
R = int(sys.argv[1]); D = SC / f"kimi_pilot/out/rep{R}"; F = SC / "tpu/fleet"
code = {}
for f in glob.glob(str(D / f"k31_rep{R}_*.jsonl")):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok") and d.get("code"):
            code[d["problem_id"]] = d["code"]
probs = {json.loads(l)["problem_id"]: json.loads(l) for l in open(SC / "kimi_pilot/full_problems.jsonl")}
ref = {}
for pf in {p["problem_file"] for p in probs.values()}:
    for l in open(pf):
        x = json.loads(l)
        if x["problem_id"] in probs and x["problem_id"] not in ref:
            ref[x["problem_id"]] = x["ref_code"]
ok = [json.loads(l) for l in open(D / "cpu_scored.jsonl")]
ok = [s for s in ok if s["static"] == "static_ok" and s["problem_id"] in code]
groups = {k: [] for k in range(1, 9)}
for s in ok:
    groups[zlib.crc32(s["task_name"].encode()) % 8 + 1].append(
        {"id": f"kimi31a{R}/" + s["problem_id"], "task": s["task_name"], "ref_code": ref[s["problem_id"]], "code": code[s["problem_id"]]})
for k, items in groups.items():
    (F / f"kimi-a{R}-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
print(f"repair round {R}: {len(ok)} step-1 survivors -> per VM", {k: len(v) for k, v in groups.items()}, flush=True)
