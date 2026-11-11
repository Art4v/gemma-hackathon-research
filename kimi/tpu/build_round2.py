"""Round-2 TPU items: every Kimi K3.1 kernel that passed oracle step 1 on CPU (out/cpu_scored.jsonl, static_ok) and was not in round 1
(round1_ids.txt), as {id, task, ref_code, code}, split 8 ways by task."""
import json, glob, zlib, pathlib
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
F = SC / "tpu/fleet"
gen = {}
for f in glob.glob(str(SC / "kimi_pilot/out/k31_*.jsonl")):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok") and d.get("code"):
            gen[d["problem_id"]] = d
probs = {json.loads(l)["problem_id"]: json.loads(l) for l in open(SC / "kimi_pilot/full_problems.jsonl")}
ref = {}
for pf in {p["problem_file"] for p in probs.values()}:
    for l in open(pf):
        r = json.loads(l)
        if r["problem_id"] in probs and r["problem_id"] not in ref:
            ref[r["problem_id"]] = r["ref_code"]
# round-1 items count as done only if a TPU verdict row came back (Nov 11: vm5 lost its session after 27 of 47; those 20 go to round 2)
r1 = set()
for f in glob.glob(str(SC / "tpu/ops1/repo/oracle/jobs/kimi-k31-r1-s*-*/results/*.jsonl")):
    for l in open(f):
        if l.strip() and '"meta": true' not in l:
            r1.add(json.loads(l)["id"].split("/", 1)[1])
ok = [json.loads(l) for l in open(SC / "kimi_pilot/out/cpu_scored.jsonl")]
ok = [s for s in ok if s["static"] == "static_ok" and s["problem_id"] in gen and s["problem_id"] not in r1]
groups = {k: [] for k in range(1, 9)}
for s in ok:
    groups[zlib.crc32(s["task_name"].encode()) % 8 + 1].append(
        {"id": "kimi31r1/" + s["problem_id"], "task": s["task_name"], "ref_code": ref[s["problem_id"]], "code": gen[s["problem_id"]]["code"]})
for k, items in groups.items():
    (F / f"kimi-k31-r2-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
(F / "round2_ids.txt").write_text("".join(s["problem_id"] + "\n" for s in ok))
print(f"round 2: {len(ok)} step-1 survivors -> per VM", {k: len(v) for k, v in groups.items()}, flush=True)
