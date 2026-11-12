"""Mop-up TPU items: every Kimi K3.1 kernel that passed oracle step 1 on CPU but has no TPU verdict from round 1 or round 2 and is not
recorded as harness_crash; kernels a crashloop set aside as crash suspects go LAST in their batch (a crash can wedge a VM). Split
across N VMs by task.   python3 build_mopup.py N"""
import glob, json, pathlib, re, sys, zlib
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
F, O = SC / "tpu/fleet", SC / "tpu/ops1"
N = int(sys.argv[1])
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
judged = set()
for f in glob.glob(str(O / "repo/oracle/jobs/kimi-k31-r[12]-s*-*/results/*.jsonl")):
    for l in open(f):
        if l.strip() and '"meta": true' not in l:
            judged.add(json.loads(l)["id"].split("/", 1)[1])
dead = set()
for f in glob.glob(str(O / "kimi-k31-r*.crashes.json")):
    for k, v in json.load(open(f)).items():
        if "harness_crash" in v:
            dead.add(k.split("/", 1)[1])
suspects = set()
for f in glob.glob(str(F / "crashloop-kimi-k31-r*.log")):
    for m in re.finditer(r"first item kimi31r1/(\S+) set aside", open(f, errors="replace").read()):
        suspects.add(m.group(1))
ok = [json.loads(l) for l in open(SC / "kimi_pilot/out/cpu_scored.jsonl")]
todo = [s for s in ok if s["static"] == "static_ok" and s["problem_id"] in gen and s["problem_id"] not in judged and s["problem_id"] not in dead]
groups = {k: [] for k in range(1, N + 1)}
for s in sorted(todo, key=lambda s: s["problem_id"] in suspects):  # suspects last
    groups[zlib.crc32(s["task_name"].encode()) % N + 1].append(
        {"id": "kimi31r1/" + s["problem_id"], "task": s["task_name"], "ref_code": ref[s["problem_id"]], "code": gen[s["problem_id"]]["code"]})
for k, items in groups.items():
    (F / f"kimi-k31-mop-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
print(f"mop-up: {len(todo)} unjudged step-1 survivors ({len(suspects & {s['problem_id'] for s in todo})} known crash suspects placed last;"
      f" {len(dead)} harness_crash excluded) -> per VM", {k: len(v) for k, v in groups.items()}, flush=True)
