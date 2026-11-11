"""Mop-up items for Kimi repair round R: every repair that passed oracle step 1 on CPU but has no TPU verdict from the kimi-a<R> jobs,
crash suspects last; split across N VMs.   python3 build_mopup_rep.py R N  ->  fleet/kimi-a<R>mop-s<K>.items.jsonl"""
import glob, json, pathlib, re, sys, zlib
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
F, O, D = SC / "tpu/fleet", SC / "tpu/ops1", None
R, N = int(sys.argv[1]), int(sys.argv[2]); D = SC / f"kimi_pilot/out/rep{R}"
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
judged = set()
for f in glob.glob(str(O / f"repo/oracle/jobs/kimi-a{R}*/results/*.jsonl")):
    for l in open(f):
        if l.strip() and '"meta": true' not in l:
            x = json.loads(l)
            if not (x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))):
                judged.add(x["id"].split("/", 1)[1])  # halt victims (an earlier kernel halted the core) count as unjudged
dead = set()
for f in glob.glob(str(O / f"kimi-a{R}*.crashes.json")):
    for k, v in json.load(open(f)).items():
        if "harness_crash" in v:
            dead.add(k.split("/", 1)[1])
sus = set()
for f in glob.glob(str(F / f"crashloop-kimi-a{R}-*.log")):
    sus |= set(re.findall(rf"first item kimi31a{R}/(\S+) set aside", open(f, errors="replace").read()))
ok = [json.loads(l) for l in open(D / "cpu_scored.jsonl")]
todo = [s for s in ok if s["static"] == "static_ok" and s["problem_id"] in code and s["problem_id"] not in judged and s["problem_id"] not in dead]
groups = {k: [] for k in range(1, N + 1)}
for s in sorted(todo, key=lambda s: s["problem_id"] in sus):
    groups[zlib.crc32(s["task_name"].encode()) % N + 1].append(
        {"id": f"kimi31a{R}/" + s["problem_id"], "task": s["task_name"], "ref_code": ref[s["problem_id"]], "code": code[s["problem_id"]]})
for k, items in groups.items():
    (F / f"kimi-a{R}mop-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
print(f"repair round {R} mop-up: {len(todo)} unjudged survivors -> per VM", {k: len(v) for k, v in groups.items()}, flush=True)
