"""TPU items for the NEW Kimi problems (e1 re-admits, e2 kbkd, e3 scale variants), attempt R (1 = first attempt, 2 = repair round 2):
every kernel that passed oracle step 1 on CPU and has no TPU verdict yet from the kimi-n<R>* jobs (harness-crash kernels skipped,
crash suspects last), split N ways by task.   python3 build_items_n.py R N TAG  ->  fleet/<TAG>-s<K>.items.jsonl  (TAG e.g. kimi-n1, kimi-n1mop)"""
import glob, json, pathlib, re, sys, zlib
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
K, F, O = SC / "kimi_pilot", SC / "tpu/fleet", SC / "tpu/ops1"
R, N, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
D = K / "out/new" if R == 1 else K / f"out/new/rep{R}"
gglob = D / ("k31_*.jsonl" if R == 1 else f"k31_rep{R}_*.jsonl")
code = {}
for f in glob.glob(str(gglob)):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok") and d.get("code"):
            code[d["problem_id"]] = d["code"]
ref = {}
for f in (K / "new/new_refs.jsonl", K / "scale/scale_refs.jsonl"):
    for l in open(f):
        x = json.loads(l); ref[x["problem_id"]] = x["ref_code"]
judged = set()
for f in glob.glob(str(O / f"repo/oracle/jobs/kimi-n{R}*/results/*.jsonl")):
    for l in open(f):
        if l.strip() and '"meta": true' not in l:
            x = json.loads(l)
            if not (x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))):
                judged.add(x["id"].split("/", 1)[1])  # halt victims (an earlier kernel halted the core) count as unjudged
dead = set()
for f in glob.glob(str(O / f"kimi-n{R}*.crashes.json")):
    for k, v in json.load(open(f)).items():
        if "harness_crash" in v:
            dead.add(k.split("/", 1)[1])
sus = set()
for f in glob.glob(str(F / f"crashloop-kimi-n{R}*.log")):
    sus |= set(re.findall(rf"first item kimiN{R}/(\S+) set aside", open(f, errors="replace").read()))
ok = [json.loads(l) for l in open(D / "cpu_scored.jsonl")]
todo = [s for s in ok if s["static"] == "static_ok" and s["problem_id"] in code and s["problem_id"] in ref
        and s["problem_id"] not in judged and s["problem_id"] not in dead]
todo = list({s["problem_id"]: s for s in todo}.values())  # one item per problem even if step 1 scored it twice
groups = {k: [] for k in range(1, N + 1)}
for s in sorted(todo, key=lambda s: s["problem_id"] in sus):
    groups[zlib.crc32(s["task_name"].encode()) % N + 1].append(
        {"id": f"kimiN{R}/" + s["problem_id"], "task": s["task_name"], "ref_code": ref[s["problem_id"]], "code": code[s["problem_id"]]})
for k, items in groups.items():
    (F / f"{TAG}-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
print(f"{TAG}: {len(todo)} unjudged step-1 survivors -> per VM", {k: len(v) for k, v in groups.items()}, flush=True)
