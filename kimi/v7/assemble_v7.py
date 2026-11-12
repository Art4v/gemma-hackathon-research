"""Assemble the v7 training set. Base = v6 (A6 file, Kimi targets only; Hruday: no Claude-written targets; A6 adopted Nov 12) minus Aarav's 4
filter drops, plus rows whose every target the TPU judged pass / correct_slow (oracle c9a07d9298a1):
  e4_kn     scaled variant whose EXISTING verified v6 kernel passed / was correct at exam size (kimi-kn0)  -> single / correct_slow
  e4_fix    Kimi's fix round 1 (kimi-kn1): repair chain [(v6 kernel, kn0 feedback)] -> fix, AND a turn-0 row (reference -> fix)
  e4_fix2   Kimi's fix round 2 (kimi-kn2): repair chain [(v6 kernel, kn0 fb), (fix1, its fb)?] -> fix2, AND a turn-0 row
            Per scaled problem ONE solution is kept: pass beats correct_slow, then the higher speed (a fast fix2 replaces a slow kn0 row).
  e5_gen    Aarav's generated exam-scale rows (template generator, TPU pass >= 0.7x; aarav/dataset-v4 @ de8f43a)
  r3_gap    Kimi round 3 on the v4-covered gap (kimi-r3): old-pool problems v4 trained on with no v6 row; chain [(a1, fb1), (a2, fb2)] -> a3
            (repair / repair_slow) AND a turn-0 row (reference -> a3), same as the exam fixes.
Karthik's 16 v4-only scaled items are excluded (Claude-written kernels).
  --drop-slow none|matmul|all   drop v6 correct_slow/repair_slow rows (matmul = only rows whose problem is matmul-dominant)
  --r3-slow keep|drop           keep or drop round-3 correct_slow rows
  --min-speed X                 drop exam-scale rows (e4_*) whose verified speed is below X (pass rows are always >= 0.7)
  --repeat-exam N               write every exam-scale row N times (sampling weight)
  --dev FILE                    problem-level holdout: drop every row of these problems (original, scaled, fixed)
  python3 assemble_v7.py [flags]  ->  v7/v7_train.jsonl, v7/v7_stats.json"""
import argparse, collections, glob, hashlib, json, pathlib, re, sys
V7 = pathlib.Path(__file__).resolve().parent
W = pathlib.Path.home() / "Projects/gemma-work"
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
import bench  # noqa: E402  (conversation(): the eval's own prompt)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(W / "wt-kimi/kimi/v6/v6_train.jsonl"))
ap.add_argument("--drop-slow", default="none", choices=["none", "matmul", "all"]); ap.add_argument("--min-speed", type=float, default=0.0)
ap.add_argument("--r3-slow", default="keep", choices=["keep", "drop"])
ap.add_argument("--exam-slow", default="keep", choices=["keep", "drop"])  # drop = an added row may not have a below-bar TARGET (slow kernels stay only as failed turns)
ap.add_argument("--dev-gen", default=str(V7 / "dev_items_gen.jsonl"))  # the shared dev set (aarav/dataset-v4 @ cdea3a6, 27 generated items, 24 whole chains)
ap.add_argument("--out", default="v7_train")
ap.add_argument("--a6r2", default=str(V7 / "a6r2/drops.json"))  # A6 rule 2 (conservative, Aarav's test) matches among OUR added problems
ap.add_argument("--a6r2-gen", default=str(V7 / "a6r2/aarav_drops.json"))  # Aarav's list for the generated rows (aarav/dataset-v4 @ ad6b273)
ap.add_argument("--dot-share-fill")  # opt-in: dot_share_fill.jsonl, the dot_share of slow A6 rows build_kn never measured (so --drop-slow matmul sees them)
ap.add_argument("--repeat-exam", type=int, default=1); ap.add_argument("--dev"); ap.add_argument("--gen", default=str(V7 / "v7_gen_rows.jsonl"))
a = ap.parse_args()
R2 = set(json.load(open(a.a6r2))["drops"]) if a.a6r2 and pathlib.Path(a.a6r2).exists() else set()
R2B = set(json.load(open(V7 / "a6r2/base_drops.json"))["drops"])  # the same A6 rule-2 test on the base (02:55Z decision)
R2G = set(json.load(open(a.a6r2_gen))["drop_generated_training_example_ids"]) if a.a6r2_gen and pathlib.Path(a.a6r2_gen).exists() else set()
fence = lambda c: "```python\n" + c.rstrip("\n") + "\n```"
halt = lambda x: x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))
AARAV_DROPS = {"kernelbook/Sentence_Maxpool_14501", "kernelbook/MetaBilinear_711", "kernelbook/CPC_15255", "kernelbook/QMaxPooling2d_9618"}
dev = {l.strip() for l in open(a.dev) if l.strip() and not l.startswith("#")} if a.dev else set()
base_of = lambda pid: re.sub(r"(__kn8?|__kn)$", "", pid)
OK = ("pass", "correct_slow")


def verdicts(job_glob):
    out = {}
    for f in sorted(glob.glob(str(W / f"tpu/ops1/repo/oracle/jobs/{job_glob}/results/*.jsonl"))):
        for l in open(f):
            if l.strip() and '"meta": true' not in l:
                x = json.loads(l)
                if not halt(x):
                    out[x["id"]] = x
    return out


def gen_rows(pattern):
    out = {}
    for f in glob.glob(str(V7 / pattern)):
        for l in open(f):
            d = json.loads(l)
            if d.get("ok") and d.get("code"):
                out[d["problem_id"]] = d
    return out


stats = collections.Counter(); rows = []
# ---- base: v6 (A6)
mm = {json.loads(l)["problem_id"]: json.loads(l).get("dot_share", 0) for l in open(V7 / "kn_log.jsonl")}
MMLED = set(json.load(open(V7 / "a6r2/mm_led_slow_problems.json")))  # first heavy op is a matmul (traced; kn_log dot_share covered only scale candidates)
if a.dot_share_fill:  # kn_log.jsonl only covers build_kn's candidates (v6 strict, no e3_scale, unflagged); a slow row outside it read as 0
    mm.update({x["problem_id"]: x["dot_share"] for x in map(json.loads, open(a.dot_share_fill)) if x["problem_id"] not in mm})
for l in open(a.base):
    r = json.loads(l); pid = r["problem_id"]
    if pid in AARAV_DROPS:
        stats["v6_aarav_drop"] += 1; continue
    if pid in R2B:
        stats["v6_a6_rule2_dropped"] += 1; continue
    if base_of(pid) in dev or pid in dev:
        stats["v6_dev_holdout"] += 1; continue
    slow = r["kind"] in ("correct_slow", "repair_slow")
    if slow and (a.drop_slow == "all" or (a.drop_slow == "matmul" and (mm.get(pid, 0) >= 0.5 or pid in MMLED))):
        stats["v6_slow_dropped"] += 1; continue
    rows.append(r); stats["v6_" + r["kind"]] += 1
# ---- exam scale: scaled variants (kn0), Kimi fixes rounds 1 (kn1) and 2 (kn2); one solution per scaled problem
items = {}
for f in ("kn_items_existing.jsonl", "kn_items_existing_8.jsonl", "kn_items_existing_p.jsonl"):
    for l in open(V7 / f):
        x = json.loads(l); items[x["id"]] = x
refs = {}
for f in ("kn_refs.jsonl", "kn_refs_8.jsonl", "kn_refs_p.jsonl"):
    for l in open(V7 / f):
        x = json.loads(l); refs[x["problem_id"]] = x
k0, k1, k2 = verdicts("kimi-kn0-*"), verdicts("kimi-kn1-*"), verdicts("kimi-kn2-*")
fix1, fix2 = gen_rows("out/fix1/k31_fix1_*.jsonl"), gen_rows("out/fix2/k31_fix2_*.jsonl")
exam = []
for iid, it in items.items():
    pid = iid.split("/", 1)[1]; meta = refs[pid]
    if base_of(pid) in dev or meta["scaled_from"] in dev:
        stats["exam_dev_holdout"] += 1; continue
    if pid in R2:
        stats["exam_a6_rule2_dropped"] += 1; continue
    base = {"source": meta["source"], "task": it["task"], "problem_id": pid, "teacher": "kimi-k3.1", "thinking": False,
            "scaled_from": meta["scaled_from"], "kn": meta["kn"], "rule3_family": False, "generic_stack": False}
    ref0 = bench.conversation(it["ref_code"], [], None)
    cands = []  # (verdict, speed, set, history, code, attempt)
    v0 = k0.get(iid)
    if v0 and v0["verdict"] in OK:
        cands.append((v0, "e4_kn", [], it["code"], 1))
    v1 = k1.get("kimiKN1/" + pid)
    if v0 and v1 and v1["verdict"] in OK and pid in fix1:
        cands.append((v1, "e4_fix", [{"code": it["code"], "feedback": v0["feedback"]}], fix1[pid]["code"], 2))
    v2 = k2.get("kimiKN2/" + pid)
    if v2 and v2["verdict"] in OK and pid in fix2:
        cands.append((v2, "e4_fix2", fix2[pid]["history"], fix2[pid]["code"], len(fix2[pid]["history"]) + 1))
    if not cands:
        stats["exam_unsolved"] += 1; continue
    if a.exam_slow == "drop":
        cands = [c for c in cands if c[0]["verdict"] == "pass"]
        if not cands:
            stats["exam_slow_only_dropped"] += 1; continue
    v, s, hist, code, att = max(cands, key=lambda c: (c[0]["verdict"] == "pass", c[0].get("speed_vs_xla") or 0))
    if (v.get("speed_vs_xla") or 0) < a.min_speed:
        stats["exam_below_min_speed"] += 1; continue
    slow = v["verdict"] == "correct_slow"; b2 = {**base, "set": s, "verdict": v["verdict"], "speed_vs_xla": v.get("speed_vs_xla")}
    if not hist:
        exam.append({"example_id": f"{it['task']}.kimi.a1", "kind": "correct_slow" if slow else "single",
                     "messages": ref0 + [{"role": "assistant", "content": fence(code)}], **b2})
    else:
        exam.append({"example_id": f"{it['task']}.kimi.a{att}", "kind": "repair_slow" if slow else "repair",
                     "repair_from": [h["feedback"][:120] for h in hist],
                     "messages": bench.conversation(it["ref_code"], hist, None) + [{"role": "assistant", "content": fence(code)}], **b2})
        exam.append({"example_id": f"{it['task']}.kimi.a{att}t0", "kind": "correct_slow" if slow else "single",
                     "messages": ref0 + [{"role": "assistant", "content": fence(code)}], **b2})
    stats[f"exam_{s}_{'slow' if slow else 'pass'}"] += 1
# ---- generated rows (Aarav's template generator), TPU pass only
gen = [json.loads(l) for l in open(a.gen)] if pathlib.Path(a.gen).exists() else []
devgen = {json.loads(l)["example_id"] for l in open(a.dev_gen)} if a.dev_gen and pathlib.Path(a.dev_gen).exists() else set()
for g in gen:
    if g["example_id"] in devgen or g.get("surface_of") in devgen:  # surface_of: a surface_gen.py variant is held out with its original
        stats["gen_dev_holdout"] += 1; continue
    if g["example_id"] in R2G or g.get("surface_of") in R2G:
        stats["gen_a6_rule2_dropped"] += 1; continue
    exam.append({**g, "set": "e5_gen"}); stats["exam_gen"] += 1
# ---- round 3 on the v4-covered gap
gap = json.load(open(V7 / "r3/gap.json")) if (V7 / "r3/gap.json").exists() else {}
k3, a3 = verdicts("kimi-r3-*"), gen_rows("out/r3/k31_r3_*.jsonl")
for pid, d in sorted(a3.items()):
    v = k3.get("kimiR3/" + pid)
    if not v or v["verdict"] not in OK:
        stats["r3_" + (v["verdict"] if v else "no_tpu_verdict")] += 1; continue
    if pid in dev or pid in AARAV_DROPS:
        stats["r3_dropped"] += 1; continue
    if pid in R2:
        stats["r3_a6_rule2_dropped"] += 1; continue
    slow = v["verdict"] == "correct_slow"
    if slow and (a.r3_slow == "drop" or a.exam_slow == "drop"):
        stats["r3_slow_dropped"] += 1; continue
    g = gap[pid]; hist = [{"code": c[0], "feedback": c[2]} for c in g["chain"]]; att = len(hist) + 1
    b3 = {"source": g["source"], "task": g["task"], "problem_id": pid, "teacher": "kimi-k3.1", "thinking": False, "set": "r3_gap",
          "scaled_from": None, "rule3_family": False, "generic_stack": False, "verdict": v["verdict"], "speed_vs_xla": v.get("speed_vs_xla")}
    exam.append({"example_id": f"{g['task']}.kimi.a{att}", "kind": "repair_slow" if slow else "repair", "repair_from": [h["feedback"][:120] for h in hist],
                 "messages": bench.conversation(g["ref"], hist, None) + [{"role": "assistant", "content": fence(d["code"])}], **b3})
    exam.append({"example_id": f"{g['task']}.kimi.a{att}t0", "kind": "correct_slow" if slow else "single",
                 "messages": bench.conversation(g["ref"], [], None) + [{"role": "assistant", "content": fence(d["code"])}], **b3})
    stats["r3_" + ("slow" if slow else "pass")] += 1
# eval form (Karthik's audit 6813213833): a history turn is what bench.py sends back, "```python\n" + extract_code(answer) + "\n```" (extract_code
# keeps the code's trailing newline). Final targets: one newline before the closing fence for every source (generator rows had two).
def eval_form(r):
    m = [dict(x) for x in r["messages"]]
    for i in range(2, len(m) - 1):
        if m[i]["role"] == "assistant":
            c = bench.extract_code(m[i]["content"])
            if c:
                m[i]["content"] = "```python\n" + c.rstrip("\n") + "\n" + "\n```"  # = what bench.py re-sends for a normally fenced answer
    t = bench.extract_code(m[-1]["content"])
    if t:
        m[-1]["content"] = "```python\n" + t.rstrip("\n") + "\n```"
    return {**r, "messages": m}
rows = [eval_form(r) for r in rows]; exam = [eval_form(r) for r in exam]
stats["added_rows_unique"] = len(exam)
rows += exam * max(1, a.repeat_exam)
ids = [r["example_id"] for r in rows]
dups = len(ids) - len(set(ids))
body = "".join(json.dumps(r) + "\n" for r in rows)
(V7 / f"{a.out}.jsonl").write_text(body)
val = [eval_form({**g, "set": "e5_gen_dev"}) for g in gen if g["example_id"] in devgen]  # the shared dev set as rows: pass as the trainer's --val
(V7 / f"{a.out}_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
stats["val_rows"] = len(val)
sha = hashlib.sha256(body.encode()).hexdigest()
by = collections.Counter((r.get("set", "pool"), r["kind"]) for r in rows)
(V7 / f"{a.out}_stats.json").write_text(json.dumps({"rows": len(rows), "sha256": sha, "flags": vars(a), "counts": dict(stats),
                                              "by_set_kind": {f"{s}/{k}": n for (s, k), n in sorted(by.items())}, "duplicate_example_ids": dups}, indent=1))
print(f"v7: {len(rows)} rows (sha {sha[:12]}), duplicate example_ids {dups} (expected only from --repeat-exam)")
print(dict(stats))
