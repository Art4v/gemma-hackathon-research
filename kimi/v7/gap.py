"""Kimi round 3 targets: old-pool problems whose task v4 trained on but v6 has no row for (the v4-covered gap). Writes r3/gap.json:
per problem the judged attempt chain [(code, verdict, feedback)] (a1 = kimi31r1, a2 = kimi31a2; CPU step-1 verdict if not static_ok,
else the TPU verdict, halt victims skipped) and the ref code recovered from the stored attempt-1 prompt (round-trip checked)."""
import collections, glob, hashlib, json, pathlib, re, sys
W = pathlib.Path.home() / "Projects/gemma-work"; K = W / "wt-kimi/kimi"; D = K / "data"
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from render_chat import render_problem  # noqa: E402
jl = lambda pat: [json.loads(l) for f in sorted(glob.glob(str(pat))) for l in open(f) if l.strip()]
probs = {p["problem_id"]: p for p in jl(D / "full_problems.jsonl")}
ref = {}
for pid, p in probs.items():
    u = p["user"]; i = u.index("```python\n"); r = u[i + len("```python\n"):-len("\n```")]
    for cand in (r + "\n", r):  # the problem file's ref ends with a newline; fence() strips it
        if render_problem(cand) == u and hashlib.sha256(cand.encode()).hexdigest() == p["ref_sha256"]:
            ref[pid] = cand; break
print("refs recovered", len(ref), "of", len(probs))
v4t = {r["task"] for r in jl(W / "data/v4_train.jsonl")}
v6 = jl(K / "v6/v6_train.jsonl"); v6p = {r["problem_id"] for r in v6} | {r.get("scaled_from") for r in v6 if r.get("scaled_from")}; v6t = {r["task"] for r in v6}
L = lambda f: {l.strip() for l in open(K / "v6" / f) if l.strip() and not l.startswith("#")}
X = L("exclude_hard.txt") | L("flag_unsure.txt")
AARAV = {"kernelbook/Sentence_Maxpool_14501", "kernelbook/MetaBilinear_711", "kernelbook/CPC_15255", "kernelbook/QMaxPooling2d_9618"}
vt = {v["task"] for v in jl(W / "data/v4_val.jsonl")}
valfam = lambda t: t in vt or any(t.startswith(v + "__") for v in vt) or re.sub(r"(__[a-z0-9_]+)+$", "", t) in vt
st = collections.Counter(); gap = []
for pid, p in probs.items():
    t = p["task_name"]
    if t not in v4t: st["not_in_v4"] += 1; continue
    if pid in v6p or t in v6t: st["in_v6"] += 1; continue
    if pid in X: st["flagged"] += 1; continue
    if pid in AARAV: st["aarav_drop"] += 1; continue
    if valfam(t): st["val_family"] += 1; continue
    if pid not in ref: st["no_ref"] += 1; continue
    gap.append(pid)
print(dict(st), "gap", len(gap))
tv = [d for d in jl(K / "tpu/tpu_verdicts.jsonl") if not d.get("halt_victim")]
tpu = {r: {d["problem_id"]: d for d in tv if d["id"].startswith(f"kimi31{r}/")} for r in ("r1", "a2")}
def att(gen, cpu, tp):
    code = {d["problem_id"]: d["code"] for d in jl(gen) if d.get("ok") and d.get("code")}
    cs = {d["problem_id"]: d for d in jl(cpu)}
    out = {}
    for pid, c in code.items():
        s = cs.get(pid)
        if s is None: out[pid] = (c, "unscored", ""); continue
        if s["static"] != "static_ok": out[pid] = (c, s["static"], s.get("static_feedback") or "")
        elif pid in tp: out[pid] = (c, tp[pid]["verdict"], tp[pid].get("feedback") or "")
        else: out[pid] = (c, "static_ok_no_tpu", "")
    return out
a1 = att(D / "out/k31_*.jsonl", D / "out/cpu_scored.jsonl", tpu["r1"]); a2 = att(D / "out/rep2/k31_rep2_*.jsonl", D / "out/rep2/cpu_scored.jsonl", tpu["a2"])
REPAIR = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}
cat = collections.Counter(); out = {}
for pid in gap:
    chain = []
    for a in (a1, a2):
        if pid in a: chain.append(a[pid])
        else: break
    last = chain[-1][1] if chain else "no_attempt"
    cat[(len(chain), last)] += 1
    out[pid] = {"task": probs[pid]["task_name"], "source": probs[pid]["source"], "family_id": probs[pid]["family_id"], "ref": ref[pid], "chain": chain}
for k, n in sorted(cat.items(), key=lambda x: -x[1]): print(k, n)
(W / "v7/r3/gap.json").write_text(json.dumps(out))
