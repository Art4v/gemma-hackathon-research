"""Assemble the v6 training set from Kimi K3.1 kernels that the TPU oracle verified, in v4's exact row format (the student's prompt from
control/bench.py conversation(): system SYSTEM, the reference turn, the fenced code; repairs add the failed kernel and the oracle's
feedback + "Reply with the corrected Python code block." exactly as JAXBench's repair turns do). Code-only targets (no plan, no
reasoning): thinking False, as v4, so v6 is evaluated under the same protocol as every other arm.
Rows:
  single        attempt 1 verdict pass (TPU, oracle c9a07d9298a1)
  correct_slow  attempt 1 verdict correct_slow (kept, labelled; the trainer can drop them)
  repair        attempt 1 failed with a kernel-attributable verdict, attempt R (2 or 3) is pass or correct_slow (repair_slow if slow)
harness_oom / too_large_for_chip rows are never used. Every row records problem_id so the Kimi reasoning/plan can be joined back.
Two problem pools: "pool" (full_problems.jsonl, the 1,672 first-attempt problems; TPU jobs kimi-k31-* / kimi-a<r>*) and "new"
(Nov 11 night: e1 = the 133 class-B problems re-admitted under the instance-level rule, e2 = Karthik's kbkd batch, e3 = JAXBench-size
scale variants of solved unflagged pool problems; TPU jobs kimi-n<r>*). Every row carries its set (pool / e1_readmit / e2_kbkd / e3_scale).
  python3 assemble_v6.py  ->  v6/v6_train_draft.jsonl, v6/v6_stats.json"""
import collections, glob, json, pathlib, sys
P = pathlib.Path(__file__).resolve().parent
SC = P.parent
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
import bench  # noqa: E402  (conversation(), the eval's own prompt)

REPAIRABLE = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}


def jl(pattern):
    out = []
    for f in sorted(glob.glob(str(pattern))):
        for l in open(f):
            if l.strip():
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return out

def _halt(d):  # Nov 11: harness_oom rows caused by an earlier kernel halting the TPU core are not verdicts (see tpu/crashloop.py)
    return d.get("verdict") == "harness_oom" and any(h in (d.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))


def paths(pool, r):
    J = SC / "tpu/ops1/repo/oracle/jobs"
    if pool == "pool":
        return ((P / "out/k31_*.jsonl", P / "out/cpu_scored.jsonl", J / "kimi-k31-*/results/*.jsonl") if r == 1 else
                (P / f"out/rep{r}/k31_rep{r}_*.jsonl", P / f"out/rep{r}/cpu_scored.jsonl", J / f"kimi-a{r}*/results/*.jsonl"))
    return ((P / "out/new/k31_*.jsonl", P / "out/new/cpu_scored.jsonl", J / "kimi-n1*/results/*.jsonl") if r == 1 else
            (P / f"out/new/rep{r}/k31_rep{r}_*.jsonl", P / f"out/new/rep{r}/cpu_scored.jsonl", J / f"kimi-n{r}*/results/*.jsonl"))


def attempt(pool, r):
    gen, cpu, tpu = paths(pool, r)
    code = {d["problem_id"]: d["code"] for d in jl(gen) if d.get("ok") and d.get("code")}
    cpu = {d["problem_id"]: d for d in jl(cpu)}
    tpu = {d["id"].split("/", 1)[1]: d for d in jl(tpu) if "id" in d and not d.get("meta") and not _halt(d)}
    out = {}
    for pid, c in code.items():
        s = cpu.get(pid)
        if s is None:
            continue
        if s["static"] != "static_ok":
            out[pid] = {"code": c, "verdict": s["static"], "feedback": s.get("static_feedback") or "", "speed": None}
        elif pid in tpu:
            t = tpu[pid]
            out[pid] = {"code": c, "verdict": t.get("verdict"), "feedback": t.get("feedback") or "", "speed": t.get("speed_vs_xla")}
    return out


probs = {json.loads(l)["problem_id"]: {**json.loads(l), "pool": "pool"} for l in open(P / "full_problems.jsonl")}
for f in ("new/new_problems.jsonl", "scale/scale_problems.jsonl"):
    if (P / f).exists():
        for l in open(P / f):
            x = json.loads(l)
            assert x["problem_id"] not in probs, x["problem_id"]
            probs[x["problem_id"]] = {**x, "pool": "new"}
ref = {}
for pf in {p["problem_file"] for p in probs.values()}:
    for l in open(pf):
        x = json.loads(l)
        if x["problem_id"] in probs and x["problem_id"] not in ref:
            ref[x["problem_id"]] = x["ref_code"]
A = {(pool, r): attempt(pool, r) for pool in ("pool", "new") for r in (1, 2, 3)}
_lst = lambda f: {l.strip() for l in open(P / "v6" / f) if l.strip() and not l.startswith("#")}
HARD, RULE3 = _lst("exclude_hard.txt"), _lst("rule3_families.txt")
UNSURE, STACKS = _lst("flag_unsure.txt"), _lst("flag_generic_stacks.txt") | (_lst("flag_generic_stacks_new.txt") if (P / "v6/flag_generic_stacks_new.txt").exists() else set())
fence = lambda c: "```python\n" + c.rstrip("\n") + "\n```"
rows, stats = [], collections.Counter()
for pid, p in probs.items():
    a1 = A[(p["pool"], 1)].get(pid)
    if not a1:
        stats["no_attempt1_verdict"] += 1
        continue
    if pid in HARD:
        stats["hard_excluded"] += 1
        continue
    if pid in UNSURE:  # default OUT until Hruday decides the 'unsure' tier
        stats["unsure_excluded"] += 1
        continue
    base = {"source": p["source"], "task": p["task_name"], "problem_id": pid, "teacher": "kimi-k3.1", "thinking": False,
            "rule3_family": pid in RULE3, "generic_stack": pid in STACKS, "set": p.get("set") or "pool",
            "scaled_from": p.get("scaled_from")}
    if a1["verdict"] in ("pass", "correct_slow"):
        kind = "single" if a1["verdict"] == "pass" else "correct_slow"
        msgs = bench.conversation(ref[pid], [], None) + [{"role": "assistant", "content": fence(a1["code"])}]
        rows.append({"example_id": f"{p['task_name']}.kimi.a1", "kind": kind, "messages": msgs, "verdict": a1["verdict"],
                     "speed_vs_xla": a1["speed"], **base})
        stats[kind] += 1
        continue
    if a1["verdict"] not in REPAIRABLE:
        stats["a1_" + str(a1["verdict"])] += 1
        continue
    hist, fixed = [{"code": a1["code"], "feedback": a1["feedback"]}], None
    for r in (2, 3):
        ar = A[(p["pool"], r)].get(pid)
        if not ar:
            break
        if ar["verdict"] in ("pass", "correct_slow"):
            fixed = (r, ar)
            break
        if ar["verdict"] not in REPAIRABLE:
            break
        hist.append({"code": ar["code"], "feedback": ar["feedback"]})
    if fixed is None:
        stats["unsolved_so_far"] += 1
        continue
    r, ar = fixed
    kind = "repair" if ar["verdict"] == "pass" else "repair_slow"
    msgs = bench.conversation(ref[pid], hist, None) + [{"role": "assistant", "content": fence(ar["code"])}]
    rows.append({"example_id": f"{p['task_name']}.kimi.a{r}", "kind": kind, "messages": msgs, "verdict": ar["verdict"],
                 "speed_vs_xla": ar["speed"], "repair_from": [h["feedback"][:120] for h in hist], **base})
    stats[kind] += 1
out = P / "v6"; out.mkdir(exist_ok=True)
(out / "v6_train_draft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
by_src = collections.Counter((r["source"], r["kind"]) for r in rows)
by_set = collections.Counter((r["set"], r["kind"]) for r in rows)
(out / "v6_stats.json").write_text(json.dumps({"rows": len(rows), "kinds": dict(stats), "by_source_kind": {f"{a}/{b}": n for (a, b), n in by_src.items()},
                                                 "by_set_kind": {f"{a}/{b}": n for (a, b), n in sorted(by_set.items())}}, indent=1))
print(f"v6 draft: {len(rows)} rows; {dict(stats)}")
print("by source:", dict(collections.Counter(r["source"] for r in rows)))
print("by set/kind:", dict(sorted(by_set.items())))
