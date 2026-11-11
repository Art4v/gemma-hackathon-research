"""Item 3 on issue #5: weight-scale v4-solved matmul problems that have no v6-strict row, in the kimi/v7 format.
Mirrors kimi/v7/build_kn.py; the kernel in each item is the problem's verified v4 target, unchanged.
  python build_v4kn.py KIMI_WT V4_TRAIN OUT_DIR JAX_PYTHON"""
import collections, concurrent.futures as cf, hashlib, json, pathlib, re, subprocess, sys

K, V4, OUT, PY = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, sys.argv[5])
from render_chat import render_problem  # noqa: E402

J = lambda p: [json.loads(l) for l in open(p)]
L = lambda f: {l.strip() for l in open(K / "kimi/v6" / f) if l.strip() and not l.startswith("#")}
SKIP = (L("exclude_hard.txt") | L("flag_unsure.txt") | L("rule3_families.txt") | L("flag_generic_stacks.txt") | L("flag_generic_stacks_new.txt")
        | L("a6_dependent_problem_ids.txt")
        | {"kernelbook/Sentence_Maxpool_14501", "kernelbook/MetaBilinear_711", "kernelbook/CPC_15255", "kernelbook/QMaxPooling2d_9618"})
pool = {}
for f in ("kimi/data/full_problems.jsonl", "kimi/data/new/new_problems.jsonl"):
    for r in J(K / f):
        pool[r["problem_id"]] = r
by_task = collections.defaultdict(list)
for p, r in pool.items():
    by_task[r["task_name"]].append(p)
in_v6 = {r["problem_id"] for r in J(K / "kimi/v6/v6_train.jsonl")}
fence = lambda c: c[len("```python\n"):-len("\n```")] if c.startswith("```python\n") and c.endswith("\n```") else c
RANK = {"single": 0, "shape_variant": 1, "repair": 2}  # all three end in a TPU-verified kernel for this exact reference
best, stat = {}, collections.Counter()
for r in J(V4):
    if r["kind"] in RANK and (r["task"] not in best or RANK[r["kind"]] < RANK[best[r["task"]]["kind"]]):
        best[r["task"]] = r
cand = []
for task, r in sorted(best.items()):
    pids = by_task.get(task, [])
    if len(pids) != 1:
        stat["no unique pool id"] += 1; continue
    pid = pids[0]
    if pid in in_v6:
        stat["has a v6 row"] += 1; continue
    if pid in SKIP:
        stat["flagged"] += 1; continue
    m = re.search(r"```python\n(.*?)```", pool[pid]["user"], re.S)
    v4ref = re.search(r"```python\n(.*?)```", next(x["content"] for x in r["messages"] if x["role"] == "user"), re.S)
    if not m or not v4ref or m.group(1).strip() != v4ref.group(1).strip():  # the renders differ by one trailing newline
        stat["pool ref differs from the ref v4 verified against"] += 1; continue
    cand.append((pid, m.group(1), fence(r["messages"][-1]["content"]), r))
print(len(cand), "candidates", dict(stat), flush=True)


def one(c):
    try:
        p = subprocess.run([PY, str(K / "kimi/v7/scale_kn_one.py")], input=json.dumps({"code": c[1]}), capture_output=True, text=True, timeout=300,
                           env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"})
        line = [x for x in p.stdout.splitlines() if x.startswith("{")]
        return c, (json.loads(line[-1]) if line else {"ok": False, "reason": "no output: " + p.stderr[-200:]})
    except subprocess.TimeoutExpired:
        return c, {"ok": False, "reason": "timeout"}


log, refs, probs, items, out = [], [], [], [], collections.Counter()
with cf.ThreadPoolExecutor(4) as ex:
    for (pid, code, kernel, row), d in ex.map(one, cand):
        log.append({"problem_id": pid, **{k: v for k, v in d.items() if k != "code"}})
        if not d.get("ok"):
            out[re.sub(r"\d+", "#", d.get("reason", "?"))[:50]] += 1; continue
        if d["factor_features"] == 1 and d["batch_divisor"] == 1:
            out["already exam scale"] += 1; continue
        if d.get("dot_share", 0) < 0.5:
            out["matmul share < 0.5"] += 1; continue
        out["scaled"] += 1
        u, nc, npid = pool[pid], d["code"], pid + "__kn"
        meta = {"problem_id": npid, "task_name": u["task_name"] + "__kn", "source": u["source"], "family_id": u["family_id"], "set": "e4_kn",
                "scaled_from": pid, "ref_sha256": hashlib.sha256(nc.encode()).hexdigest()}
        refs.append({**meta, "ref_code": nc, "kn": {k: d[k] for k in ("K0", "N0", "K", "N", "weight_mb", "factor_features", "batch_divisor",
                                                                     "rule_b", "dot_share", "bytes_old", "bytes_new")},
                     "orig_kind": row["kind"], "orig_speed": None, "kernel_from": "v4", "kernel_teacher": row["teacher"]})
        probs.append({**meta, "problem_file": "kimi/v7/v4only/kn_refs.jsonl", "user": render_problem(nc)})
        items.append({"id": "kimiKN0/" + npid, "task": meta["task_name"], "ref_code": nc, "code": kernel})
for name, rows in (("kn_log", log), ("kn_refs", refs), ("kn_problems", probs), ("kn_items_existing", items)):
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
print("DONE", dict(out), "| rule_b", sum(r["kn"]["rule_b"] for r in refs), "| K or N >= 8192:", sum(max(r["kn"]["K"], r["kn"]["N"]) >= 8192 for r in refs))
