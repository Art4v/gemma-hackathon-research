"""v7 weight-scaled problems ("__kn"): solved, unflagged, matmul-dominant v6-strict problems whose main matmul is re-sized to JAXBench k-task scale
(max(K, N) 4096-16384, weight >= 64 MB, inputs <= 1.2 GB) by scale_kn_one.py. Also writes the TPU items for step (a): each problem's EXISTING verified
kernel (its v6 target code) run unchanged at the new size, so a pass is an exam-scale row for free and a vmem verdict is a real repair chain.
  python3 build_kn.py [--only PID ...] [--workers 8]
Outputs in ~/Projects/gemma-work/v7/: kn_log.jsonl, kn_refs.jsonl, kn_problems.jsonl (full_problems format, render-1), kn_items_existing.jsonl,
kn_dev_ids.txt (held-out dev slice, never trained)."""
import argparse, collections, concurrent.futures as cf, hashlib, json, pathlib, re, subprocess, sys, zlib
V7 = pathlib.Path(__file__).resolve().parent
W = pathlib.Path.home() / "Projects/gemma-work/wt-kimi/kimi"
PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from render_chat import render_problem  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--kn-min", type=int, default=4096); ap.add_argument("--suffix", default="__kn"); ap.add_argument("--tag", default=""); a = ap.parse_args()
L = lambda f: {l.strip() for l in open(W / "v6" / f) if l.strip() and not l.startswith("#")}
FLAG = L("exclude_hard.txt") | L("flag_unsure.txt") | L("rule3_families.txt") | L("flag_generic_stacks.txt") | L("flag_generic_stacks_new.txt")
AARAV_DROPS = {"kernelbook/Sentence_Maxpool_14501", "kernelbook/MetaBilinear_711", "kernelbook/CPC_15255", "kernelbook/QMaxPooling2d_9618"}
user = {}
for f in (W / "data/full_problems.jsonl", W / "data/new/new_problems.jsonl"):
    for l in open(f):
        r = json.loads(l); user[r["problem_id"]] = r
rows = [json.loads(l) for l in open(W / "v6/v6_train_strict.jsonl")]
fence = lambda c: c[len("```python\n"):-len("\n```")] if c.startswith("```python\n") and c.endswith("\n```") else c
cand = []
for r in rows:
    pid = r["problem_id"]
    if r["set"] == "e3_scale" or pid in FLAG or pid in AARAV_DROPS or pid not in user:
        continue
    if a.only and pid not in a.only:
        continue
    m = re.search(r"```python\n(.*?)```", user[pid]["user"], re.S)
    if m:
        cand.append((pid, m.group(1), fence(r["messages"][-1]["content"]), r))
print(len(cand), "candidate problems (v6 strict, not scale, unflagged)", flush=True)


def one(c):
    pid, code, kernel, row = c
    try:
        p = subprocess.run([PY, str(V7 / "scale_kn_one.py")], input=json.dumps({"code": code}), capture_output=True, text=True, timeout=300,
                           env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin", "KN_MIN": str(a.kn_min), "KN_SUFFIX": a.suffix})
        line = [x for x in p.stdout.splitlines() if x.startswith("{")]
        return c, (json.loads(line[-1]) if line else {"ok": False, "reason": "no output: " + p.stderr[-200:]})
    except subprocess.TimeoutExpired:
        return c, {"ok": False, "reason": "timeout"}


log, refs, probs, items, stat = [], [], [], [], collections.Counter()
with cf.ThreadPoolExecutor(a.workers) as ex:
    for n, ((pid, code, kernel, row), d) in enumerate(ex.map(one, cand), 1):
        log.append({"problem_id": pid, **{k: v for k, v in d.items() if k != "code"}})
        stat["ok" if d.get("ok") else ("not_matmul" if "matmul" in d.get("reason", "") else "fail")] += 1
        if n % 50 == 0:
            print(n, dict(stat), flush=True)
        if d.get("ok") and d.get("factor_features") == 1 and d.get("batch_divisor") == 1:
            stat["already_exam_scale"] += 1; stat["ok"] -= 1  # its v6 row is already an exam-scale example; a copy adds nothing
            continue
        if not d.get("ok") or d.get("dot_share", 0) < 0.5:
            if d.get("ok"):
                stat["ok_but_matmul_share<0.5"] += 1; stat["ok"] -= 1
            continue
        u = user[pid]; nc = d["code"]; npid = pid + a.suffix
        meta = {"problem_id": npid, "task_name": u["task_name"] + a.suffix, "source": u["source"], "family_id": u["family_id"], "set": "e4_kn",
                "scaled_from": pid, "ref_sha256": hashlib.sha256(nc.encode()).hexdigest()}
        refs.append({**meta, "ref_code": nc, "kn": {k: d[k] for k in ("K0", "N0", "K", "N", "weight_mb", "factor_features", "batch_divisor",
                                                                     "rule_b", "dot_share", "bytes_old", "bytes_new")},
                     "orig_kind": row["kind"], "orig_speed": row.get("speed_vs_xla")})
        probs.append({**meta, "problem_file": str(V7 / f"kn_refs{a.tag}.jsonl"), "user": render_problem(nc)})
        items.append({"id": "kimiKN0/" + npid, "task": meta["task_name"], "ref_code": nc, "code": kernel})
(V7 / f"kn_log{a.tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
(V7 / f"kn_refs{a.tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))
(V7 / f"kn_problems{a.tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in probs))
(V7 / f"kn_items_existing{a.tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
# dev slice: ~15% by hash, at least 25, stratified so rule-B problems are represented; never trained on
dev = sorted(r["problem_id"] for r in refs if zlib.crc32(r["problem_id"].encode()) % 100 < 15)
(V7 / f"kn_dev_ids{a.tag}.txt").write_text("# v7 held-out dev slice (weight-scaled problems never trained on)\n" + "".join(p + "\n" for p in dev))
print("KN BUILD DONE:", len(refs), "variants of", len(cand), dict(stat), "| rule_b", sum(r["kn"]["rule_b"] for r in refs),
      "| weight MB median", sorted(r["kn"]["weight_mb"] for r in refs)[len(refs) // 2] if refs else None, "| dev", len(dev), flush=True)
