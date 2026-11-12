"""dot_share for the v6 (A6) rows whose target is below the speed bar and whose problem has no entry in kn_log.jsonl.

Why. assemble_v7's `--drop-slow matmul` drops a slow v6 row when kn_log.jsonl gives its problem dot_share >= 0.5. kn_log.jsonl was built by
build_kn.py from v6_train_STRICT, without the e3_scale rows and without the flagged problems (generic stacks, rule-3 families). The v7 base is
the A6 file, so a slow row outside that candidate list has no entry, reads as dot_share 0 and is never dropped, however matmul-dominant it is.
This fills the gap with the same measure (scale_kn_one.analyze: the largest non-batched dot's share of dot + conv FLOPs, from the reference's
jaxpr at abstract shapes; nothing is allocated or run).
  python3 dot_share_fill.py   ->  dot_share_fill.jsonl  ({problem_id, dot_share} for every slow A6 row missing from kn_log.jsonl)"""
import json, os, pathlib, re, sys
os.environ["JAX_PLATFORMS"] = "cpu"
V7 = pathlib.Path(__file__).resolve().parent
W = pathlib.Path.home() / "Projects/gemma-work"
src = (V7 / "scale_kn_one.py").read_text()
assert src.rstrip().endswith("main()")
ns = {}
exec(compile(src.rstrip()[:-len("main()")], "scale_kn_one.py", "exec"), ns)  # its functions, without running main()
have = {json.loads(l)["problem_id"] for l in open(V7 / "kn_log.jsonl")}
out, seen = [], set()
for l in open(W / "wt-kimi/kimi/v6/v6_train.jsonl"):
    r = json.loads(l); pid = r["problem_id"]
    if r["kind"] not in ("correct_slow", "repair_slow") or pid in have or pid in seen:
        continue
    seen.add(pid)
    u = r["messages"][1]["content"]; code = u[u.index("```python\n") + 10:u.rindex("```")]
    try:
        d0, _ = ns["analyze"](ns["load"](code))
        out.append({"problem_id": pid, "dot_share": round(d0[3], 3) if d0 else 0.0, "reason": None if d0 else "no non-batched matmul"})
    except Exception as e:
        out.append({"problem_id": pid, "dot_share": 0.0, "reason": f"{type(e).__name__}: {str(e)[:120]}"})
(V7 / "dot_share_fill.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
n = sum(x["dot_share"] >= 0.5 for x in out)
print(f"{len(out)} slow-row problems missing from kn_log.jsonl; dot_share >= 0.5: {n}; errors: {sum(bool(x['reason']) and 'matmul' not in x['reason'] for x in out)}")
