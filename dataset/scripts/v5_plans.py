"""Dataset v5: dataset-v4's rows with the constructor's saved plan written as a comment block at the top of every kernel.

  python dataset/scripts/v5_plans.py collect --candidates DIR   # once: history_plans.jsonl + rewrite_in/batch_*.jsonl
  python dataset/scripts/v5_plans.py build                      # train.jsonl, val.jsonl, MANIFEST.json in dataset/v5/

Why: v4's targets are code only, and v4 fails on shape bookkeeping and block sizes, which is what the plans spell out. The plan goes
inside the code block as comments, so the prompt, the "one code block" reply format and the oracle's view of the kernel do not change.

Which plan a kernel gets:
- a target, and a failed attempt in a real repair chain: that candidate's own plan (a repair plan may refer to the failure; it is in context);
- a kernel from attempt 2 or 3 shown outside its chain (single, correct_slow, injected repair): the plan rewritten to stand alone,
  from rewrite_out/, because "the reshape was the slowdown" would refer to an attempt the row does not show;
- the bugged kernel of an injected repair: the plan of the kernel the bug was planted in.
Rows never checked on a TPU (shape_variant, generated) are left out: a variant's plan quotes the original shape's numbers, and generated
kernels have no plan.
"""
from __future__ import annotations

import argparse, ast, collections, hashlib, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
V5 = ROOT / "v5"
SOURCES = ["final/train.jsonl", "final/val.jsonl", "v3/repairs.jsonl", "v3/correct_slow.jsonl", "v4/m1.jsonl", "v4/kbk.jsonl", "v4/kbkc.jsonl",
           "v4/kbkd.jsonl", "v4/prod.jsonl"]
SKIP_KINDS = {"shape_variant", "generated"}
MAX_TARGET_CHARS = 18000  # v4_build's cap: a longer target would be cut off by the eval's 8,192-token limit
BATCH = 55


def read(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


def write(p, rows):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def attempt(cid: str) -> int:
    m = re.search(r"\.a(\d+)\.", cid or "")
    return int(m.group(1)) if m else 1


def header(plan: str) -> str:
    lines = ["# PLAN"] + [("# " + l.rstrip()) if l.strip() else "#" for l in plan.strip().splitlines()]
    return "\n".join(lines) + "\n"


def load():
    src = {}
    for f in SOURCES:
        for e in read(ROOT / f):
            src.setdefault(e["example_id"], e)
    rows = [(split, r) for split in ("train", "val") for r in read(ROOT / "v4" / "final" / f"{split}.jsonl") if r["kind"] not in SKIP_KINDS]
    return src, rows


def needs_standalone(kind: str, e: dict) -> bool:
    return kind != "repair" and attempt(e["turns"][-1].get("candidate_id")) > 1


def feedback_of(e: dict) -> list:
    return [t["feedback"] for t in e["turns"] if t.get("type") == "oracle_feedback"]


def collect(a) -> int:
    src, rows = load()
    cands = {}
    for f in sorted(pathlib.Path(a.candidates).glob("*.jsonl")) + sorted((ROOT / "candidates").rglob("*.jsonl")):
        for c in read(f):
            if c.get("candidate_id") and (c.get("plan") or "").strip():
                cands.setdefault((c["candidate_id"], sha_text(c["code"])), c["plan"])
    hist, missing, standalone = {}, [], {}
    for _, r in rows:
        e = src[r["example_id"]]
        cid = e["turns"][-1].get("candidate_id")
        if r["kind"] == "repair":
            for t in e["turns"][1:-1]:
                if t["role"] == "assistant":
                    key = (t["candidate_id"], sha_text(t["code"]))
                    if key in cands:
                        hist[key] = cands[key]
                    else:
                        missing.append(key[0])
        elif needs_standalone(r["kind"], e):
            standalone.setdefault(cid, {"id": cid, "plan": e["meta"]["plan"], "earlier_feedback": []})
    for _, r in rows:  # the real chain's feedback, as context for the rewriter
        e = src[r["example_id"]]
        cid = e["turns"][-1].get("candidate_id")
        if r["kind"] == "repair" and cid in standalone:
            standalone[cid]["earlier_feedback"] = feedback_of(e)
    write(V5 / "history_plans.jsonl", [{"candidate_id": k[0], "code_sha256": k[1], "plan": p} for k, p in sorted(hist.items())])
    items = sorted(standalone.values(), key=lambda x: x["id"])
    (V5 / "rewrite_in").mkdir(parents=True, exist_ok=True)
    for old in (V5 / "rewrite_in").glob("batch_*.jsonl"):
        old.unlink()
    for i in range(0, len(items), BATCH):
        write(V5 / "rewrite_in" / f"batch_{i // BATCH:02d}.jsonl", items[i:i + BATCH])
    print(f"history plans {len(hist)}; failed attempts with no saved plan {len(missing)} {sorted(set(missing))[:5]}; plans to rewrite standalone {len(items)}")
    return 0


REFERS_BACK = re.compile(r"\b(previous|earlier|attempt|feedback|oracle|was the|were the|no longer|this time|the fix|fix:|unchanged|slowdown|regress)", re.I)


def check_rewrite(old: str, new: str) -> list:
    """Reasons a standalone rewrite is not acceptable; empty when it is. A rewrite may only remove or rephrase, so it adds no number."""
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
    out = []
    if added := sorted(nums(new) - nums(old)):
        out.append(f"adds numbers {added[:6]}")
    if not 0.4 * len(old) <= len(new) <= 1.15 * len(old):
        out.append(f"length {len(old)} -> {len(new)}")
    if m := REFERS_BACK.search(new):
        out.append(f"still refers back: {m.group(0)!r}")
    if "```" in new:
        out.append("contains a code fence")
    return out


def build(a) -> int:
    src, rows = load()
    hist = {(h["candidate_id"], h["code_sha256"]): h["plan"] for h in read(V5 / "history_plans.jsonl")}
    originals = {x["id"]: x["plan"] for f in sorted((V5 / "rewrite_in").glob("batch_*.jsonl")) for x in read(f)}
    alone, rejected = {}, {}
    for f in sorted((V5 / "rewrite_out").glob("batch_*.jsonl")):
        for x in read(f):
            if x["id"] in originals:
                why = check_rewrite(originals[x["id"]], x["plan"])
                if why:
                    rejected[x["id"]] = why
                else:
                    alone[x["id"]] = x["plan"]
    out, dropped, stats, added = {"train": [], "val": []}, collections.Counter(), collections.Counter(), []
    for split, r in rows:
        e = src[r["example_id"]]
        code_turns = [t for t in e["turns"] if t["role"] == "assistant"]
        target_plan = e["meta"]["plan"]
        if needs_standalone(r["kind"], e):
            cid = code_turns[-1].get("candidate_id")
            if cid not in alone:
                dropped["standalone plan missing or rejected"] += 1
                continue
            target_plan = alone[cid]
        plans = [target_plan if r["kind"] == "repair_injected" else hist.get((t.get("candidate_id"), sha_text(t["code"]))) for t in code_turns[:-1]]
        plans.append(target_plan)
        if any(p is None or not p.strip() or "```" in p for p in plans):
            dropped["a kernel in the row has no usable plan"] += 1
            continue
        msgs, k = [], 0
        for m in r["messages"]:
            m = dict(m)
            if m["role"] == "assistant":
                code = code_turns[k]["code"]
                assert m["content"] == f"```python\n{code}\n```", r["example_id"]  # the row is v4's, untouched, before the plan goes in
                new = header(plans[k]) + code
                if k == len(code_turns) - 1:
                    ast.parse(new)  # the target still parses; failed attempts may not have parsed to begin with
                m["content"] = f"```python\n{new}\n```"
                k += 1
            msgs.append(m)
        assert k == len(code_turns), r["example_id"]
        if len(msgs[-1]["content"]) > MAX_TARGET_CHARS:
            dropped["target over the 18,000-char cap with its plan"] += 1
            continue
        out[split].append({**r, "messages": msgs})
        stats[f"{split}/{r['kind']}"] += 1
        added.append(len(msgs[-1]["content"]) - len(r["messages"][-1]["content"]))
    for split in out:
        write(V5 / f"{split}.jsonl", out[split])
    sha_file = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    added.sort()
    manifest = {"from": {f"v4/final/{s}.jsonl": sha_file(ROOT / "v4" / "final" / f"{s}.jsonl") for s in ("train", "val")},
                "rows": dict(sorted(stats.items())), "train_rows": len(out["train"]), "val_rows": len(out["val"]),
                "left_out_kinds": sorted(SKIP_KINDS), "dropped": dict(dropped), "standalone_rewrites_used": len(alone),
                "standalone_rewrites_rejected": dict(sorted(rejected.items())),
                "plan_header_chars": {"median": added[len(added) // 2], "max": added[-1]} if added else {},
                "files": {f"{s}.jsonl": sha_file(V5 / f"{s}.jsonl") for s in out}}
    (V5 / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "standalone_rewrites_rejected"}, indent=1))
    print(f"rejected rewrites: {len(rejected)}")
    return 0


def demo(_=None) -> int:
    assert header("a\n\nb ") == "# PLAN\n# a\n#\n# b\n"
    assert attempt("x.a2.s0") == 2 and attempt("x.inj.k") == 1
    assert check_rewrite("grid (4, 8), block 2 MiB; the reshape was the slowdown", "grid (4, 8), block 2 MiB; x stays in its native layout") == []
    assert check_rewrite("grid (4, 8)", "grid (4, 16)") and check_rewrite("grid (4, 8) now", "grid (4, 8), the earlier one was slow")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--candidates", required=True)
    sub.add_parser("build")
    sub.add_parser("demo")
    a = ap.parse_args()
    sys.exit({"collect": collect, "build": build, "demo": demo}[a.cmd](a))
