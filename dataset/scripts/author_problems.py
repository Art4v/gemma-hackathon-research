"""Stage 2b: the constructor authors new p-style problems (model operators outside the exam's op set) as plain-JAX baselines.

  python dataset/scripts/author_problems.py [--categories a,b,...] [--per-category 1] [--limit 10] [--backend claude_cli] [--model M]

Each module is checked in a `python -I` subprocess: it loads, `create_inputs()` shapes resolve, `workload` traces, input bytes are
between 20 MB and the HBM budget / 3. If signature.py is available, the op-chain is compared against the 50 exam references right
away and a match is sent back to the constructor as a rejection (at most 2 retries). The full filters run afterwards.
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CANDIDATES, HBM_BUDGET_BYTES, PIPELINE_VERSION, PROBLEMS, Isolated, now, read_jsonl, sha256, task_name, validate, write_jsonl  # noqa: E402
from constructor import Request, get_constructor, save_raw  # noqa: E402
from constructor.prompts import AUTHOR_SCHEMA, AUTHOR_SYSTEM, CATEGORIES, PROMPT_VERSION, author_user  # noqa: E402
from common import unfence  # noqa: E402

MIN_BYTES = 20e6

SHAPE_CHECK = r'''
import json, sys, importlib.util
import jax, jax.numpy as jnp
spec = importlib.util.spec_from_file_location("ref", "ref.py"); m = importlib.util.module_from_spec(spec); sys.modules["ref"] = m
spec.loader.exec_module(m)
abstract = jax.eval_shape(m.create_inputs)
leaves = jax.tree_util.tree_leaves(abstract)
nbytes = int(sum(x.size * x.dtype.itemsize for x in leaves))
out = jax.eval_shape(m.workload, *abstract)
print(json.dumps({"ok": True, "input_bytes": nbytes, "shapes": [list(x.shape) for x in leaves], "dtypes": [str(x.dtype) for x in leaves],
                  "out_shapes": [list(x.shape) for x in jax.tree_util.tree_leaves(out)]}))
'''


def shape_check(code: str, python: str | None) -> dict:
    r = Isolated(python, timeout=300).run(["check.py"], files={"check.py": SHAPE_CHECK, "ref.py": code})
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        return {"ok": False, "reason": (r.stderr.strip().splitlines() or ["no output"])[-1][-400:]}
    d = json.loads(line)
    if d["input_bytes"] < MIN_BYTES:
        d.update(ok=False, reason=f"inputs too small: {d['input_bytes'] / 1e6:.0f} MB (< 20 MB)")
    elif 3 * d["input_bytes"] > HBM_BUDGET_BYTES:
        d.update(ok=False, reason=f"inputs too large: 3 x {d['input_bytes'] / 1e9:.1f} GB exceeds the 28 GB HBM budget")
    return d


def exam_opchain_match(code: str, python: str | None):
    """(exam task, chain) if the op-chain equals an exam reference's, else None. Silently skipped if signature.py is missing."""
    try:
        import signature as sg
    except ImportError:
        return None
    sig = sg.signature_of(code, python)
    if "error" in sig:
        return None
    for task, s in sg.exam_signatures().items():
        if "error" not in s and s.get("op_chain") == sig.get("op_chain"):
            return task, sig
    return None


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:60] or "problem"


def record(cat: str, name: str, code: str, chk: dict, status: str, reason: str | None, plan: str) -> dict:
    pid = f"authored/{cat}/{name}"
    return {"problem_id": pid, "family_id": pid, "task_name": task_name(pid), "source": "authored",
            "source_ref": {"repo": None, "commit": None, "path": None, "license": None, "category": cat, "spec": CATEGORIES.get(cat, "")},
            "ref_code": code, "ref_sha256": sha256(code), "signature": None, "input_bytes": chk.get("input_bytes"),
            "fits_hbm": (3 * chk["input_bytes"] <= HBM_BUDGET_BYTES) if chk.get("input_bytes") else None, "port_check": None,
            "plan": plan, "status": status, "exclude_reason": reason, "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION,
            "created_at": now()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--categories", default=",".join(CATEGORIES)); ap.add_argument("--per-category", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--backend", default="claude_cli"); ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--python", default=None); ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args(argv)
    existing = read_jsonl(PROBLEMS / "problems.jsonl")
    have = {}
    for r in existing:
        if r["source"] == "authored":
            have.setdefault(r["source_ref"]["category"], []).append(r["problem_id"].split("/")[-1])
    todo = []
    for cat in a.categories.split(","):
        if cat not in CATEGORIES:
            raise SystemExit(f"unknown category {cat}; known: {', '.join(CATEGORIES)}")
        for _ in range(max(0, a.per_category - len(have.get(cat, [])))):
            todo.append(cat)
    todo = todo[:a.limit] if a.limit else todo
    print(f"authoring {len(todo)} problems")
    con = get_constructor(a.backend, a.model, workers=a.workers)
    raw_dir = CANDIDATES / "raw_responses"
    pending = [(cat, None, 0) for cat in todo]   # (category, retry note, retries used)
    kept, excluded = [], []
    while pending:
        reqs = [Request(f"author/{cat}/{i}", AUTHOR_SYSTEM, author_user(cat, CATEGORIES[cat], avoid=have.get(cat), retry_note=note), AUTHOR_SCHEMA)
                for i, (cat, note, _) in enumerate(pending)]
        meta = {f"author/{cat}/{i}": (cat, n) for i, (cat, _, n) in enumerate(pending)}
        pending = []
        for resp in con.generate(reqs):
            cat, n = meta[resp.key]
            save_raw(raw_dir, resp)
            if not resp.ok:
                excluded.append(record(cat, f"failed_{n}", "", {}, "excluded", f"author_failed:constructor:{resp.error}", "")); continue
            name = slug(resp.output["name"])
            code = (unfence(resp.output["code"]) or resp.output["code"]).strip("\n") + "\n"
            chk = shape_check(code, a.python)
            note = None
            if not chk.get("ok"):
                note = f"the module failed its shape check: {chk.get('reason')}"
            else:
                hit = exam_opchain_match(code, a.python)
                if hit:
                    note = f"its op-chain equals exam task {hit[0]} ({' -> '.join(hit[1]['op_chain'][:12])}); choose a different operator"
            if note is None:
                kept.append(record(cat, name, code, chk, "ok", None, resp.output["plan"]))
                have.setdefault(cat, []).append(name)
                print(f"  {cat}/{name}: ok {chk['input_bytes'] / 1e6:.0f} MB")
            elif n < a.retries:
                pending.append((cat, note, n + 1)); print(f"  {cat}/{name}: retry ({note[:80]})")
            else:
                excluded.append(record(cat, name, code, chk, "excluded", f"author_rejected:{note[:200]}", resp.output["plan"]))
                print(f"  {cat}/{name}: excluded ({note[:80]})")
    for r in kept:
        errs = validate("problem", r)
        if errs:
            raise SystemExit(f"{r['problem_id']}: {errs}")
        out = PROBLEMS / "authored" / r["source_ref"]["category"] / (r["problem_id"].split("/")[-1] + ".py")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r["ref_code"])
    write_jsonl(PROBLEMS / "problems.jsonl", kept, append=True)
    write_jsonl(PROBLEMS / "excluded.jsonl", excluded, append=True)
    print(f"authored {len(kept)}, excluded {len(excluded)}")


if __name__ == "__main__":
    main()
