"""Dataset v3, part 3: extras that need no new kernels.

  python dataset/scripts/v3_extras.py slow                       # correct_slow kernels the TPU oracle already judged
  python dataset/scripts/v3_extras.py variants [--python PY]     # shape variants of the train references, checked on CPU

correct_slow: every kernel the oracle found correct but under the speed bar, on a train-family problem, not already a dataset-v1 target. Row kind
correct_slow, final_verdict correct_slow: they are correct, not fast, so a trainer can weight or drop them.

Shape variants: the same kernel against a reference whose CONFIG has one or two size entries halved. The kernel is unchanged, so a variant only survives
if it reads its shapes at runtime and still lowers and matches the reference (v3_verify.py). Candidates are limited to small inputs, because interpret mode
holds everything in memory and runs the kernel on the CPU.
"""
from __future__ import annotations

import argparse, ast, pathlib, random, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import split_of  # noqa: E402
from common import CANDIDATES, PROBLEMS, VERDICTS, read_jsonl, sha256, task_name, write_jsonl  # noqa: E402
from v3_common import V3, derived_row, train_singles  # noqa: E402
from v3_verify import verify  # noqa: E402
from v3_common import FINAL  # noqa: E402

NOT_A_SIZE = re.compile(r"seed|kernel|stride|pad|dilat|group|window|name|hash|rank|axis|bits|steps|iter|top|shift|num_heads|heads|layers|p$", re.I)
MAX_INPUT_BYTES = 256e6  # before halving; the smallest dataset-v1 train problem is 50 MB, and 94 of the 213 are at or under this


def slow() -> int:
    val_families = {e["family_id"] for e in read_jsonl(FINAL / "val.jsonl")}
    have = {sha256(e["turns"][-1]["code"]) for e in read_jsonl(FINAL / "train.jsonl") + read_jsonl(FINAL / "val.jsonl")}
    problems = {p["problem_id"]: p for p in read_jsonl(PROBLEMS / "problems.jsonl")}
    cands = {c["candidate_id"]: c for p in sorted(CANDIDATES.glob("**/*.jsonl")) for c in read_jsonl(p)}
    out, skip = [], {"no_candidate": 0, "problem_not_ok": 0, "val_family": 0, "already_a_target": 0, "duplicate": 0}
    seen = set()
    for vf in sorted(VERDICTS.glob("*.jsonl")):
        for v in read_jsonl(vf):
            if v["verdict"] != "correct_slow":
                continue
            c = cands.get(v["candidate_id"])
            if c is None:
                skip["no_candidate"] += 1; continue
            p = problems.get(c["problem_id"])
            if p is None or p["status"] != "ok":
                skip["problem_not_ok"] += 1; continue
            if p["family_id"] in val_families or split_of(p["family_id"], 0.1) != "train":
                skip["val_family"] += 1; continue
            h = sha256(c["code"])
            if h in have:
                skip["already_a_target"] += 1; continue
            if h in seen:
                skip["duplicate"] += 1; continue
            seen.add(h)
            base = {"problem_id": p["problem_id"], "family_id": p["family_id"], "source": p["source"], "task_name": p["task_name"], "speed_vs_xla": v["speed_vs_xla"], "at_parity": False,
                    "oracle": {"commits": [v["oracle_commit"]], "shas": [v["oracle_sha"]], "job_ids": [v["job_id"]]}, "split": "train", "example_id": None, "final_verdict": "correct_slow",
                    "meta": {"plan": c.get("plan", ""), "target_origin": "constructor", "constructor": {k: (c.get("constructor") or {}).get(k) for k in ("backend", "model", "prompt_version")}}}
            turns = [{"role": "user", "type": "problem", "ref_code": p["ref_code"]}, {"role": "assistant", "type": "kernel", "code": c["code"], "candidate_id": c["candidate_id"]}]
            out.append(derived_row(base, example_id=f"{c['candidate_id']}.correct_slow", turns=turns, kind="single", origin="correct_slow", verification="tpu_correct_slow",
                                   extra={"oracle_feedback": v["feedback"]}))
    out.sort(key=lambda e: e["example_id"])
    write_jsonl(V3 / "correct_slow.jsonl", out)
    print(f"{len(out)} correct_slow rows; skipped {skip}")
    return 0


def halvable(code: str) -> list:
    """[(key, value, start offset, end offset)] for CONFIG's integer size entries that can be halved. Offsets are into the UTF-8 bytes."""
    tree, b = ast.parse(code), code.encode()
    starts = [0]
    for line in b.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets) and isinstance(n.value, ast.Dict):
            out = []
            for k, v in zip(n.value.keys, n.value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(v, ast.Constant) and type(v.value) is int and v.value >= 8 and v.value % 2 == 0 and not NOT_A_SIZE.search(k.value):
                    out.append((k.value, v.value, starts[v.lineno - 1] + v.col_offset, starts[v.end_lineno - 1] + v.end_col_offset))
            return out
    return []


def variant_code(code: str, picks: list) -> str:
    b = code.encode()
    for _, val, a, e in sorted(picks, key=lambda p: -p[2]):
        b = b[:a] + str(val // 2).encode() + b[e:]
    out = b.decode()
    return out.replace('"""', '"""[shape variant] ', 1) if out.startswith('"""') else "# shape variant\n" + out


def variants(python: str | None) -> int:
    problems = {p["problem_id"]: p for p in read_jsonl(PROBLEMS / "problems.jsonl")}
    bases, items, info, skip = train_singles(), [], {}, {"too_large": 0, "no_size_keys": 0}
    for e in bases:
        p = problems.get(e["problem_id"]) or {}
        if (p.get("input_bytes") or 1e12) > MAX_INPUT_BYTES:
            skip["too_large"] += 1; continue
        ref = e["turns"][0]["ref_code"]
        keys = halvable(ref)
        if not keys:
            skip["no_size_keys"] += 1; continue
        rng = random.Random(e["example_id"])
        picks = [[k] for k in rng.sample(keys, min(2, len(keys)))]
        for n, pick in enumerate(picks):
            cid = f"{e['task_name']}__shape{n}"
            new = variant_code(ref, pick)
            items.append({"id": cid, "task": cid, "ref_code": new, "code": e["turns"][-1]["code"]})
            info[cid] = (e, pick, new)
    print(f"{len(items)} shape variants of {len(bases)} kernels; skipped {skip}", flush=True)
    write_jsonl(V3 / "shape_candidates.jsonl", items)
    res = verify(items, V3 / "shape_verified.jsonl", python, batch=6, budget=60, log=lambda m: print(m, flush=True))
    out, fails = [], {}
    for cid, (e, pick, new) in info.items():
        r = res[cid]
        if not r.get("ok"):
            fails[r.get("stage")] = fails.get(r.get("stage"), 0) + 1
            continue
        turns = [{"role": "user", "type": "problem", "ref_code": new}, e["turns"][-1]]
        row = derived_row(e, example_id=f"{cid}.shape_variant", turns=turns, kind="single", origin="shape_variant", verification="cpu", final_verdict="cpu_verified",
                          extra={"config_delta": {k: v // 2 for k, v, _, _ in pick}, "target_static_verdict": r["static_verdict"], "lowered": r["lowered"], "ref_precision": r["ref_precision"],
                                 "numerics": r["per_seed"]})
        row["problem_id"], row["task_name"], row["signature"] = f"{e['problem_id']}__shape{cid[-1]}", task_name(f"{e['problem_id']}__shape{cid[-1]}"), r["signature"]
        out.append(row)
    write_jsonl(V3 / "shape_variants_unfiltered.jsonl", out)
    print(f"{len(out)} variants passed of {len(items)}; failed at {fails}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("slow", "variants")); ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    return slow() if a.what == "slow" else variants(a.python)


if __name__ == "__main__":
    sys.exit(main())
