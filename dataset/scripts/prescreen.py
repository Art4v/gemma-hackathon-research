"""Stage 5 (local pre-check): a parse check and the oracle's own kernel check (`tpu_oracle.py --static-only`) on each candidate.
Reject-only. Anything not a clear rejection goes to the chip.

  python dataset/scripts/prescreen.py --batch B --round r [--python PY]

Reads candidates/B/r<r>.jsonl (+ r<r>-human.jsonl), writes prescreen/B/r<r>.jsonl (one record per candidate not yet screened).
The static check runs the oracle files copied verbatim into a `python -I` subprocess on CPU with a pack.json built like
make_pack does, so `oracle_sha_used` is the oracle's own hash. Verdicts: static_ok -> pass; not_a_kernel -> reject (stage static,
with the oracle's reason); a trace/import failure at the static stage (verdict compile) -> reject (stage static); a reference
failure (harness_oom) -> abstain.
"""
from __future__ import annotations

import argparse, ast, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CANDIDATES, ORACLE, PRESCREEN, PROBLEMS, Isolated, read_jsonl, sha256, validate, write_jsonl  # noqa: E402

ORACLE_FILES = ("tpu_oracle.py", "tpu_check.py", "outside_ops.py")


def parse_check(code: str) -> str | None:
    """None if the module parses, defines workload and imports pallas; else the reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "workload" not in names:
        return "no module-level `workload` function"
    if "pallas" not in code:
        return "does not import jax.experimental.pallas"
    return None


def static_check(items: list, refs: dict, python: str | None, timeout: int) -> tuple:
    """(rows by id, oracle sha) from one --static-only run over all items."""
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    sha = sha256(b"".join((ORACLE / p).read_bytes() for p in ORACLE_FILES))[:12]
    files["pack.json"] = json.dumps({"refs": refs, "items": [{"id": i["id"], "task": i["task"], "code": i["code"]} for i in items]})
    r = Isolated(python, timeout=timeout).run(["tpu_oracle.py", "pack.json", "out.jsonl", "--static-only"], files=files, extra_env={"ORACLE_COMMIT": "prescreen"})
    rows = {}
    for line in r.stdout.splitlines():
        if line.startswith("{") and '"meta": true' not in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in d:
                rows[d["id"]] = d
    # the oracle prints a progress line per item but writes rows to out.jsonl inside the temp dir, so parse stdout's print() too
    if not rows:
        for line in r.stdout.splitlines():
            parts = line.split(" ", 3)
            if len(parts) >= 3 and parts[0].isdigit():
                rows[parts[1]] = {"id": parts[1], "verdict": parts[2], "feedback": parts[3] if len(parts) > 3 else ""}
    if not rows and r.returncode != 0:
        rows["__error__"] = {"verdict": "harness", "feedback": (r.stderr.strip().splitlines() or ["no output"])[-1][-300:]}
    return rows, sha


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", required=True); ap.add_argument("--round", type=int, required=True); ap.add_argument("--python", default=None)
    ap.add_argument("--timeout", type=int, default=0, help="seconds for the static run; default 60 + 20 per candidate")
    a = ap.parse_args(argv)
    cdir = CANDIDATES / a.batch
    cands = read_jsonl(cdir / f"r{a.round}.jsonl") + read_jsonl(cdir / f"r{a.round}-human.jsonl")
    out_path = PRESCREEN / a.batch / f"r{a.round}.jsonl"
    done = {r["candidate_id"] for r in read_jsonl(out_path)}
    cands = [c for c in cands if c["candidate_id"] not in done]
    problems = {p["problem_id"]: p for p in read_jsonl(PROBLEMS / "problems.jsonl") if p["status"] == "ok"}
    print(f"prescreen {len(cands)} candidates in {a.batch} round {a.round}")
    if not cands:
        return
    recs, items, refs = [], [], {}
    t0 = time.time()
    for c in cands:
        why = parse_check(c["code"])
        if why:
            recs.append({"candidate_id": c["candidate_id"], "outcome": "reject", "stage": "parse", "reason": why, "oracle_sha_used": None, "seconds": 0.0})
            continue
        p = problems.get(c["problem_id"])
        if p is None:
            recs.append({"candidate_id": c["candidate_id"], "outcome": "abstain", "stage": None, "reason": "problem not ok", "oracle_sha_used": None, "seconds": 0.0})
            continue
        items.append({"id": c["candidate_id"], "task": c["task_name"], "code": c["code"]})
        refs[c["task_name"]] = p["ref_code"]
    if items:
        rows, sha = static_check(items, refs, a.python, a.timeout or 60 + 20 * len(items))
        per = round((time.time() - t0) / max(1, len(items)), 2)
        err = rows.get("__error__")
        for it in items:
            row = rows.get(it["id"])
            if err or row is None:
                recs.append({"candidate_id": it["id"], "outcome": "abstain", "stage": "static", "reason": (err or {}).get("feedback", "no row from the static run"),
                             "oracle_sha_used": sha, "seconds": per}); continue
            v = row.get("verdict")
            if v == "static_ok":
                rec = {"outcome": "pass", "stage": "static", "reason": ""}
            elif v == "not_a_kernel":
                rec = {"outcome": "reject", "stage": "static", "reason": row.get("feedback", "not a Pallas kernel")}
            elif v in ("compile",):
                rec = {"outcome": "reject", "stage": "static", "reason": row.get("feedback", "failed to trace")}
            else:
                rec = {"outcome": "abstain", "stage": "static", "reason": f"{v}: {row.get('feedback', '')}"}
            recs.append({"candidate_id": it["id"], **rec, "oracle_sha_used": sha, "seconds": per})
    for r in recs:
        errs = validate("prescreen", r)
        if errs:
            raise SystemExit(f"{r['candidate_id']}: {errs}")
    write_jsonl(out_path, recs, append=True)
    n = {k: sum(1 for r in recs if r["outcome"] == k) for k in ("pass", "reject", "abstain")}
    print(f"prescreen: {n} -> {out_path}")
    for r in recs:
        if r["outcome"] == "reject":
            print(f"  reject {r['candidate_id']} [{r['stage']}] {r['reason'][:110]}")


if __name__ == "__main__":
    main()
