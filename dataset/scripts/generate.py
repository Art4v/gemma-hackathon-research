"""Stage 4: the constructor writes candidate kernels, one round at a time.

  python dataset/scripts/generate.py --batch B --round 1 [--problems id,id,...] [--limit N] [--backend claude_cli] [--model M] [--workers 3]
  python dataset/scripts/generate.py --batch B --round 2            # repairs: problems whose round-1 verdict is a kernel fault
  python dataset/scripts/generate.py --batch B --round r --resample # re-sample slots the prescreen rejected (max 2 per slot)

Round 1 takes every problem with status ok in problems.jsonl. Rounds 2 and 3 read trajectories/B.jsonl (written by oracle_collect.py)
and build the repair prompt from the prior kernels and the oracle's verbatim feedback. --resample reads prescreen/B/r<r>.jsonl and
re-asks for rejected slots with the local reason as a private hint (prescreen_hint, never a training turn). Candidates go to
candidates/B/r<r>.jsonl; raw responses to candidates/raw_responses/ (gitignored).
"""
from __future__ import annotations

import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CANDIDATES, KERNEL_FAULT, PRESCREEN, PROBLEMS, TRAJECTORIES, candidate_id, now, read_jsonl, sha256, validate, write_jsonl  # noqa: E402
from constructor import Request, get_constructor, save_raw  # noqa: E402
from constructor.prompts import PROMPT_VERSION, kernel_system, kernel_user  # noqa: E402
from render_chat import unfence  # noqa: E402

MAX_RESAMPLES = 2


def all_candidates(batch: str) -> dict:
    out = {}
    for p in sorted((CANDIDATES / batch).glob("r*.jsonl")):
        for c in read_jsonl(p):
            out[c["candidate_id"]] = c
    return out


def plan_round1(problems: dict, cands: dict, only: set | None) -> list:
    """(problem, attempt, sample, parent, feedback_in, feedback_verdict, hint, history) for every problem without a round-1 candidate."""
    have = {c["task_name"] for c in cands.values() if c["attempt"] == 1 and c["constructor"]["backend"] != "human"}
    return [(p, 1, 0, None, None, None, None, []) for p in problems.values() if p["task_name"] not in have and (only is None or p["problem_id"] in only)]


def plan_repair(batch: str, rnd: int, problems: dict, cands: dict, only: set | None) -> list:
    jobs = []
    for t in read_jsonl(TRAJECTORIES / f"{batch}.jsonl"):
        p = problems.get(t["problem_id"])
        if p is None or t.get("status") != "open" or (only is not None and p["problem_id"] not in only):
            continue
        attempts = [x for x in t.get("attempts", []) if cands.get(x["candidate_id"], {}).get("constructor", {}).get("backend") != "human"]
        if len(attempts) != rnd - 1 or attempts[-1]["verdict"] not in KERNEL_FAULT:
            continue
        if any(c["task_name"] == p["task_name"] and c["attempt"] == rnd for c in cands.values()):
            continue
        history = [(cands[x["candidate_id"]]["code"], x["feedback"]) for x in attempts]
        last = attempts[-1]
        jobs.append((p, rnd, 0, last["candidate_id"], last["feedback"], {"job_id": last["job_id"], "oracle_sha": last["oracle_sha"], "verdict": last["verdict"]}, None, history))
    return jobs


def plan_resample(batch: str, rnd: int, problems: dict, cands: dict) -> list:
    pres = {r["candidate_id"]: r for r in read_jsonl(PRESCREEN / batch / f"r{rnd}.jsonl")}
    jobs, seen = [], set()
    for cid, r in pres.items():
        c = cands.get(cid)
        if r["outcome"] != "reject" or c is None or c["constructor"]["backend"] == "human":
            continue
        slot = (c["task_name"], c["attempt"])
        if slot in seen:
            continue
        seen.add(slot)
        samples = [x for x in cands.values() if (x["task_name"], x["attempt"]) == slot]
        if len(samples) > MAX_RESAMPLES or any(pres.get(x["candidate_id"], {}).get("outcome") in ("pass", "abstain", None) for x in samples):
            continue
        latest = max(samples, key=lambda x: x["sample"])
        if pres.get(latest["candidate_id"], {}).get("outcome") != "reject":
            continue
        p = problems[c["problem_id"]]
        history = []
        if c["parent_candidate_id"]:
            # rebuild the history from the parent chain
            node = cands.get(c["parent_candidate_id"])
            chain = []
            while node is not None:
                chain.append(node)
                node = cands.get(node["parent_candidate_id"]) if node["parent_candidate_id"] else None
            for x in reversed(chain):
                fb = next((y["feedback_in"] for y in cands.values() if y["parent_candidate_id"] == x["candidate_id"] and y["feedback_in"]), None)
                history.append((x["code"], fb or ""))
        hint = pres[latest["candidate_id"]]["reason"]
        jobs.append((p, c["attempt"], latest["sample"] + 1, c["parent_candidate_id"], c["feedback_in"], c["feedback_in_verdict"], hint, history))
    return jobs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", required=True); ap.add_argument("--round", type=int, required=True); ap.add_argument("--resample", action="store_true")
    ap.add_argument("--problems", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="claude_cli"); ap.add_argument("--model", default=None); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--effort", default=None); ap.add_argument("--no-docs", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse-raw", action="store_true", help="use saved responses in candidates/raw_responses before asking the constructor")
    a = ap.parse_args(argv)
    problems = {p["problem_id"]: p for p in read_jsonl(PROBLEMS / "problems.jsonl") if p["status"] == "ok"}
    cands = all_candidates(a.batch)
    only = set(a.problems.split(",")) if a.problems else None
    if a.resample:
        jobs = plan_resample(a.batch, a.round, problems, cands)
    elif a.round == 1:
        # each problem belongs to one batch (problem.batch; pilot problems predate the field); round 1 only takes this batch's
        jobs = plan_round1({k: v for k, v in problems.items() if v.get("batch", "pilot") == a.batch}, cands, only)
    else:
        jobs = plan_repair(a.batch, a.round, problems, cands, only)
    jobs = jobs[:a.limit] if a.limit else jobs
    print(f"{len(jobs)} requests for {a.batch} round {a.round}{' (resample)' if a.resample else ''}")
    if a.dry_run or not jobs:
        for p, att, s, *_ in jobs:
            print(f"  {p['task_name']} a{att} s{s}")
        return
    system, docs_sha = kernel_system(with_docs=not a.no_docs)
    con = get_constructor(a.backend, a.model, workers=a.workers, **({"effort": a.effort} if a.effort else {}))
    reqs, meta = [], {}
    for p, att, s, parent, fb, fbv, hint, history in jobs:
        cid = candidate_id(p["task_name"], att, s)
        user = kernel_user(p["ref_code"], history)
        if hint:
            user += "\n\n[local check]\n" + hint + "\nFix this and return the complete module."
        reqs.append(Request(cid, system, user))
        meta[cid] = (p, att, s, parent, fb, fbv, hint)
    recs, failed = [], []
    responses = []
    if a.reuse_raw:  # a run interrupted after the constructor answered but before records were written costs no calls
        from port_kernelbench import saved_response
        for r in reqs:
            s = saved_response(CANDIDATES / "raw_responses", r.key)
            if s is not None:
                responses.append(s)
        got = {r.key for r in responses}
        reqs = [r for r in reqs if r.key not in got]
        print(f"reusing {len(responses)} saved responses, asking for {len(reqs)}", flush=True)
    if reqs:
        responses += con.generate(reqs)
    for resp in responses:
        p, att, s, parent, fb, fbv, hint = meta[resp.key]
        raw_sha = save_raw(CANDIDATES / "raw_responses", resp)
        if not resp.ok:
            failed.append((resp.key, resp.error)); continue
        code = (unfence(resp.output["code"]) or resp.output["code"]).strip("\n") + "\n"
        rec = {"candidate_id": resp.key, "problem_id": p["problem_id"], "task_name": p["task_name"], "attempt": att, "sample": s,
               "parent_candidate_id": parent, "feedback_in": fb, "feedback_in_verdict": fbv, "prescreen_hint": hint,
               "constructor": {"backend": resp.backend, "model": resp.model, "prompt_version": PROMPT_VERSION, "docs_sha256": docs_sha},
               "plan": resp.output["plan"], "code": code, "code_sha256": sha256(code), "usage": resp.usage, "raw_sha256": raw_sha, "created_at": now()}
        errs = validate("candidate", rec)
        if errs:
            failed.append((resp.key, f"invalid record: {errs}")); continue
        recs.append(rec)
    write_jsonl(CANDIDATES / a.batch / f"r{a.round}.jsonl", recs, append=True)
    toks = sum(r["usage"].get("output_tokens", 0) for r in recs)
    print(f"wrote {len(recs)} candidates ({toks} output tokens), {len(failed)} failed -> candidates/{a.batch}/r{a.round}.jsonl")
    for k, e in failed:
        print(f"  failed {k}: {e}")


if __name__ == "__main__":
    main()
