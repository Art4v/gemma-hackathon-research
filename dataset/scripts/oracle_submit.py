"""Stage 6a: build an oracle job from a round's candidates: items JSONL + request JSON per oracle/QUEUE.md.

  python dataset/scripts/oracle_submit.py --batch B --round r [--owner aarav] [--priority 2] [--probes] [--enqueue] [--python PY]

Selection per (task, attempt) slot: the latest sample the prescreen passed or abstained on; if every sample was rejected, the latest
one anyway (so the chain gets a real verdict). --probes also submits every prescreen-rejected sample, listed in <job>.meta.json, to
measure the prescreen's false-reject rate (Phase 1). Human candidates (r<r>-human.jsonl) are always included.

Writes dataset/submissions/<job_id>.items.jsonl, <job_id>.request.json and <job_id>.meta.json. The chip-time estimate comes from
oracle/estimate.task_cost run in a CPU subprocess over the job's references; max_chip_minutes is 1.5x the estimate. Nothing is
copied into oracle/queue/ unless --enqueue is given: the TPU is not available yet, so by default the job only waits in submissions/.
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CANDIDATES, N_SEEDS, ORACLE, PRESCREEN, PROBLEMS, SPEED_BAR, SUBMISSIONS, Isolated, now, read_jsonl, write_json, write_jsonl  # noqa: E402

sys.path.insert(0, str(ORACLE))
import make_pack  # noqa: E402  (pure at import: validates items like the operator will)

ESTIMATE = r'''
import json, sys
sys.path.insert(0, ".")
import tpu_oracle as o, estimate
o.init(cpu_smoke=True)
plan = json.load(open("plan.json"))
total, rows = 0.0, []
for task, (code, n) in plan.items():
    try:
        c = estimate.task_cost(task, code, %d)
        s = c["ref_only_s"] + n * c["kernel_eval_s"] if c["fits_hbm"] else 0.0
        rows.append({"task": task, "n": n, "fits_hbm": c["fits_hbm"], "seconds": s, "input_bytes": c["input_bytes"]})
        total += s
    except Exception as e:
        rows.append({"task": task, "n": n, "error": f"{type(e).__name__}: {str(e)[-160:]}"})
print(json.dumps({"total_s": total, "rows": rows}))
'''


def select(cands: list, pres: dict, probes: bool) -> tuple:
    chosen, probe_ids = {}, []
    by_slot = {}
    for c in cands:
        by_slot.setdefault((c["task_name"], c["attempt"], c["constructor"]["backend"] == "human"), []).append(c)
    for slot, cs in by_slot.items():
        cs.sort(key=lambda c: c["sample"])
        if slot[2]:
            chosen[cs[-1]["candidate_id"]] = cs[-1]; continue
        ok = [c for c in cs if pres.get(c["candidate_id"], {}).get("outcome") in ("pass", "abstain", None)]
        pick = ok[-1] if ok else cs[-1]
        chosen[pick["candidate_id"]] = pick
        if probes:
            for c in cs:
                if c["candidate_id"] != pick["candidate_id"] and pres.get(c["candidate_id"], {}).get("outcome") == "reject":
                    chosen[c["candidate_id"]] = c; probe_ids.append(c["candidate_id"])
    return list(chosen.values()), probe_ids


def estimate_minutes(items: list, python: str | None) -> dict:
    plan = {}
    for it in items:
        plan.setdefault(it["task"], [it["ref_code"], 0])[1] += 1
    files = {p: (ORACLE / p).read_text() for p in ("tpu_oracle.py", "tpu_check.py", "outside_ops.py", "estimate.py")}
    files["est.py"] = ESTIMATE % N_SEEDS
    files["plan.json"] = json.dumps(plan)
    r = Isolated(python, timeout=60 + 30 * len(plan)).run(["est.py"], files=files)
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        return {"total_s": None, "error": (r.stderr.strip().splitlines() or ["no output"])[-1][-300:]}
    return json.loads(line)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", required=True); ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--owner", default="aarav"); ap.add_argument("--priority", type=int, default=2)
    ap.add_argument("--probes", action="store_true"); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--python", default=None)
    ap.add_argument("--model-field", default=None, help="value for each item's `model` field (default: the candidate's constructor model)")
    a = ap.parse_args(argv)
    job_id = f"dataset-{a.batch}-r{a.round}"
    cdir = CANDIDATES / a.batch
    cands = read_jsonl(cdir / f"r{a.round}.jsonl") + read_jsonl(cdir / f"r{a.round}-human.jsonl")
    pres = {r["candidate_id"]: r for r in read_jsonl(PRESCREEN / a.batch / f"r{a.round}.jsonl")}
    problems = {p["problem_id"]: p for p in read_jsonl(PROBLEMS / "problems.jsonl") if p["status"] == "ok"}
    chosen, probe_ids = select([c for c in cands if c["problem_id"] in problems], pres, a.probes)
    items = [{"id": c["candidate_id"], "task": c["task_name"], "ref_code": problems[c["problem_id"]]["ref_code"], "code": c["code"],
              "model": a.model_field or c["constructor"]["model"]} for c in sorted(chosen, key=lambda c: (c["task_name"], c["candidate_id"]))]
    if not items:
        raise SystemExit("no items to submit")
    make_pack.load_items(items)  # the operator's own validation: unique ids, one reference per task
    est = estimate_minutes(items, a.python)
    minutes = math.ceil(est["total_s"] / 60 * 1.5) if est.get("total_s") else None
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    items_path = SUBMISSIONS / f"{job_id}.items.jsonl"
    write_jsonl(items_path, items)
    req = {"job_id": job_id, "owner": a.owner, "step": f"dataset-r{a.round}", "items_path": items_path.name, "seeds": N_SEEDS, "speed_bar": SPEED_BAR,
           "priority": a.priority, "max_chip_minutes": minutes or 60}
    write_json(SUBMISSIONS / f"{job_id}.request.json", req)
    write_json(SUBMISSIONS / f"{job_id}.meta.json", {"job_id": job_id, "n_items": len(items), "n_tasks": len({i["task"] for i in items}),
                                                      "probe_ids": probe_ids, "estimate": est, "created_at": now()})
    if a.enqueue:
        q = ORACLE / "queue"
        q.mkdir(exist_ok=True)
        (q / items_path.name).write_text(items_path.read_text())
        write_json(q / f"{job_id}.json", req)
    print(f"{job_id}: {len(items)} items over {req and len({i['task'] for i in items})} tasks, {len(probe_ids)} probes; "
          f"estimate {est.get('total_s') and round(est['total_s'] / 60, 1)} min -> max_chip_minutes {req['max_chip_minutes']}; "
          f"{'ENQUEUED to oracle/queue' if a.enqueue else 'written to dataset/submissions only (not enqueued)'}")
    for row in est.get("rows", []):
        if row.get("error") or not row.get("fits_hbm", True):
            print(f"  estimate: {row['task']}: {row.get('error') or 'does not fit HBM'}")


if __name__ == "__main__":
    main()
