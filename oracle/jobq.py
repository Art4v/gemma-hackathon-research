"""The TPU oracle job queue: requests in oracle/queue/*.json become job directories (make_pack.py) for one operator to run on a v6e VM.
Not named queue.py: in oracle/ that would shadow the stdlib queue module that JAX imports.

  python oracle/jobq.py list     [JOB_ID ...] [--queue oracle/queue] [--jobs-dir oracle/jobs]
  python oracle/jobq.py estimate [JOB_ID ...] [--queue ...]                     needs JAX (CPU is enough); exit 1 if any job is refused
  python oracle/jobq.py build    [JOB_ID ...] [--queue ...] [--jobs-dir ...] [--shards N]    estimate, then pack every job not yet built

A request: {"job_id", "owner", "step", "items_path" (relative to the request file) or "items" (inline), "seeds", "speed_bar",
"priority" (lower runs first), "max_chip_minutes"}. Items are make_pack.py's.
The estimate is estimate.py's roofline per task: each task that fits the HBM budget costs its reference once (ref_only_s) plus, per
kernel, kernel_eval_s; tasks that do not fit never reach the chip. A job whose estimate exceeds max_chip_minutes, or whose references
cannot be estimated, is refused and never built.
"""
import argparse, collections, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_pack  # noqa: E402

FIELDS = ("job_id", "owner", "step", "seeds", "speed_bar", "priority", "max_chip_minutes")


def load_requests(queue: pathlib.Path, only: list) -> list:
    """[(request, items or the ValueError that made it invalid)] in priority order, then job_id."""
    out, ids = [], collections.Counter()
    for p in sorted(pathlib.Path(queue).glob("*.json")):
        req = json.loads(p.read_text())
        ids[req.get("job_id")] += 1
        try:
            missing = [k for k in FIELDS if k not in req]
            if missing or ("items" in req) == ("items_path" in req):
                raise ValueError(f"{p.name}: needs {missing or ''} and exactly one of items, items_path")
            if not make_pack.JOB_ID.fullmatch(str(req["job_id"])):
                raise ValueError(f"{p.name}: job_id: letters, digits, '.', '_', '-' only")
            num = lambda k: isinstance(req[k], (int, float)) and not isinstance(req[k], bool) and req[k] > 0
            if not (num("max_chip_minutes") and num("speed_bar") and isinstance(req["priority"], (int, float))
                    and isinstance(req["seeds"], int) and req["seeds"] >= 1):
                raise ValueError(f"{p.name}: need max_chip_minutes > 0, speed_bar > 0, a numeric priority, seeds an int >= 1")
            items = (make_pack.load_items(req["items"], p.parent) if "items" in req else
                     make_pack.load_items(p.parent / req["items_path"]))
        except (ValueError, OSError, KeyError, TypeError) as e:
            items = ValueError(str(e))
        out.append((req, items))
    for req, _ in out:
        if ids[req.get("job_id")] > 1:
            raise SystemExit(f"job_id {req.get('job_id')!r} is requested more than once in {queue}")
    out = [x for x in out if not only or x[0].get("job_id") in only]
    return sorted(out, key=lambda x: (x[0].get("priority", 1e9), str(x[0].get("job_id"))))


def estimate(req: dict, items: list) -> tuple:
    """(chip minutes or None, reason it is refused or "")."""
    import estimate as est  # JAX on CPU, only when estimating
    est.o.init(cpu_smoke=True)
    per_task = collections.Counter(i["task"] for i in items)
    refs = {i["task"]: i["ref_code"] for i in items}
    total = 0.0
    for t, n in per_task.items():
        try:
            c = est.task_cost(t, refs[t], req["seeds"])
        except Exception as e:  # a reference that cannot be traced cannot be bounded
            return None, f"could not estimate task {t}: {type(e).__name__}: {str(e)[-200:]}"
        if c["fits_hbm"]:
            total += c["ref_only_s"] + n * c["kernel_eval_s"]
    mins = total / 60
    return mins, (f"estimate {mins:.2f} min > max_chip_minutes {req['max_chip_minutes']}" if mins > req["max_chip_minutes"] else "")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("list", "estimate", "build")); ap.add_argument("job_ids", nargs="*")
    ap.add_argument("--queue", default=str(HERE / "queue")); ap.add_argument("--jobs-dir", default=str(HERE / "jobs"))
    ap.add_argument("--shards", type=int, default=1, help="build: parallel shards, one per TPU chip (v6e-1: 1, v6e-8: 8)")
    a = ap.parse_args(argv)
    jobs, failed = pathlib.Path(a.jobs_dir), False
    reqs = load_requests(pathlib.Path(a.queue), a.job_ids)
    if a.cmd == "list":
        print(f"{'job_id':32} {'owner':12} {'step':12} {'prio':>5} {'items':>6} {'max min':>8}  status")
    for req, items in reqs:
        jid, d = req.get("job_id"), jobs / str(req.get("job_id"))
        if a.cmd == "list":
            status = (f"INVALID: {items}" if isinstance(items, Exception) else "collected" if (d / "summary.json").exists() else
                      "built" if (d / "MANIFEST.json").exists() else "queued")
            n = "-" if isinstance(items, Exception) else len(items)
            print(f"{jid!s:32} {req.get('owner')!s:12} {req.get('step')!s:12} {req.get('priority')!s:>5} {n!s:>6} "
                  f"{req.get('max_chip_minutes')!s:>8}  {status}")
            continue
        if isinstance(items, Exception):
            print(f"{jid}  INVALID: {items}"); failed = True
            continue
        if a.cmd == "build" and d.exists():
            print(f"{jid}  already built: {d}")
            continue
        mins, refused = estimate(req, items)
        if refused:
            print(f"{jid}  REFUSED: {refused}"); failed = True
            continue
        if a.cmd == "estimate":
            print(f"{jid}  OK: estimate {mins:.2f} of {req['max_chip_minutes']} chip minutes (ESTIMATE)")
            continue
        m = make_pack.build(items, d, job_id=jid, owner=req["owner"], step=req["step"], seeds=req["seeds"], speed_bar=req["speed_bar"],
                            shards=a.shards, extra={"priority": req["priority"], "max_chip_minutes": req["max_chip_minutes"],
                                   "estimate_chip_minutes": round(mins, 3)})
        print(f"{jid}  built {d}: {m['n_items']} items, {m['shards']} shards, estimate {mins:.2f} chip min")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
