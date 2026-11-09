"""Overnight driver: SERA-style distillation rounds against the TPU oracle, plus Phase 2 problem production. The only git writer.

  python dataset/scripts/overnight.py [--poll 300] [--batch-size 120] [--max-round 3]

Two interleaved loops, both resumable from dataset/overnight_state.json:

Results loop (every --poll seconds): `git fetch`; for each batch whose submitted round has a new `origin/oracle-results/<job_id>`
branch, check out oracle/results/<job_id>, run oracle_collect.py, then (if open trajectories remain and round < max) generate.py the
next round, prescreen.py, one --resample pass for prescreen rejects, prescreen.py again, oracle_submit.py --enqueue, check
`oracle/jobq.py estimate`, commit + push to dataset-gen, and comment on issue #3. After every collection: assemble.py + verify_dataset.py.

Production loop (one stage at a time, as a subprocess, so heavy CPU work never overlaps): port_kernelbench (all L1+L2 survivors) ->
author_problems (3 per category) -> filters -> cpu_check -> form batches (<= --batch-size) and submit round 1 for each -> make_variants
-> filters -> cpu_check -> form batches for the variants and submit their round 1.

Rules held here: oracle files are never edited (oracle_sha checked before every submit); job ids dataset-<batch>-r<round> are unique;
feedback turns come verbatim from oracle rows (oracle_collect/generate); jobs stay near or under 60 chip-minutes via batch size.
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, pathlib, subprocess, sys, time, traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATASET, ORACLE, PROBLEMS, REPO, TRAJECTORIES, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402

PY = str(DATASET / ".venv" / "bin" / "python")
STATE = DATASET / "overnight_state.json"
LOG = DATASET / "overnight.log"
BRANCH, ISSUE, ORACLE_SHA = "dataset-gen", "3", "c9a07d9298a1"
import os as _os
EXTERNAL_GEN = set(filter(None, _os.environ.get("EXTERNAL_GEN", "b2").split(",")))
PRODUCTION = [
    ("port", [PY, "-u", str(HERE / "port_kernelbench.py"), "--level", "1,2", "--pick", "all", "--limit", "0", "--workers", "3", "--check-workers", "1", "--reuse-raw"]),
    ("author", [PY, "-u", str(HERE / "author_problems.py"), "--per-category", "3", "--workers", "3"]),
    ("filter1", [PY, "-u", str(HERE / "filter_contamination.py")]),
    ("neardup1", [PY, "-u", str(HERE / "filter_neardup.py")]),
    ("cpu1", [PY, "-u", str(HERE / "cpu_check.py"), "--timeout", "600"]),
    ("batches", None),   # ports + authored + scraped go to the TPU before the variants are built (Nov 8 04:xx reorder)
    ("variants", [PY, "-u", str(HERE / "make_variants.py")]),
    ("filter2", [PY, "-u", str(HERE / "filter_contamination.py")]),
    ("neardup2", [PY, "-u", str(HERE / "filter_neardup.py")]),
    ("cpu2", [PY, "-u", str(HERE / "cpu_check.py"), "--timeout", "600"]),
    ("batches_variants", None),
]


def log(msg: str) -> None:
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: list, timeout: int = 7200, check: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    tail = (p.stdout.strip().splitlines() or [""])[-1][:200]
    log(f"$ {' '.join(str(c) for c in cmd[1:4] if not str(c).startswith('-u'))} ... -> rc {p.returncode} | {tail}")
    if p.returncode and p.stderr.strip():
        log("  stderr: " + p.stderr.strip().splitlines()[-1][:300])
    if check and p.returncode:
        raise RuntimeError(f"{cmd[:4]} failed rc {p.returncode}: {(p.stderr or p.stdout)[-400:]}")
    return p


def oracle_sha() -> str:
    return hashlib.sha256(b"".join((ORACLE / f).read_bytes() for f in ("tpu_oracle.py", "tpu_check.py", "outside_ops.py"))).hexdigest()[:12]


def load_state() -> dict:
    if STATE.exists():
        return read_json(STATE)
    return {"batches": {"pilot": {"submitted": 1, "collected": 1, "done": False}}, "production": 0, "next_batch": 1, "finished": False}


def git_commit_push(msg: str) -> bool:
    sh(["git", "add", "-A", "dataset", "oracle/queue", "oracle/results"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        return False
    sh(["git", "commit", "-q", "-m", msg], check=True)
    for _ in range(3):
        if sh(["git", "push", "-q", "origin", BRANCH], timeout=300).returncode == 0:
            return True
        time.sleep(30)
    log("push failed three times; will retry on the next commit")
    return False


def comment(body: str) -> None:
    sh(["gh", "issue", "comment", ISSUE, "--body", body], timeout=120)


def open_count(batch: str) -> int:
    return sum(1 for t in read_jsonl(TRAJECTORIES / f"{batch}.jsonl") if t.get("status") == "open")


def submit_round(batch: str, rnd: int, st: dict) -> bool:
    """generate -> prescreen -> resample -> prescreen -> enqueue -> estimate -> push -> comment. False if nothing to submit."""
    if oracle_sha() != ORACLE_SHA:
        raise RuntimeError(f"oracle_sha is {oracle_sha()}, not {ORACLE_SHA}: refusing to submit")
    job = f"dataset-{batch}-r{rnd}"
    if (ORACLE / "queue" / f"{job}.json").exists() or (DATASET / "submissions" / f"{job}.request.json").exists():
        log(f"{job} already built; skipping (job ids must be unique)")
        return True
    g = [PY, "-u", str(HERE / "generate.py"), "--batch", batch, "--round", str(rnd), "--workers", "8", "--reuse-raw"]
    for attempt in range(3):  # constructor failures (rate limits) leave slots empty; generate.py is idempotent, so just rerun
        p = sh(g, timeout=6 * 3600)
        if "0 failed" in p.stdout or "0 requests" in p.stdout:
            break
        time.sleep(120)
    if not read_jsonl(DATASET / "candidates" / batch / f"r{rnd}.jsonl") and not read_jsonl(DATASET / "candidates" / batch / f"r{rnd}-human.jsonl"):
        log(f"{job}: no candidates; nothing to submit")
        return False
    sh([PY, "-u", str(HERE / "prescreen.py"), "--batch", batch, "--round", str(rnd)], timeout=3 * 3600)
    sh(g + ["--resample"], timeout=3 * 3600)
    sh([PY, "-u", str(HERE / "prescreen.py"), "--batch", batch, "--round", str(rnd)], timeout=3 * 3600)
    p = sh([PY, "-u", str(HERE / "oracle_submit.py"), "--batch", batch, "--round", str(rnd), "--enqueue"], timeout=3 * 3600)
    if p.returncode:
        return False
    est = sh([PY, str(ORACLE / "jobq.py"), "estimate", job], timeout=3 * 3600)
    if est.returncode:
        log(f"{job}: jobq.py estimate refused it; removing from the queue")
        for f in (ORACLE / "queue").glob(f"{job}*"):
            f.unlink()
        comment(f"`{job}`: built but `oracle/jobq.py estimate` refused it, so it was not queued. Needs a look:\n```\n{(est.stdout + est.stderr)[-800:]}\n```")
        return False
    meta = read_json(DATASET / "submissions" / f"{job}.meta.json")
    req = read_json(DATASET / "submissions" / f"{job}.request.json")
    pushed = git_commit_push(f"{job}: {meta['n_items']} items over {meta['n_tasks']} tasks (constructor round {rnd})")
    est_min = meta.get("estimate", {}).get("total_s")
    comment(f"Queued `{job}` on `{BRANCH}`: **{meta['n_items']} items** over {meta['n_tasks']} tasks, round {rnd}. "
            f"Roofline estimate {round(est_min / 60, 1) if est_min else '?'} min, max_chip_minutes {req['max_chip_minutes']}. "
            f"oracle_sha {ORACLE_SHA}.{'' if pushed else ' (push pending)'}")
    st["batches"].setdefault(batch, {"submitted": 0, "collected": 0, "done": False})["submitted"] = rnd
    return True


def handle_results(st: dict, max_round: int) -> bool:
    sh(["git", "fetch", "-q", "origin"], timeout=300)
    remote = set(subprocess.run(["git", "branch", "-r"], cwd=REPO, capture_output=True, text=True).stdout.split())
    did = False
    for batch, b in sorted(st["batches"].items()):
        if b["done"] or b["submitted"] <= b["collected"]:
            continue
        job = f"dataset-{batch}-r{b['submitted']}"
        if f"origin/oracle-results/{job}" not in remote:
            continue
        log(f"results for {job}")
        sh(["git", "checkout", f"origin/oracle-results/{job}", "--", f"oracle/results/{job}"], check=True)
        p = sh([PY, "-u", str(HERE / "oracle_collect.py"), job, "--jobs-dir", "oracle/results", "--results-dir", "oracle/results"])
        if p.returncode:
            # Nov 8: a partial branch (a shard crashed) gets completed by the operator's rerun: retry every poll, never drop the batch
            log(f"{job}: collection failed its checks; will retry when the results branch changes")
            if b.get("collect_failed") != job:
                comment(f"`{job}`: the results branch failed collection, so its next round waits; the driver retries every poll:\n```\n{(p.stdout + p.stderr)[-800:]}\n```")
                b["collect_failed"] = job
            continue
        b["collected"] = b["submitted"]
        summ = (p.stdout.strip().splitlines() or [""])[-1]
        q = (ORACLE / "queue" / f"{job}.json")
        for f in (ORACLE / "queue").glob(f"{job}*"):  # the job ran: take it off the queue so the watcher never sees it again
            f.unlink()
        did = True
        if b["collected"] < max_round and open_count(batch) and batch in EXTERNAL_GEN:
            # Nov 8: this batch's repairs are written by subagents outside the driver (claude -p on this account is too slow);
            # repair_package.py submits the round. Never marked done here.
            b["external"] = b["collected"] + 1
            log(f"{batch}: round {b['external']} left to external generation ({open_count(batch)} open)")
        elif b["collected"] < max_round and open_count(batch):
            if not submit_round(batch, b["collected"] + 1, st):
                b["done"] = True
        else:
            b["done"] = True
        assemble_verify()
        write_json(STATE, st)
        git_commit_push(f"collect {job}: {summ[:120]}")
    return did


def assemble_verify() -> None:
    if sh([PY, "-u", str(HERE / "assemble.py"), "--phase", "full"], timeout=1800).returncode == 0:
        sh([PY, "-u", str(HERE / "verify_dataset.py"), str(DATASET / "final")], timeout=1800)


def form_batches(st: dict, size: int) -> list:
    rows = read_jsonl(PROBLEMS / "problems.jsonl")
    ready = [r for r in rows if r["status"] == "ok" and not r.get("batch") and (r.get("cpu_check") or {}).get("passed")]
    # group by source so each job is homogeneous: authored + scraped first (the exam's hard families), then ports, then variants
    order = {"authored": 0, "scraped": 0, "kernelbench": 1}
    ready.sort(key=lambda r: (order.get(r["source"], 2), r["problem_id"].endswith("__bf16"), r["problem_id"]))
    names = []
    for i in range(0, len(ready), size):
        name = f"b{st['next_batch']}"
        st["next_batch"] += 1
        ids = {r["problem_id"] for r in ready[i:i + size]}
        for r in rows:
            if r["problem_id"] in ids:
                r["batch"] = name
        st["batches"][name] = {"submitted": 0, "collected": 0, "done": False, "n": len(ids)}
        names.append(name)
    write_jsonl(PROBLEMS / "problems.jsonl", rows)
    log(f"formed batches {names} from {len(ready)} new problems")
    return names


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--poll", type=int, default=300); ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--max-round", type=int, default=3)
    a = ap.parse_args(argv)
    st = load_state()
    log(f"overnight driver start; state {json.dumps({k: v for k, v in st.items() if k != 'batches'})}, batches {st['batches']}")
    proc, stage = None, None
    while True:
        try:
            # production: one subprocess at a time
            if proc is not None and proc.poll() is not None:
                log(f"production stage {stage} finished rc {proc.returncode}")
                proc = None
                st["production"] += 1
                write_json(STATE, st)
                git_commit_push(f"phase 2: {stage} stage done")
            if proc is None and st["production"] < len(PRODUCTION):
                stage, cmd = PRODUCTION[st["production"]]
                if cmd is None:  # batches: form, then round 1 for each, sequentially
                    for name in form_batches(st, a.batch_size):
                        write_json(STATE, st)
                        if not submit_round(name, 1, st):
                            st["batches"][name]["done"] = True
                        write_json(STATE, st)
                        handle_results(st, a.max_round)  # do not starve result handling while round-1 jobs are built
                    st["production"] += 1
                    write_json(STATE, st)
                    git_commit_push("phase 2: batches formed and round 1 submitted")
                else:
                    log(f"production stage {stage} starting")
                    proc = subprocess.Popen(cmd, cwd=REPO, stdout=open(DATASET / f"overnight_{stage}.log", "w"), stderr=subprocess.STDOUT)
            handle_results(st, a.max_round)
            write_json(STATE, st)
            if proc is None and st["production"] >= len(PRODUCTION) and all(b["done"] for b in st["batches"].values()):
                assemble_verify()
                st["finished"] = True
                write_json(STATE, st)
                git_commit_push("overnight: all batches done; final dataset assembled")
                stats = read_json(DATASET / "final" / "stats.json") if (DATASET / "final" / "stats.json").exists() else {}
                comment("Overnight run finished: every batch reached round 3 or had nothing left to repair. "
                        f"Final dataset assembled on `{BRANCH}` (`dataset/final/`). Stats: ```{json.dumps(stats)[:1500]}```")
                log("finished")
                return
        except Exception:
            log("ERROR " + traceback.format_exc()[-1500:])
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
