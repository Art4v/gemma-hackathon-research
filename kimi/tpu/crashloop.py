"""Run one oracle round to completion despite XLA compiler crashes (a crash kills the oracle process, so ops_loop stops SHORT).
Each part runs the not-yet-judged items; on a short stop the next unjudged item in the oracle's order (sorted by task) is the
suspect and is set aside.
Suspects then run alone, one job each. A suspect that crashes alone too is excluded as harness_crash (two crashes, the team rule).
  ZONE=europe-west4-a python3 crashloop.py BASE ITEMS.jsonl [--done RESULTS.jsonl ...] [--suspect ID ...]
Light on the Mac: it only packs, pushes and polls (ops_loop.sh); the work runs on the VM.
"""
import argparse, json, os, pathlib, subprocess

S = pathlib.Path(__file__).resolve().parent
JOBS = S / "repo/oracle/jobs"


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if pathlib.Path(p).exists() else []


HALT = ("halted unexpectedly", "RuntimeUnexpectedCoreHalt")


def halt_victim(r):
    """Nov 11: once one kernel halts the TPU core (its own verdict: runtime), every later item in the same process fails in the
    reference with 'program continuator has halted unexpectedly' / RuntimeUnexpectedCoreHalt and is labelled harness_oom. Those rows
    are not verdicts on those kernels: they are re-run in a fresh process (the next part), like a crash."""
    return r.get("verdict") == "harness_oom" and any(h in (r.get("feedback") or "") for h in HALT)


def rows(job):
    return [r for r in jsonl(JOBS / job / "results" / f"{job}.jsonl") if not r.get("meta") and "id" in r]


def judged(job):
    return {r["id"] for r in rows(job) if not halt_victim(r)}


def crashed(job):
    """True if the shard log fetched back from the VM shows the oracle process died on a signal (an XLA/Mosaic compiler crash), as
    opposed to a lost session (no log, or a clean log). Nov 11: a Kimi kernel segfaulted XLA on the FIRST item of a part, so the part
    had 0 rows and the old rule ("0 rows twice = check the session") stopped the round instead of isolating the item."""
    for f in (JOBS / job / "logs").glob("shard*.log"):
        t = f.read_text(errors="replace")
        if "Raising signal" in t or "coredump_hook" in t or "Fatal Python error" in t:
            return True
    return False


def free_name(base, k):
    """The first {base}-p{k}, {base}-p{k+1}, ... with no job directory yet: make_pack builds a directory once, and an aborted earlier
    attempt (Nov 10: a Colab run) can leave one behind."""
    while (JOBS / f"{base}-p{k}").exists():
        k += 1
    return k


def run(job, items):
    src = S / f"{job}.items.jsonl"
    src.write_text("".join(json.dumps(i) + "\n" for i in items))
    subprocess.run(["python3", "oracle/make_pack.py", str(src), f"oracle/jobs/{job}", "--job-id", job, "--owner", "hruday", "--step", job,
                    "--shards", os.environ.get("SHARDS", "1")],   # SHARDS=8 on a v6e-8: one shard per chip
                   cwd=S / "repo", check=True, capture_output=True)
    for attempt in range(4):  # Nov 11: a transient SSH/scp failure killed whole batches (vm5 r1, a2vm2); retry before giving up
        pr = subprocess.run(["bash", os.environ.get("RUNNER", str(S / "vm8.sh")), "push", f"oracle/jobs/{job}"], cwd=S / "repo", capture_output=True)
        if pr.returncode == 0:
            break
        print(f"push {job} failed (attempt {attempt + 1}): {(pr.stderr or b'').decode(errors='replace')[-200:]}", flush=True)
        import time; time.sleep(45)
    else:
        raise SystemExit(f"push {job} failed 4 times; the VM is unreachable")
    subprocess.run(["bash", "ops_loop.sh", job], cwd=S)   # exit 2 on SHORT is expected here
    sync(job)
    return judged(job)


def sync(job):
    """Nov 11: an oracle process can judge every item, then hang at exit and be SIGTERMed; the rows are in shards/*.jsonl (written and
    flushed per item) but never collected into results/<job>.jsonl, so the part looked like '0 rows + crash' and was re-run in full
    (a2mop3: the same 74 kernels judged three times). Collect every shard row the results file lacks."""
    res = JOBS / job / "results" / f"{job}.jsonl"
    have = {r["id"] for r in jsonl(res) if not r.get("meta") and "id" in r}
    meta, extra = None, []
    for f in sorted((JOBS / job / "shards").glob("*.jsonl")):
        for r in jsonl(f):
            if r.get("meta"):
                meta = meta or r
            elif "id" in r and r["id"] not in have:
                extra.append(r); have.add(r["id"])
    if extra:
        res.parent.mkdir(parents=True, exist_ok=True)
        with res.open("a") as fh:
            if res.stat().st_size == 0 and meta:
                fh.write(json.dumps(meta) + "\n")
            fh.write("".join(json.dumps(r) + "\n" for r in extra))
        print(f"{job}: {len(extra)} rows collected from shards (the process was killed after judging them)", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("items")
    ap.add_argument("--done", nargs="*", default=[]); ap.add_argument("--suspect", nargs="*", default=[])
    a = ap.parse_args()
    items = jsonl(a.items)
    done = set().union(*(set(r["id"] for r in jsonl(p) if not r.get("meta") and "id" in r) for p in a.done)) if a.done else set()
    suspects, crashes, k = list(a.suspect), {}, 0
    while True:
        todo = [i for i in items if i["id"] not in done and i["id"] not in suspects]
        if not todo:
            break
        k = free_name(a.base, k + 1)
        job = f"{a.base}-p{k}"
        got = run(job, todo)
        # tpu_oracle.py judges items sorted by task (stable), not in pack order (Nov 11: blaming the pack-order first item set aside
        # an innocent kernel each part while the real crasher stayed in, 14 parts in a row on a2mop-s3)
        order = [i["id"] for i in sorted(todo, key=lambda i: i["task"])]
        seen = {r["id"] for r in rows(job)}
        no_row = [i for i in order if i not in seen]   # a crash stops the process at the first of these
        if not got and (crashed(job) or seen):
            sid = no_row[0] if (crashed(job) and no_row) else order[0]
            suspects.append(sid)
            why = "a crash in the shard log" if crashed(job) else "only halt-victim rows"
            print(f"part {k}: 0 judged and {why}: first item {sid} set aside as a suspect", flush=True)
            continue
        if not got:
            # 0 rows is not a crash signal (a crash still leaves the earlier rows): a hung or timed-out fetch, NOSTART, or a lost
            # session. Retry the part once before blaming its first item.
            k = free_name(a.base, k + 1)
            job = f"{a.base}-p{k}"
            got = run(job, todo)
            if not got:
                raise SystemExit(f"part {k}: 0 rows twice; check the TPU session before continuing")
            seen = {r["id"] for r in rows(job)}
            no_row = [i for i in order if i not in seen]
        done |= got
        left = [i for i in order if i not in got]
        print(f"part {k}: {len(got)} judged, {len(left)} left ({len(left) - len(no_row)} halt victims to re-run)", flush=True)
        halted = any(any(h in (r.get("feedback") or "") for h in HALT) for r in rows(job))
        if no_row and not halted:  # stopped short: the first item with no row at all crashed the process
            suspects.append(no_row[0])
        # a process that already halted the core (halter judged: runtime) often aborts later on its own: that abort is the halt's
        # doing, not the next kernel's, so nobody is set aside and the rest simply re-run in a fresh process (progress: >= 1 judged)
    for n, sid in enumerate(suspects, 1):
        got = run(f"{a.base}-x{n}", [i for i in items if i["id"] == sid])
        crashes[sid] = "judged alone" if sid in got else "harness_crash (crashed in a part and again alone)"
        print(f"suspect {sid}: {crashes[sid]}", flush=True)
    (S / f"{a.base}.crashes.json").write_text(json.dumps(crashes, indent=1))
    print("ROUND DONE", a.base, f"{len(done)} judged in parts,", sum(v == 'judged alone' for v in crashes.values()), "suspects judged alone")


if __name__ == "__main__":
    assert os.environ.get("ZONE") or os.environ.get("RUNNER") == "oracle/colab.sh", "set ZONE (or RUNNER=oracle/colab.sh)"
    main()
