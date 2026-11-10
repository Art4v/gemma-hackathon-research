"""Run one oracle round to completion despite XLA compiler crashes (a crash kills the oracle process, so ops_loop stops SHORT).
Each part runs the not-yet-judged items; on a short stop the next unjudged item (pack order) is the suspect and is set aside.
Suspects then run alone, one job each. A suspect that crashes alone too is excluded as harness_crash (two crashes, the team rule).
  ZONE=europe-west4-a python3 crashloop.py BASE ITEMS.jsonl [--done RESULTS.jsonl ...] [--suspect ID ...]
Light on the Mac: it only packs, pushes and polls (ops_loop.sh); the work runs on the VM.
"""
import argparse, json, os, pathlib, subprocess

S = pathlib.Path(__file__).resolve().parent
JOBS = S / "repo/oracle/jobs"


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if pathlib.Path(p).exists() else []


def judged(job):
    return {r["id"] for r in jsonl(JOBS / job / "results" / f"{job}.jsonl") if not r.get("meta") and "id" in r}


def run(job, items):
    src = S / f"{job}.items.jsonl"
    src.write_text("".join(json.dumps(i) + "\n" for i in items))
    subprocess.run(["python3", "oracle/make_pack.py", str(src), f"oracle/jobs/{job}", "--job-id", job, "--owner", "karthik", "--step", job],
                   cwd=S / "repo", check=True, capture_output=True)
    subprocess.run(["bash", os.environ.get("RUNNER", "oracle/vm.sh"), "push", f"oracle/jobs/{job}"], cwd=S / "repo", check=True, capture_output=True)
    subprocess.run(["bash", "ops_loop.sh", job], cwd=S)   # exit 2 on SHORT is expected here
    return judged(job)


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
        k += 1
        got = run(f"{a.base}-p{k}", todo)
        done |= got
        left = [i["id"] for i in todo if i["id"] not in got]
        print(f"part {k}: {len(got)} judged, {len(left)} left", flush=True)
        if left:
            suspects.append(left[0])
    for n, sid in enumerate(suspects, 1):
        got = run(f"{a.base}-x{n}", [i for i in items if i["id"] == sid])
        crashes[sid] = "judged alone" if sid in got else "harness_crash (crashed in a part and again alone)"
        print(f"suspect {sid}: {crashes[sid]}", flush=True)
    (S / f"{a.base}.crashes.json").write_text(json.dumps(crashes, indent=1))
    print("ROUND DONE", a.base, f"{len(done)} judged in parts,", sum(v == 'judged alone' for v in crashes.values()), "suspects judged alone")


if __name__ == "__main__":
    assert os.environ.get("ZONE") or os.environ.get("RUNNER") == "oracle/colab.sh", "set ZONE (or RUNNER=oracle/colab.sh)"
    main()
