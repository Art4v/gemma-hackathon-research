"""Ingest Karthik's TPU verdicts for round-3 groups 5 and 6 (his GCP, oracle c9a07d9298a1 from karthik/kbk @ 86c31a9) into the local job layout the
assembler and check read: tpu/ops1/repo/oracle/jobs/kimi-r3-s<k>-k1/results/karthik.jsonl. Reads every origin/oracle-results/kimi-r3-s{5,6}* branch
(and oracle-results-all / karthik-* as fallbacks), keeps rows whose id is kimiR3/<pid> AND belongs to that group's pushed items file, asserts the oracle
sha and the ref sha, and keeps one row per id: the last non-halt-victim row (a halt-cascade harness_oom is not a verdict).
  python3 ingest_karthik.py   ->  counts per verdict"""
import collections, hashlib, json, pathlib, subprocess
W = pathlib.Path.home() / "Projects/gemma-work"; REPO = "/Users/hrudaynara/Projects/gemma-hackathon-research"
git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout
halt = lambda x: x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))
git("fetch", "-q", "origin")
branches = [b.strip() for b in git("branch", "-r").splitlines() if "kimi-r3" in b or b.strip() in ("origin/oracle-results-all",) or "karthik" in b]
for k in (5, 6):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(W / f"tpu/fleet/kimi-r3-s{k}.items.jsonl")}
    refsha = {i["task"]: hashlib.sha256(i["ref_code"].encode()).hexdigest() for i in items.values()}
    got = {}
    for b in branches:
        for path in git("ls-tree", "-r", "--name-only", b).splitlines():
            if not path.endswith(".jsonl") or "fixtures" in path:
                continue
            body = git("show", f"{b}:{path}")
            if "kimiR3/" not in body:
                continue
            for l in body.splitlines():
                if not l.strip() or '"meta": true' in l:
                    continue
                x = json.loads(l)
                if x.get("id") not in items or halt(x):
                    continue
                assert x.get("oracle_sha") == "c9a07d9298a1", (b, path, x.get("oracle_sha"))
                assert x.get("ref_sha256") in (None, refsha[x["task"]]), ("ref mismatch", x["id"])
                got[x["id"]] = x
    d = W / f"tpu/ops1/repo/oracle/jobs/kimi-r3-s{k}-k1/results"; d.mkdir(parents=True, exist_ok=True)
    (d / "karthik.jsonl").write_text("".join(json.dumps(x) + "\n" for x in got.values()))
    print(f"group {k}: {len(got)}/{len(items)} judged", dict(collections.Counter(x["verdict"] for x in got.values())))
