"""Offline pre-flight for a JAXBench lora run (jaxbench_modal.sh): spends nothing, posts nothing.
Fails if the v7 run would not be like-for-like with v4final-e2 / v5full, or could mix in stale rows or a stale adapter.

  python3 -I preflight_v7.py --root <repo root holding control/> --model gemma4-e4b-lora:v7 --prefix lora-v7-modal \
      --digest <EXPECT_DIGEST> [--oracle <dir with tpu_oracle.py>] [--jobs <ops1/repo/oracle/jobs>]
"""
import argparse, hashlib, json, pathlib, re, sys

PROMPT_SHA = "b186365a1c3a56dcfedc3fbca3080c1323928544da6fdcaece1d9a4310a1827a"  # every v4final-e2 and v5full row
ORACLE_SHA = "c9a07d9298a1"
OLD_DIGESTS = ("ee6656371218", "0f764d3f6798", "641ddde330ed", "2a99a2f97822")  # base, v4 Modal, v5 Modal, v4 Mac
REF_RUN = "control/runs/gemma4-e4b-lora_v5full__modal/bare.jsonl"  # its ref_sha256 per task is the v4/v5 exam


def main():
    ap = argparse.ArgumentParser()
    for k in ("root", "model", "prefix", "digest"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--oracle"); ap.add_argument("--jobs")
    a = ap.parse_args()
    root, bad = pathlib.Path(a.root).resolve(), []
    sys.path.insert(0, str(root / "control"))
    import bench  # repo code (stdlib only), not dataset rows

    d = a.digest.lower()
    if len(d) < 12 or not re.fullmatch(r"[0-9a-f]+", d):
        bad.append(f"EXPECT_DIGEST {a.digest!r}: need >= 12 hex chars (check_rows matches by prefix)")
    if any(d.startswith(o) or o.startswith(d) for o in OLD_DIGESTS):
        bad.append(f"EXPECT_DIGEST {a.digest!r} is a base/v4/v5 digest: a stale adapter would pass check_rows")
    if bench.prompt_sha256("bare") != PROMPT_SHA:
        bad.append(f"prompt_sha256 {bench.prompt_sha256('bare')} != v4/v5 {PROMPT_SHA}")
    if (bench.NUM_CTX, bench.MAX_TOKENS, bench.TURNS, bench.TERMINAL) != (65536, 8192, 3, {"pass", "too_large_for_chip", "harness_oom"}):
        bad.append(f"bench settings changed: {bench.NUM_CTX, bench.MAX_TOKENS, bench.TURNS, bench.TERMINAL}")
    body = bench.request_body([], 0, a.model)
    if body["think"] is not False or body["options"]["temperature"] != 0:
        bad.append(f"request body changed: {body}")
    want = {r["task"]: r["ref_sha256"] for r in map(json.loads, (root / REF_RUN).read_text().splitlines()) if r}
    now = {t: hashlib.sha256(ref.encode()).hexdigest() for t, (ref, _) in bench.tasks().items()}
    if now != want:
        bad.append(f"task set/refs differ from v4/v5: missing {sorted(set(want) - set(now))[:5]}, extra {sorted(set(now) - set(want))[:5]}, "
                   f"changed {[t for t in want if t in now and now[t] != want[t]][:5]}")
    bench.BACKEND = "modal"
    run = bench.run_dir(a.model)
    if any(run.glob("*.jsonl")):
        bad.append(f"{run} already has rows: generate resumes them (stale adapter / earlier attempt), delete or rename first")
    stale = sorted(p.name for p in (root / "control/runs/cpu_checks").glob(f"{a.prefix}-t*"))
    if stale:
        bad.append(f"stale CPU step-1 files for PREFIX {a.prefix}: {stale[:3]}")
    if not (root / "control/cpu_check.py").exists():
        bad.append(f"{root}/control/cpu_check.py missing: jaxbench_modal.sh calls it every turn and ignores its failure")
    if a.oracle:
        o = pathlib.Path(a.oracle)
        sha = hashlib.sha256(b"".join((o / p).read_bytes() for p in ("tpu_oracle.py", "tpu_check.py", "outside_ops.py"))).hexdigest()[:12]
        if sha != ORACLE_SHA:
            bad.append(f"oracle sha {sha} != {ORACLE_SHA}")
    if a.jobs:
        old = sorted(p.name for p in pathlib.Path(a.jobs).glob(f"control-{a.prefix}-t*"))
        if old:
            bad.append(f"oracle job dirs already exist for this PREFIX (the merge would read them): {old[:3]}")
    print("\n".join(["FAIL " + b for b in bad] or ["OK: settings, prompt, refs, digest and run dir are like-for-like and fresh"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
