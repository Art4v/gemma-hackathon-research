"""Run the control benchmark (bare and docs arms) to the end, unattended: generate on Modal, judge on the CPU what the CPU can decide, judge
the rest on a Colab TPU v6e, and go round until every task has finished its 3 turns. Writes a log, the summary and a report (nothing is committed).

  caffeinate -i python3 control/overnight.py --py <JAX 0.10.2 python> [--start-anyway-at 05:00] [--dry-run]
  LoRA arm: ... --model gemma4-e4b-lora:pilot25 --arms bare --prefix lora-pilot25-
  (--prefix keeps that run's CPU-check tags and TPU job names apart from the base run's: check*, control-tpu*)

The loop, until nothing is left:
  1 if any attempt can be generated (a task whose last attempt has a verdict and turns left), generate them on Modal, both arms at once
  2 else if an attempt waits for a verdict and has not been CPU-checked, run control/cpu_check.py (ingests what step 1 decides)
  3 else the waiting attempts all passed the CPU: one Colab TPU session judges them (and checks the level 2 references, once), then stops
Colab: our session is "control-overnight". Another TPU or GPU session on the account (the user's) is waited for, until --start-anyway-at; then ours
starts beside it. Our session is always stopped, on success or failure. Every step retries a few times and every action is logged.
"""
import argparse, datetime, json, os, pathlib, re, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

bench.BACKEND = "modal"
MODEL, ARMS, PREFIX = bench.MODEL, bench.ARMS, ""  # set from --model, --arms, --prefix in main(); the defaults are the base run's
RUN = bench.run_dir(MODEL)
LOG = RUN / "overnight.log"
SESSION = "control-overnight"  # not colab.sh's default "oracle-colab": the user's own session had that name (Nov 8)
ENV = {**os.environ, "MODAL_PROFILE": os.environ.get("BENCH_WORKSPACE", "splashgaming532"), "NAME": SESSION}  # Nov 9: LoRA arm on codecraftacademy981
MAX_TPU_SESSIONS, MAX_GEN_RETRIES, JOB_HOURS = 8, 3, 4


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: list, timeout=None, check=True, quiet=False) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    if not quiet:
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        log(f"$ {' '.join(map(str, cmd))[:160]} -> exit {p.returncode}" + (f" | {' / '.join(tail)[:300]}" if tail else ""))
    if check and p.returncode:
        raise RuntimeError(f"{cmd[:4]} failed with exit {p.returncode}")
    return p


def retry(fn, what: str, tries=3, wait=60):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log(f"{what}: attempt {k + 1}/{tries} failed: {type(e).__name__}: {str(e)[:200]}")
            if k + 1 == tries:
                raise
            time.sleep(wait)


# ---- state, read fresh from the run's files every time

def pending_generation() -> int:
    vs, all_tasks = bench.verdicts(MODEL), bench.tasks()
    return sum(len(bench.pending(arm, bench.read_jsonl(RUN / f"{arm}.jsonl"), vs, all_tasks)) for arm in ARMS)


def waiting() -> list:
    return [i for i in bench.items(MODEL) if i["id"].split("/")[0] in ARMS]


def cpu_passed() -> set:
    """Ids that passed the CPU check (step 1) in any earlier check: they wait for the TPU."""
    pattern = f"{PREFIX}check*_tpu_items.jsonl" if PREFIX else "check*_tpu_items.jsonl"  # item ids carry no model: keep runs apart by tag
    return {json.loads(l)["id"] for p in (ROOT / "control/runs/cpu_checks").glob(pattern) for l in p.read_text().splitlines() if l.strip()}


# ---- steps

def generate() -> None:
    def once():
        procs = [subprocess.Popen([sys.executable, "control/bench.py", "--backend", "modal", "--model", MODEL, "generate", "--arm", arm, "--workers", "3"],
                                  cwd=ROOT, env=ENV, stdout=open(RUN / f"gen_{arm}.log", "a"), stderr=subprocess.STDOUT) for arm in ARMS]
        rcs = [p.wait() for p in procs]
        if any(rcs):
            raise RuntimeError(f"generate exit codes {rcs} (see {RUN}/gen_*.log)")
    n = pending_generation()
    log(f"generate on Modal: {n} attempts")
    retry(once, "generate", MAX_GEN_RETRIES, 120)


def cpu_check(tag: str, py: str) -> None:
    log(f"CPU check {tag}: {len(waiting())} attempts waiting")
    p = retry(lambda: sh([sys.executable, "control/cpu_check.py", "--backend", "modal", "--model", MODEL, "--tag", tag, "--py", py, "--ingest"], timeout=7200),
              "cpu_check")
    for line in p.stdout.splitlines():
        log(f"  {line}")


def other_sessions() -> list:
    out = sh(["colab", "sessions"], timeout=120, check=False, quiet=True).stdout
    return [l for l in out.splitlines() if "Hardware" in l and "Hardware: CPU" not in l and f"[{SESSION}]" not in l]  # TPU/GPU only


def wait_for_other_session(start_anyway_at: str) -> None:
    hh, mm = map(int, start_anyway_at.split(":"))
    now = datetime.datetime.now()
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline < now - datetime.timedelta(hours=12):
        deadline += datetime.timedelta(days=1)
    while True:
        others = other_sessions()
        if not others:
            return
        if datetime.datetime.now() >= deadline:
            log(f"another Colab session is still running at {start_anyway_at}: starting ours beside it ({others[0][:80]})")
            return
        log(f"waiting for the other Colab session to finish: {others[0][:80]}")
        time.sleep(300)


def colab(*args, timeout=3600, check=True):
    return sh(["bash", "oracle/colab.sh", *args], timeout=timeout, check=check)


def run_on_tpu(job: str) -> None:
    colab("run", job, timeout=300)
    t0 = time.time()
    while True:
        time.sleep(90)
        p = retry(lambda: colab("status", job, timeout=300, check=False), f"status {job}")
        m = re.search(r"rows written: (\d+); (running|not running)", p.stdout)
        if m and m.group(2) == "not running":
            log(f"{job}: finished, {m.group(1)} rows")
            return
        if time.time() - t0 > JOB_HOURS * 3600:
            raise RuntimeError(f"{job} still running after {JOB_HOURS} h")


def clock(hhmm: str) -> datetime.datetime:
    """Today's (or, if more than 12 h past, tomorrow's) datetime at HH:MM."""
    hh, mm = map(int, hhmm.split(":"))
    now = datetime.datetime.now()
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t + datetime.timedelta(days=1) if t < now - datetime.timedelta(hours=12) else t


def start_until(give_up_at: str) -> None:
    """colab.sh start, every 10 minutes until it works or give_up_at: Colab answers 'Service Unavailable' when no v6e is free."""
    deadline = clock(give_up_at)
    while True:
        try:
            colab("start", timeout=1800)
            return
        except Exception as e:  # noqa: BLE001
            colab("stop", timeout=600, check=False)  # a half-made session must not be left billing
            if datetime.datetime.now() >= deadline:
                raise RuntimeError(f"no Colab TPU v6e by {give_up_at}") from e
            log("no TPU v6e on Colab right now (Service Unavailable or similar): asking again in 10 minutes")
            time.sleep(600)


def tpu_session(k: int, items: list, verify_needed: bool, start_anyway_at: str, give_up_at: str) -> None:
    job = f"{PREFIX}tpu{k}" if PREFIX else f"control-tpu{k}"
    items_path = ROOT / f"control/runs/cpu_checks/{job}_items.jsonl"
    items_path.write_text("".join(json.dumps(i) + "\n" for i in items))
    if (ROOT / f"oracle/jobs/{job}").exists():
        subprocess.run(["rm", "-rf", f"oracle/jobs/{job}"], cwd=ROOT, check=True)
    sh([sys.executable, "oracle/make_pack.py", str(items_path), f"oracle/jobs/{job}", "--job-id", job, "--owner", "overnight", "--step", job])
    jobs = (["verify-refs-1"] if verify_needed else []) + [job]
    if os.environ.get("TPU_MODE") == "queue":  # Nov 9: Colab has no v6e; the team's TPU operator (Karthik's agent) runs queued jobs
        queue_session(job, items_path, len(items))
        sh([sys.executable, "oracle/collect.py", f"oracle/jobs/{job}"], check=False)
        sh([sys.executable, "control/bench.py", "--backend", "modal", "--model", MODEL, "ingest", f"oracle/jobs/{job}/results/{job}.jsonl"])
        return
    wait_for_other_session(start_anyway_at)
    log(f"TPU session {k}: {len(items)} kernels in {job}" + (" + the reference check" if verify_needed else ""))
    try:
        start_until(give_up_at)
        retry(lambda: colab("setup", timeout=2400), "colab setup", 3, 60)
        log(f"compute units: {sh(['colab', 'usage'], timeout=120, check=False, quiet=True).stdout.strip()}")
        retry(lambda: colab("push", *[f"oracle/jobs/{j}" for j in jobs], timeout=1800), "colab push", 3, 60)
        for j in jobs:
            run_on_tpu(j)
            retry(lambda: colab("fetch", j, timeout=1800), f"fetch {j}", 3, 60)
    finally:
        colab("stop", timeout=600, check=False)
        log(f"Colab session stopped; sessions now: {other_sessions() or 'none of ours'}")
    if verify_needed:
        p = sh([sys.executable, "control/verify_refs.py", "report", "oracle/jobs/verify-refs-1/results/verify-refs-1.jsonl"])
        (RUN / "verify_refs_report.txt").write_text(p.stdout)
        for line in p.stdout.splitlines():
            log(f"  refs: {line}")
    sh([sys.executable, "oracle/collect.py", f"oracle/jobs/{job}"], check=False)
    sh([sys.executable, "control/bench.py", "--backend", "modal", "--model", MODEL, "ingest", f"oracle/jobs/{job}/results/{job}.jsonl"])


QUEUE_BRANCH = os.environ.get("QUEUE_BRANCH", "eval/lora-v1")


def queue_session(job: str, items_path: pathlib.Path, n: int) -> None:
    """Queue `job` for the team's TPU operator: a jobq request + items on QUEUE_BRANCH (a worktree, so main's checkout is untouched),
    a comment on #3, then wait for origin/oracle-results/<job> and copy its rows into oracle/jobs/<job>/results/<job>.jsonl."""
    wt = pathlib.Path(f"/tmp/eval-queue-{job}")
    sh(["git", "fetch", "-q", "origin"], timeout=300, check=False)
    sh(["rm", "-rf", str(wt)], check=False); sh(["git", "worktree", "prune"], check=False)
    have = sh(["git", "ls-remote", "origin", f"refs/heads/{QUEUE_BRANCH}"], timeout=120, check=False).stdout.strip()
    sh(["git", "worktree", "add", "-q", "--detach", str(wt), f"origin/{QUEUE_BRANCH}" if have else "origin/main"])
    q = wt / "oracle" / "queue"; q.mkdir(parents=True, exist_ok=True)
    (q / f"{job}.items.jsonl").write_text(items_path.read_text())
    req = {"job_id": job, "owner": "hnaracodes", "step": job, "items_path": f"{job}.items.jsonl", "seeds": 3, "speed_bar": 0.7,
           "priority": 0, "max_chip_minutes": max(10, 2 * n)}
    (q / f"{job}.json").write_text(json.dumps(req, indent=1) + "\n")
    sh(["git", "-C", str(wt), "add", "oracle/queue"]); sh(["git", "-C", str(wt), "commit", "-q", "-m", f"{job}: {n} items ({MODEL}, eval)"])
    sh(["git", "-C", str(wt), "push", "-q", "origin", f"HEAD:refs/heads/{QUEUE_BRANCH}"], timeout=300)
    body = (f"@LordKarV (from Hruday's agent, posting as hnaracodes) **`{job}` is queued** on branch `{QUEUE_BRANCH}` (`oracle/queue/{job}.json`): "
            f"**{n} items**, the `{MODEL}` eval attempts that passed oracle step 1 on CPU. oracle_sha c9a07d9298a1, 3 seeds, speed bar 0.7. "
            f"Please claim it and push results to `oracle-results/{job}`. If an item crashes the oracle twice, leave it out and name it here.")
    sh(["gh", "issue", "comment", "3", "--body", body], timeout=120, check=False)
    log(f"{job}: {n} items queued on {QUEUE_BRANCH}; waiting for oracle-results/{job}")
    while True:
        sh(["git", "fetch", "-q", "origin"], timeout=300, check=False)
        if sh(["git", "ls-remote", "origin", f"refs/heads/oracle-results/{job}"], timeout=120, check=False).stdout.strip():
            break
        time.sleep(120)
    dest = ROOT / "oracle" / "jobs" / job / "results"; dest.mkdir(parents=True, exist_ok=True)
    files = sh(["git", "ls-tree", "-r", "--name-only", f"origin/oracle-results/{job}"], check=False).stdout.split()
    rows = next((f for f in files if f.endswith(f"/{job}.jsonl")), None)
    if rows is None:
        raise RuntimeError(f"oracle-results/{job} has no {job}.jsonl: {files[:10]}")
    (dest / f"{job}.jsonl").write_text(sh(["git", "show", f"origin/oracle-results/{job}:{rows}"], quiet=True).stdout)
    man = next((f for f in files if f.endswith("MANIFEST.json")), None)
    if man:
        (ROOT / "oracle" / "jobs" / job / "MANIFEST.json").write_text(sh(["git", "show", f"origin/oracle-results/{job}:{man}"], quiet=True).stdout)
    log(f"{job}: results fetched from oracle-results/{job}")
    sh(["git", "worktree", "remove", "--force", str(wt)], check=False)


def report() -> None:
    s = {arm: x for arm, x in bench.summary(MODEL).items() if arm in ARMS}
    (RUN / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    refs = (RUN / "verify_refs_report.txt").read_text() if (RUN / "verify_refs_report.txt").exists() else "(not run)"
    vs = bench.verdicts(MODEL)
    cpu = sum(1 for v in vs.values() if str(v.get("judged_on", "")).startswith("cpu"))
    lines = [f"# Control benchmark: {MODEL} on Modal ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
             f"Verdicts: {len(vs)} ({cpu} decided on the CPU at oracle step 1, {len(vs) - cpu} on the TPU).", "",
             "| arm | tasks judged | pass@1 | pass within 3 | correct within 3 | parity (>= 0.97x) | median speed of correct |", "|---|---|---|---|---|---|---|"]
    for arm, x in s.items():
        lines.append(f"| {arm} | {x['judged']} / {x['tasks']} | {x['pass_at_1']} | {x['pass_within_3']} | {x['correct_within_3']} | "
                     f"{x['parity_within_3']} | {x['median_speed_correct']} |")
    for arm, x in s.items():
        lines += ["", f"{arm}, turn 0 verdicts: {x['turn0_verdicts']}", f"{arm}, final verdicts: {x['final_verdicts']}"]
    left = waiting()
    if left:
        lines += ["", f"Not judged (passed the CPU check, but no TPU verdict): {len(left)}", *[f"- {i['id']}" for i in left]]
    lines += ["", "Level 2 reference check:", "```", refs.strip(), "```"]
    (RUN / "REPORT.md").write_text("\n".join(lines) + "\n")
    log("report written: " + str(RUN / "REPORT.md"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--py", required=True); ap.add_argument("--start-anyway-at", default="05:00")
    ap.add_argument("--tpu-give-up-at", default="08:00", help="stop asking Colab for a v6e at this time and write the report")
    ap.add_argument("--dry-run", action="store_true", help="say the next step and stop")
    ap.add_argument("--model", default=bench.MODEL, help="the Ollama tag to generate with (default: the base model)")
    ap.add_argument("--arms", nargs="+", choices=bench.ARMS, default=list(bench.ARMS))
    ap.add_argument("--prefix", default="", help="CPU-check tag and TPU job prefix for a non-base run, e.g. lora-pilot25-")
    a = ap.parse_args(argv)
    global MODEL, ARMS, PREFIX, RUN, LOG
    MODEL, ARMS, PREFIX = a.model, tuple(a.arms), a.prefix
    assert PREFIX or MODEL == bench.MODEL, "a non-base model needs --prefix, or its CPU-check tags and TPU jobs collide with the base run's"
    RUN = bench.run_dir(MODEL); LOG = RUN / "overnight.log"
    RUN.mkdir(parents=True, exist_ok=True)
    verify_needed = not (ROOT / "oracle/jobs/verify-refs-1/results/verify-refs-1.jsonl").exists()
    tpu_k = len(list((ROOT / "oracle/jobs").glob(f"{PREFIX}tpu*"))) if PREFIX else 0
    check_k = len(list((ROOT / "control/runs/cpu_checks").glob(f"{PREFIX}check*_static.jsonl" if PREFIX else "*_static.jsonl")))
    log(f"overnight start (dry run: {a.dry_run})")
    while True:
        while not a.dry_run and subprocess.run(["pgrep", "-f", "bench.py --backend modal.* generate"], capture_output=True).returncode == 0:
            log("a generator started by hand is still running: waiting for it"); time.sleep(60)
        n, w = pending_generation(), waiting()
        unchecked = [i for i in w if i["id"] not in cpu_passed()]
        step = "generate" if n else "cpu_check" if unchecked else "tpu" if w else "done"
        log(f"state: {n} to generate, {len(w)} waiting ({len(unchecked)} not CPU-checked) -> {step}")
        if a.dry_run:
            return
        if step == "generate":
            generate()
        elif step == "cpu_check":
            check_k += 1
            cpu_check(f"{PREFIX}check{check_k}", a.py)
        elif step == "tpu":
            tpu_k += 1
            if tpu_k > MAX_TPU_SESSIONS:
                log(f"stopping: more than {MAX_TPU_SESSIONS} TPU sessions; {len(w)} attempts still wait for the TPU")
                break
            try:
                tpu_session(tpu_k, w, verify_needed, a.start_anyway_at, a.tpu_give_up_at)
            except Exception as e:  # noqa: BLE001
                log(f"stopping: TPU step failed: {type(e).__name__}: {str(e)[:200]}; {len(w)} attempts are left unjudged")
                break
            verify_needed = False
            if [i for i in waiting() if i["id"] in {x["id"] for x in w}]:
                log("some TPU attempts came back without a row: stopping rather than looping")
                break
        else:
            break
    report()
    log("overnight done")


if __name__ == "__main__":
    main()
