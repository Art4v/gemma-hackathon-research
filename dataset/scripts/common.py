"""Shared pieces for the dataset pipeline: paths, hashing, JSONL io, ids, record validation, and the one way model-written or
downloaded code is ever executed locally (a `python -I` subprocess in a temp cwd on the CPU JAX platform). See PLAN.md and PROCESS.md.

Pure: imports nothing from the oracle and no JAX, so every stage's bookkeeping runs on any Python.
"""
from __future__ import annotations

import datetime as _dt, hashlib, json, os, pathlib, re, subprocess, sys, tempfile

PIPELINE_VERSION = "pilot-1"

HERE = pathlib.Path(__file__).resolve().parent          # dataset/scripts
DATASET = HERE.parent                                     # dataset/
REPO = DATASET.parent                                     # repo root
ORACLE = REPO / "oracle"
BENCH = REPO / "benchmarks" / "jaxbench"
RAW = DATASET / "raw"
PROBLEMS = DATASET / "problems"
CANDIDATES = DATASET / "candidates"
PRESCREEN = DATASET / "prescreen"
SUBMISSIONS = DATASET / "submissions"
VERDICTS = DATASET / "verdicts"
TRAJECTORIES = DATASET / "trajectories"
FINAL = DATASET / "final"
VENV_PY = DATASET / ".venv" / "bin" / "python"

SPEED_BAR = 0.7          # the oracle's policy bar (oracle/tpu_oracle.py SPEED_BAR); every row is checked against it
PARITY = 0.97            # "at parity or faster": the share reported alongside the pass rate (oracle/DESCRIPTION.md)
N_SEEDS = 3
HBM_BUDGET_BYTES = 28e9  # oracle/tpu_check.py; a task runs only when 3 x input bytes fits
TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SOURCES = ("kernelbench", "authored", "scraped")
VERDICTS_ALL = ("pass", "correct_slow", "wrong", "not_a_kernel", "compile", "vmem", "mosaic", "timeout", "runtime", "harness_oom",
                "too_large_for_chip")
KERNEL_FAULT = ("wrong", "not_a_kernel", "compile", "vmem", "mosaic", "timeout", "runtime", "correct_slow")  # consume an attempt
HARNESS_FAULT = ("harness_oom", "too_large_for_chip")                                                       # do not


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def read_jsonl(path) -> list:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(path, rows, append: bool = False) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def write_json(path, obj) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def task_name(problem_id: str) -> str:
    """The oracle `task` field for a problem id: '/' -> '__', anything else outside [A-Za-z0-9._-] -> '-'."""
    t = re.sub(r"[^A-Za-z0-9._-]", "-", problem_id.replace("/", "__"))
    if not TASK_NAME.fullmatch(t):
        raise ValueError(f"cannot make a task name from {problem_id!r}")
    return t


def candidate_id(task: str, attempt: int, sample: int) -> str:
    return f"{task}.a{attempt}.s{sample}"


# ---- record validation: required keys and their types; extra keys are allowed ----

_PROBLEM = {"problem_id": str, "family_id": str, "task_name": str, "source": str, "source_ref": dict, "ref_code": str, "ref_sha256": str,
            "signature": (dict, type(None)), "input_bytes": (int, type(None)), "fits_hbm": (bool, type(None)), "status": str,
            "exclude_reason": (str, type(None)), "pipeline_version": str, "created_at": str}
_CANDIDATE = {"candidate_id": str, "problem_id": str, "task_name": str, "attempt": int, "sample": int, "parent_candidate_id": (str, type(None)),
              "feedback_in": (str, type(None)), "feedback_in_verdict": (dict, type(None)), "prescreen_hint": (str, type(None)),
              "constructor": dict, "plan": str, "code": str, "code_sha256": str, "usage": dict, "created_at": str}
_PRESCREEN = {"candidate_id": str, "outcome": str, "stage": (str, type(None)), "reason": str, "oracle_sha_used": (str, type(None)),
              "seconds": (int, float)}
_VERDICT = {"candidate_id": str, "job_id": str, "oracle_commit": str, "oracle_sha": str, "verdict": str, "feedback": str,
            "speed_vs_xla": (int, float, type(None)), "accepted": bool, "row": dict, "collected_at": str}
_EXAMPLE = {"example_id": str, "kind": str, "derived_from": (str, type(None)), "problem_id": str, "family_id": str, "source": str,
            "task_name": str, "turns": list, "final_verdict": str, "speed_vs_xla": (int, float, type(None)), "at_parity": bool,
            "n_repairs": int, "oracle": dict, "meta": dict, "split": str}
SCHEMAS = {"problem": _PROBLEM, "candidate": _CANDIDATE, "prescreen": _PRESCREEN, "verdict": _VERDICT, "example": _EXAMPLE}


def validate(kind: str, rec: dict) -> list:
    """Problems with `rec` as a `kind` record (empty list = valid)."""
    errs, schema = [], SCHEMAS[kind]
    for k, t in schema.items():
        if k not in rec:
            errs.append(f"missing {k}")
        elif not isinstance(rec[k], t):
            errs.append(f"{k}: expected {t}, got {type(rec[k]).__name__}")
    if kind == "problem":
        if rec.get("source") not in SOURCES:
            errs.append(f"source not in {SOURCES}")
        if rec.get("status") not in ("ok", "excluded"):
            errs.append("status must be ok or excluded")
        if rec.get("ref_code") is not None and rec.get("ref_sha256") != sha256(rec["ref_code"]):
            errs.append("ref_sha256 does not match ref_code")
        if rec.get("task_name") and not TASK_NAME.fullmatch(rec["task_name"]):
            errs.append("task_name has characters the oracle rejects")
    if kind == "candidate":
        if rec.get("attempt") not in (1, 2, 3):
            errs.append("attempt must be 1, 2 or 3")
        if rec.get("code") is not None and rec.get("code_sha256") != sha256(rec["code"]):
            errs.append("code_sha256 does not match code")
        if (rec.get("attempt") == 1) != (rec.get("parent_candidate_id") is None):
            errs.append("attempt 1 has no parent; later attempts need one")
        if (rec.get("attempt") == 1) != (rec.get("feedback_in") is None):
            errs.append("attempt 1 has no feedback_in; later attempts need it")
    if kind == "prescreen" and rec.get("outcome") not in ("pass", "reject", "abstain"):
        errs.append("outcome must be pass, reject or abstain")
    if kind == "verdict":
        if rec.get("verdict") not in VERDICTS_ALL + ("static_ok",):
            errs.append(f"unknown verdict {rec.get('verdict')!r}")
        if rec.get("accepted") != (rec.get("verdict") == "pass"):
            errs.append("accepted must be (verdict == pass)")
    if kind == "example":
        if rec.get("kind") not in ("single", "repair"):
            errs.append("kind must be single or repair")
        if rec.get("split") not in ("train", "val"):
            errs.append("split must be train or val")
        if rec.get("final_verdict") != "pass":
            errs.append("final_verdict must be pass")
        turns = rec.get("turns") or []
        if len(turns) not in (2, 4, 6):
            errs.append("turns must be 2 (single), 4 or 6 (repair)")
        if turns and (turns[0].get("type") != "problem" or turns[-1].get("type") != "kernel"):
            errs.append("turns must start with the problem and end with a kernel")
    return errs


# ---- the only local execution of untrusted code ----

class Isolated:
    """A subprocess running `python -I` in a fresh temp directory with a minimal environment and JAX_PLATFORMS=cpu. Used for the
    oracle's --static-only check and for shape-only tracing of references. Not a security sandbox: it limits accidents (imports from
    the cwd or site customisations, writes into the repo), not a hostile kernel. Nothing here runs on the TPU."""

    def __init__(self, python: str | None = None, timeout: int = 300, max_rss_gb: float = 5.0):
        self.python = python or (str(VENV_PY) if VENV_PY.exists() else sys.executable)
        self.timeout, self.max_rss_gb = timeout, max_rss_gb

    def run(self, argv: list, files: dict | None = None, extra_env: dict | None = None) -> subprocess.CompletedProcess:
        """Run `argv` (paths relative to the temp cwd) with `files` ({relative path: text}) written there first. A watchdog polls the
        child's RSS once a second and kills it over `max_rss_gb` (rc 137, '[memory cap]' in stderr): the machine has 17 GB and the
        Nov 8 crash came from two checks holding multi-GB tensors at once."""
        import time
        with tempfile.TemporaryDirectory(prefix="dataset-iso-") as d:
            for rel, text in (files or {}).items():
                p = pathlib.Path(d) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            env = {"PATH": os.environ.get("PATH", ""), "HOME": d, "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1",
                   "XLA_FLAGS": "--xla_force_host_platform_device_count=1", "OMP_NUM_THREADS": "4", **(extra_env or {})}
            out_f, err_f = open(pathlib.Path(d) / "_stdout", "w+"), open(pathlib.Path(d) / "_stderr", "w+")
            proc = subprocess.Popen([self.python, "-I", *argv], cwd=d, env=env, stdout=out_f, stderr=err_f, text=True)
            t0, rc, note, peak = time.time(), None, "", 0
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                if time.time() - t0 > self.timeout:
                    proc.kill(); rc, note = 124, f"\n[timeout after {self.timeout}s]"; proc.wait(); break
                rss = _rss_gb(proc.pid)
                peak = max(peak, rss)
                if rss > self.max_rss_gb:
                    proc.kill(); rc, note = 137, f"\n[memory cap: {rss:.1f} GB > {self.max_rss_gb} GB]"; proc.wait(); break
                time.sleep(0.5)
            out_f.seek(0); err_f.seek(0)
            res = subprocess.CompletedProcess(proc.args, rc, out_f.read(), err_f.read() + note)
            out_f.close(); err_f.close()
            res.peak_rss_gb = round(peak, 2)  # type: ignore[attr-defined]
            return res


def _rss_gb(pid: int) -> float:
    """Resident set size of `pid` and its children, in GB (ps-based; works on macOS and Linux)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=,ppid=,pid=", "-ax"], capture_output=True, text=True).stdout
    except Exception:
        return 0.0
    rows = [l.split() for l in out.splitlines() if l.strip()]
    kids, total = {pid}, 0
    changed = True
    while changed:
        changed = False
        for r in rows:
            if len(r) == 3 and r[1].isdigit() and int(r[1]) in kids and int(r[2]) not in kids:
                kids.add(int(r[2])); changed = True
    for r in rows:
        if len(r) == 3 and r[2].isdigit() and int(r[2]) in kids:
            total += int(r[0])
    return total / 1e6  # ps reports KB


def kb_stem(code: str) -> str | None:
    """CONFIG['name'] of a baseline module, by regex (no execution)."""
    m = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", code)
    return m.group(1) if m else None


def exam_files() -> list:
    """The 73 exam baseline files: (task dir name, level, path)."""
    out = []
    for level in ("level1", "level2"):
        for p in sorted((BENCH / level).glob("*/baseline.py")):
            out.append((p.parent.name, level, p))
    return out


def exam_kb_stems() -> dict:
    """KernelBench stem -> exam task, for every k task (the 33 names the contamination filter excludes outright)."""
    out = {}
    for task, level, p in exam_files():
        if level == "level1" and re.match(r"\d+k_", task):
            stem = kb_stem(p.read_text())
            if stem:
                out[stem] = task
    return out
