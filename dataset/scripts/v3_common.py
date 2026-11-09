"""Shared pieces for the dataset v3 build: where v3 lives, the oracle's kernel check run over many kernels with full feedback, and rows.

v3 never edits dataset-v1 (dataset/final/). Everything it writes is under dataset/v3/. Every row carries meta.v3.verification, which says
how far its target was actually checked:
  tpu              the target is a kernel the TPU oracle passed (dataset-v1)
  tpu_correct_slow the TPU oracle found the target correct but below the speed bar
  cpu              the target passed the CPU checks in v3_verify.py only; it has never been on a chip
A `cpu` row has final_verdict "cpu_verified", never "pass": in dataset-v1, "pass" means the TPU oracle's verdict.
"""
from __future__ import annotations

import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATASET, FINAL, ORACLE, Isolated, read_jsonl, sha256  # noqa: E402

V3 = DATASET / "v3"
ORACLE_FILES = ("tpu_oracle.py", "tpu_check.py", "outside_ops.py")
ORACLE_SHA = sha256(b"".join((ORACLE / p).read_bytes() for p in ORACLE_FILES))[:12]

# tpu_oracle prints feedback cut to 120 characters, so read its full rows from the file it writes. The oracle files are not modified.
_STATIC_CHILD = r'''
import json, pathlib, sys
sys.path.insert(0, ".")
import tpu_oracle
tpu_oracle.main(["pack.json", "out.jsonl", "--static-only"])
print("@@ROWS@@")
for line in pathlib.Path("out.jsonl").read_text().splitlines():
    if line.strip() and '"meta": true' not in line:
        print(line)
'''


def static_check(items: list, refs: dict, python: str | None, *, timeout: int | None = None, max_rss_gb: float = 6.0) -> tuple:
    """(rows by id, error or None): tpu_oracle.py --static-only over `items` ({id, task, code}) in one isolated child."""
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    files["pack.json"] = json.dumps({"refs": refs, "items": [{"id": i["id"], "task": i["task"], "code": i["code"]} for i in items]})
    files["child.py"] = _STATIC_CHILD
    r = Isolated(python, timeout=timeout or 120 + 10 * len(items), max_rss_gb=max_rss_gb).run(["child.py"], files=files, extra_env={"ORACLE_COMMIT": "v3-static"})
    if "@@ROWS@@" not in r.stdout:
        return {}, (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][-300:]
    rows = [json.loads(l) for l in r.stdout.split("@@ROWS@@", 1)[1].splitlines() if l.startswith("{")]
    return {d["id"]: d for d in rows}, None


def checked_in_batches(items: list, refs: dict, python: str | None, size: int = 60, log=print) -> dict:
    """Rows by id, one child per batch and one at a time (the Mac has 17 GB). A batch that dies is retried item by item, so one crash costs one item."""
    out: dict = {}
    for k in range(0, len(items), size):
        batch = items[k:k + size]
        rows, err = static_check(batch, {i["task"]: refs[i["task"]] for i in batch}, python)
        if err:
            log(f"  batch at {k} failed ({err[:120]}); retrying its {len(batch)} items alone")
            for i in batch:
                one, e = static_check([i], {i["task"]: refs[i["task"]]}, python)
                out.update(one if not e else {i["id"]: {"id": i["id"], "verdict": "harness_crash", "feedback": e}})
        else:
            out.update(rows)
        log(f"  checked {min(k + size, len(items))} of {len(items)}")
    return out


def train_singles() -> list:
    """dataset-v1's solved single-turn train rows. Val families are left alone so val still measures what it measured for v1 and v2."""
    return [e for e in read_jsonl(FINAL / "train.jsonl") if e["kind"] == "single"]


def derived_row(base: dict, *, example_id: str, turns: list, kind: str, origin: str, verification: str, final_verdict: str | None = None,
                extra: dict | None = None) -> dict:
    """A v3 row in dataset-v1's example shape, keeping the base row's split, family and provenance."""
    return {**{k: base[k] for k in ("problem_id", "family_id", "source", "task_name", "speed_vs_xla", "at_parity", "oracle", "split")},
            "example_id": example_id, "kind": kind, "derived_from": base["example_id"], "turns": turns,
            "final_verdict": final_verdict or base["final_verdict"], "n_repairs": (len(turns) - 2) // 2,
            "meta": {**base["meta"], "failing_verdicts": [t["verdict"] for t in turns if t.get("type") == "oracle_feedback"],
                     "v3": {"origin": origin, "verification": verification, **(extra or {})}}}
