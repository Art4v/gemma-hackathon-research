"""Stage 2e: port KernelBook modules (GitHub nn.Modules, GPUMODE/KernelBook, June 9 Researcher Reciprocity License) to plain-JAX
references with port_kernelbench.py's constructor prompt, schema and torch equivalence check. Recorded as source "kernelbook" with the
original repo, pinned commit and the license checked at that commit (kernelbook/kbsel.py). One retry with the failure text, as for
KernelBench. KernelBook's modules keep their own class name and return get_init_inputs() as [args, kwargs], so each source gets a
KernelBench-format adapter appended (a `Model` subclass and a flat get_init_inputs) before the constructor and the check see it.

  python dataset/scripts/port_kernelbook.py --shards A,B [--limit N] [--workers 6] [--check-workers 2] [--batch kbk]
Resumable: stems already in problems.jsonl or excluded.jsonl are skipped. Run one process per machine (it appends to problems.jsonl).
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, json, pathlib, shutil, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import port_kernelbench as pk  # noqa: E402
from common import DATASET, PROBLEMS, CANDIDATES, read_jsonl, task_name, validate, write_jsonl  # noqa: E402
from constructor import Request, get_constructor, save_raw  # noqa: E402
from constructor.prompts import PORT_SYSTEM, port_user  # noqa: E402
from render_chat import unfence  # noqa: E402

SELECTION = DATASET / "kernelbook" / "selection.jsonl"
ADAPTER = '''

# --- KernelBench-format adapter (added for porting; KernelBook returns get_init_inputs() as [args, kwargs]) ---
_KB_ARGS, _KB_KWARGS = get_init_inputs()


class Model({cls}):
    def __init__(self):
        super().__init__(*_KB_ARGS, **_KB_KWARGS)


def get_init_inputs():
    return []
'''


def kb_source(sel: dict) -> str:
    """KernelBook rows get the adapter; rows already in KernelBench format (needs_adapter false, e.g. GitHub references) don't."""
    if not sel.get("needs_adapter", True):
        return sel["python_code"]
    return sel["python_code"].rstrip() + "\n" + ADAPTER.format(cls=sel["module_name"])


def pid_of(sel: dict) -> str:
    return f"{sel.get('src', 'kernelbook')}/{sel['stem']}"


def record(sel: dict, code: str, check: dict, status: str, reason: str | None, plan: str) -> dict:
    rec = pk.record(0, sel["stem"], code, check, "", status, reason, plan)
    pid, src = pid_of(sel), sel.get("src", "kernelbook")
    ref = {"repo": sel["repo_name"], "commit": sel["sha"], "repo_link": sel["repo_link"], "license": sel["license_checked"],
           "module_name": sel["module_name"], "family_hint": sel["family"]}
    if src == "kernelbook":
        ref.update(dataset="GPUMODE/KernelBook", dataset_license="June 9 Researcher Reciprocity License", uuid=sel["uuid"])
    if sel.get("why_not_p_task"):
        ref["why_not_p_task"] = sel["why_not_p_task"]
    rec.update(problem_id=pid, family_id=pid, task_name=task_name(pid), source=src, source_ref=ref)
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", default="A"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", default="kbk")
    ap.add_argument("--backend", default="claude_cli"); ap.add_argument("--model", default=None); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check-workers", type=int, default=2); ap.add_argument("--max-rss-gb", type=float, default=5.0)
    ap.add_argument("--python", default=None); ap.add_argument("--selection", default=str(SELECTION))
    a = ap.parse_args(argv)
    done = {r["problem_id"] for r in read_jsonl(PROBLEMS / "problems.jsonl")} | {r["problem_id"] for r in read_jsonl(PROBLEMS / "excluded.jsonl")}
    sels = [s for s in read_jsonl(pathlib.Path(a.selection)) if s["shard"] in a.shards.split(",") and s.get("import_ok", True) and pid_of(s) not in done]
    sels = sels[:a.limit] if a.limit else sels
    print(f"{len(sels)} KernelBook modules to port (shards {a.shards})", flush=True)
    # CLAUDE_BIN: a stable copy of the binary; the installed one is rewritten every ~10 min and an exec during the rewrite fails
    cli = {"claude_bin": __import__("os").environ.get("CLAUDE_BIN") or shutil.which("claude") or "/opt/homebrew/bin/claude"} if a.backend == "claude_cli" else {}
    con = get_constructor(a.backend, a.model, workers=a.workers, **cli)
    raw_dir = CANDIDATES / "raw_responses"
    # ponytail: chunks of 6x workers, each written before the next starts, so a crash hours in loses one chunk, not the run
    size = 6 * a.workers
    for k in range(0, len(sels), size):
        port_chunk(sels[k:k + size], con, raw_dir, a)
        print(f"-- {min(k + size, len(sels))} of {len(sels)} done", flush=True)


def port_chunk(sels, con, raw_dir, a):
    by_key = {f"port/kbk/{s['stem']}": s for s in sels}
    kept, excluded = [], []

    def run(reqs, label):
        work, failed = [], []
        for resp in con.generate(reqs):
            s = by_key[resp.key]
            save_raw(raw_dir, resp)
            if not resp.ok:
                excluded.append(record(s, "", {}, "excluded", f"port_failed:constructor:{resp.error}", "")); continue
            work.append((resp, s, pk.strip_module(unfence(resp.output["code"]) or resp.output["code"])))
        with cf.ThreadPoolExecutor(max_workers=a.check_workers) as ex:
            chks = list(ex.map(lambda w: pk.check_port(w[2], w[0].output["adapter"], kb_source(w[1]), a.python, a.max_rss_gb), work))
        for (resp, s, code), chk in zip(work, chks):
            print(f"  {s['stem']}{label}: {'ok' if chk.get('ok') else 'fail'} {chk.get('max_rel_err', chk.get('reason', ''))}", flush=True)
            if chk.get("ok"):
                kept.append(record(s, code, chk, "ok", None, resp.output["plan"]))
            else:
                failed.append((s, code, chk, resp))
        return failed

    failed = run([Request(k, PORT_SYSTEM, port_user(s["stem"], kb_source(s)), pk.PORT_SCHEMA) for k, s in by_key.items()], "")
    if failed:
        retry = [Request(f"port/kbk/{s['stem']}", PORT_SYSTEM, port_user(s["stem"], kb_source(s), retry_note=json.dumps(chk)[:1500]), pk.PORT_SCHEMA)
                 for s, code, chk, _ in failed]
        for s, code, chk, resp in run(retry, " (retry)"):
            excluded.append(record(s, code, chk, "excluded", f"port_failed:{str(chk.get('reason') or chk.get('max_rel_err'))[:200]}", resp.output["plan"]))
    for r in kept:
        r["batch"] = a.batch
        errs = validate("problem", r)
        if errs:
            raise SystemExit(f"{r['problem_id']}: {errs}")
        out = PROBLEMS / f"{r['problem_id']}.py"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r["ref_code"])
    write_jsonl(PROBLEMS / "problems.jsonl", kept, append=True)
    write_jsonl(PROBLEMS / "excluded.jsonl", excluded, append=True)
    print(f"ported {len(kept)}, excluded {len(excluded)} -> {PROBLEMS / 'problems.jsonl'}")


if __name__ == "__main__":
    main()
