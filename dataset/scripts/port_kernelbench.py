"""Stage 2a: port KernelBench problems (PyTorch nn.Module) to JAXBench-style plain-JAX baseline modules, check each port against
PyTorch at KernelBench's own shapes, and record the survivors as problems.

  python dataset/scripts/port_kernelbench.py --pick easy --limit 20 [--batch pilot] [--backend claude_cli] [--model MODEL] [--workers 3]
  python dataset/scripts/port_kernelbench.py --stems 63_Gemm_ReLU_Divide,... --level 2

The constructor returns the module plus a `from_torch(model, inputs)` adapter (torch Model and its get_inputs() -> the JAX workload's
arguments). The equivalence check runs both sides in a `python -I` subprocess on CPU: BatchNorm in batch-statistics mode and Dropout
off, like the JAXBench convention; max_rel_err <= 1e-3 passes. One retry with the failure text, then `excluded: port_failed`.
Exam KernelBench stems (common.exam_kb_stems) and a hand list of KernelBench L1 problems that are exam p-tasks in all but name are
skipped before any constructor call. The op-chain and near-dup filters run afterwards (filter_contamination.py, filter_neardup.py).
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (CANDIDATES, HBM_BUDGET_BYTES, PIPELINE_VERSION, PROBLEMS, RAW, Isolated, exam_kb_stems, now, read_json, read_jsonl,  # noqa: E402
                    sha256, task_name, validate, write_jsonl)
from constructor import Request, get_constructor, save_raw  # noqa: E402
from constructor.prompts import PORT_SYSTEM, PROMPT_VERSION, port_user  # noqa: E402
from render_chat import unfence  # noqa: E402

KB = RAW / "kernelbench" / "KernelBench"
# KernelBench level-1 problems that are exam p-tasks in all but name (8p GEMM, 12p RMSNorm, 13p Cross Entropy, 1p/2p attention).
# The op-chain filter would catch them later; skipping here saves constructor calls.
P_TASK_TWINS = {"1_Square_matrix_multiplication_", "2_Standard_matrix_multiplication_", "6_Matmul_with_large_K_dimension_",
                "7_Matmul_with_small_K_dimension_", "8_Matmul_with_irregular_shapes_", "16_Matmul_with_transposed_A",
                "17_Matmul_with_transposed_B", "18_Matmul_with_transposed_both", "36_RMSNorm_", "95_CrossEntropyLoss",
                "97_ScaledDotProductAttention", "9_Tall_skinny_matrix_multiplication_", "13_Matmul_for_symmetric_matrices"}
EASY_L1 = re.compile(r"(relu|sigmoid|tanh|softmax|gelu|swish|selu|elu|softplus|softsign|hardtanh|hardsigmoid|norm|reduction|loss|mish|"
                     r"cumsum|cumprod|triangular|diag|argmax|argmin)", re.I)
HARD = re.compile(r"(conv|pool|bmm|3d|transpose_conv|convtranspose|scan|lstm|rnn|gru|attention)", re.I)
CONV = re.compile(r"(conv)", re.I)

PORT_SCHEMA = {"type": "object", "properties": {"plan": {"type": "string"}, "code": {"type": "string"},
               "adapter": {"type": "string", "description": "Python source of `def from_torch(model, inputs): ...` returning the tuple of "
                           "jax arrays `workload` takes, built from the torch Model's parameters and get_inputs() (numpy conversion, any "
                           "transposes the port needs). May import numpy, jax, jax.numpy, torch."}},
               "required": ["plan", "code", "adapter"], "additionalProperties": False}

CHECK = r'''
import json, sys, types, importlib.util, numpy as np, pathlib
import torch, jax, jax.numpy as jnp
torch.manual_seed(0); np.random.seed(0)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m
# KernelBench's "small" inputs can be gigabytes (24_LogSoftmax: 4096 x 393216 f32 = 6.4 GB) and the machine has 17 GB, so the batch
# (leading) dimension of every tensor get_inputs() builds is capped at 64. Model parameters are built from module constants, not
# from the batch, so the semantics check is unchanged; both sides see the same shrunken inputs.
_CAP, _MAX_ELEMS = 64, 16_000_000  # batch <= 64 and <= 16M elements (64 MB f32) per tensor; the batch is the only dim shrunk
def _shrink(fn):
    def wrapped(*size, **kw):
        orig = (size, dict(kw))
        try:
            if "size" in kw and not size:
                size, kw = tuple(kw.pop("size")), kw
            if size and isinstance(size[0], (tuple, list, torch.Size)):
                size = tuple(size[0]) + tuple(size[1:])
            if len(size) >= 2 and all(isinstance(s, int) for s in size):
                rest = 1
                for s in size[1:]:
                    rest *= s
                b = min(size[0], _CAP, max(2, _MAX_ELEMS // max(rest, 1)))
                size = (b,) + tuple(size[1:])
            return fn(*size, **kw)
        except TypeError:
            return fn(*orig[0], **orig[1])  # an argument shape this wrapper does not understand: build it as written
    return wrapped
def _patch_factories():
    for name in ("rand", "randn", "ones", "zeros", "empty"):
        setattr(torch, name, _shrink(getattr(torch, name)))
kb = load("kb.py", "kb"); port = load("port.py", "port"); ad = load("adapter.py", "adapter")
model = kb.Model(*kb.get_init_inputs())
_patch_factories()  # only get_inputs() below sees the shrunken factories; the model's parameters are already built
model.train()  # BatchNorm: batch statistics (JAXBench convention)
for m in model.modules():
    if isinstance(m, torch.nn.Dropout) or m.__class__.__name__.startswith("Dropout"):
        m.eval()
inputs = kb.get_inputs()  # KernelBench "small" shapes can be gigabytes (24_LogSoftmax: 4096 x 393216 f32), so no full-size temporaries
with torch.no_grad():
    want = model(*inputs)
want = [w.detach().cpu().numpy() for w in (want if isinstance(want, (tuple, list)) else [want])]
args = ad.from_torch(model, inputs)
del inputs
got = jax.jit(port.workload)(*args)
got = [np.asarray(g) for g in (got if isinstance(got, (tuple, list)) else [got])]
del args
if len(want) != len(got) or any(w.shape != g.shape for w, g in zip(want, got)):
    print(json.dumps({"ok": False, "reason": f"shape mismatch: torch {[w.shape for w in want]} vs jax {[g.shape for g in got]}"})); sys.exit()
def chunked_err(w, g, rows=64):
    w, g = np.atleast_1d(w), np.atleast_1d(g)
    if w.ndim == 0 or w.shape[0] < 2:
        return float(np.abs(w.astype(np.float32) - g.astype(np.float32)).max()), float(np.abs(w.astype(np.float32)).max())
    e = m = 0.0
    for i in range(0, w.shape[0], rows):
        a, b = w[i:i + rows].astype(np.float32), g[i:i + rows].astype(np.float32)
        e = max(e, float(np.abs(a - b).max())); m = max(m, float(np.abs(a).max()))
    return e, m
errs = [chunked_err(w, g) for w, g in zip(want, got)]
err = max(e / max(m, 1e-30) for e, m in errs)
abstract = jax.eval_shape(port.create_inputs)
nbytes = int(sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(abstract)))
print(json.dumps({"ok": err <= 1e-3, "max_rel_err": err, "shapes": [list(w.shape) for w in want], "input_bytes": nbytes,
                  "full_shapes": [list(x.shape) for x in jax.tree_util.tree_leaves(abstract)]}))
'''


def kb_files(levels: list) -> list:
    out = []
    for lv in levels:
        for p in sorted((KB / f"level{lv}").glob("*.py")):
            out.append((lv, p.stem, p))
    return out


def pick(files: list, mode: str, exam: dict, limit: int) -> list:
    chosen = []
    for lv, stem, p in files:
        if stem in exam or stem in P_TASK_TWINS:
            continue
        if mode == "easy" and (HARD.search(stem) or (lv == 1 and not EASY_L1.search(stem))):
            continue
        if mode == "conv" and not CONV.search(stem):
            continue
        chosen.append((lv, stem, p))
    return chosen[:limit] if limit else chosen


def check_port(code: str, adapter: str, kb_source: str, python: str | None, max_rss_gb: float = 5.0) -> dict:
    r = Isolated(python, timeout=600, max_rss_gb=max_rss_gb).run(["check.py"], files={"check.py": CHECK, "kb.py": kb_source, "port.py": code, "adapter.py": adapter})
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    peak = getattr(r, "peak_rss_gb", None)
    if line:
        return {**json.loads(line), "peak_rss_gb": peak}
    return {"ok": False, "reason": (r.stderr.strip().splitlines() or ["no output"])[-1][-400:], "peak_rss_gb": peak}


def saved_response(raw_dir: pathlib.Path, key: str, backend: str | None = None):
    """The newest saved constructor response for `key` (from a run that was interrupted before it wrote its records), or None.
    With `backend`, only that backend's responses count: keys are problem and candidate ids, which a rerun on another backend
    shares, so without it a Gemini run would silently reuse cached Claude answers."""
    best = None
    for f in raw_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if backend and d.get("backend", "claude_cli") != backend:
            continue
        if d.get("key") == key and d.get("raw", "").startswith("{"):
            raw = json.loads(d["raw"])
            if isinstance(raw.get("structured_output"), dict) and (best is None or f.stat().st_mtime > best[0]):
                best = (f.stat().st_mtime, raw, d)
    if best is None:
        return None
    from constructor import Response
    raw, d = best[1], best[2]
    u = raw.get("usage") or {}
    return Response(key, True, raw["structured_output"], d.get("model", "default"), d.get("backend", "claude_cli"),
                    {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "wall_s": None, "cost_usd": raw.get("total_cost_usd")},
                    d["raw"])


def strip_module(code: str) -> str:
    """Drop a benchmark() function and a __main__ block if the constructor added them."""
    code = re.sub(r"\n\s*def benchmark\(.*?(?=\n\S|\Z)", "\n", code, flags=re.S)
    code = re.sub(r"\nif __name__ == .__main__.:.*\Z", "\n", code, flags=re.S)
    return code.strip("\n") + "\n"


def record(lv: int, stem: str, code: str, check: dict, commit: str, status: str, reason: str | None, plan: str) -> dict:
    pid = f"kb/l{lv}/{stem}"
    return {"problem_id": pid, "family_id": pid, "task_name": task_name(pid), "source": "kernelbench",
            "source_ref": {"repo": "ScalingIntelligence/KernelBench", "commit": commit, "path": f"KernelBench/level{lv}/{stem}.py", "license": "MIT",
                           "kb_level": lv, "kb_stem": stem},
            "ref_code": code, "ref_sha256": sha256(code), "signature": None, "input_bytes": check.get("input_bytes"),
            "fits_hbm": (3 * check["input_bytes"] <= HBM_BUDGET_BYTES) if check.get("input_bytes") else None,
            "port_check": {"method": "torch_cpu", "shapes": check.get("shapes"), "max_rel_err": check.get("max_rel_err"), "ok": bool(check.get("ok")),
                           **({"reason": check["reason"]} if check.get("reason") else {})},
            "plan": plan, "status": status, "exclude_reason": reason, "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION,
            "created_at": now()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", default="1,2"); ap.add_argument("--pick", choices=["easy", "conv", "all"], default="easy")
    ap.add_argument("--stems", default=""); ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch", default="pilot"); ap.add_argument("--backend", default=None, help="claude_cli | anthropic_batch | gemini (default: $CONSTRUCTOR_BACKEND, else claude_cli)"); ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--check-workers", type=int, default=1, help="parallel torch/JAX checks; keep at 1 on a 17 GB machine")
    ap.add_argument("--max-rss-gb", type=float, default=5.0, help="kill a check above this resident size")
    ap.add_argument("--reuse-raw", action="store_true", help="use saved constructor responses from candidates/raw_responses when present")
    ap.add_argument("--python", default=None); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exam = exam_kb_stems()
    files = kb_files([int(x) for x in a.level.split(",")])
    if a.stems:
        want = set(a.stems.split(","))
        chosen = [f for f in files if f[1] in want]
    else:
        chosen = pick(files, a.pick, exam, a.limit)
    done = {r["problem_id"] for r in read_jsonl(PROBLEMS / "problems.jsonl")} | {r["problem_id"] for r in read_jsonl(PROBLEMS / "excluded.jsonl")}
    chosen = [f for f in chosen if f"kb/l{f[0]}/{f[1]}" not in done]
    print(f"{len(chosen)} KernelBench problems to port ({len(exam)} exam stems and {len(P_TASK_TWINS)} p-task twins skipped)")
    if a.dry_run:
        for lv, stem, _ in chosen:
            print(f"  l{lv}/{stem}")
        return
    commit = read_json(RAW / "SOURCES.json")["kernelbench"]["commit"]
    con = get_constructor(a.backend, a.model, workers=a.workers)
    raw_dir = CANDIDATES / "raw_responses"
    reqs = [Request(f"port/{lv}/{stem}", PORT_SYSTEM, port_user(stem, p.read_text()), PORT_SCHEMA) for lv, stem, p in chosen]
    by_key = {f"port/{lv}/{stem}": (lv, stem, p) for lv, stem, p in chosen}
    kept, excluded = [], []

    def checked(resps: list, label: str) -> list:
        """Run the equivalence checks in parallel (each is its own subprocess, often a minute of CPU); return (resp, code, chk) rows."""
        work = []
        for resp in resps:
            lv, stem, p = by_key[resp.key]
            save_raw(raw_dir, resp)
            if not resp.ok:
                excluded.append(record(lv, stem, "", {}, commit, "excluded", f"port_failed:constructor:{resp.error}", "")); continue
            work.append((resp, strip_module(unfence(resp.output["code"]) or resp.output["code"]), p))
        with cf.ThreadPoolExecutor(max_workers=a.check_workers) as ex:
            chks = list(ex.map(lambda w: check_port(w[1], w[0].output["adapter"], w[2].read_text(), a.python, a.max_rss_gb), work))
        out = []
        for (resp, code, p), chk in zip(work, chks):
            lv, stem, _ = by_key[resp.key]
            print(f"  l{lv}/{stem}{label}: {'ok' if chk.get('ok') else 'fail'} {chk.get('max_rel_err', chk.get('reason', ''))}", flush=True)
            out.append((resp, code, chk))
        return out

    if a.reuse_raw:
        have = {r.key: r for r in (saved_response(raw_dir, r.key, con.name) for r in reqs) if r is not None}
        fresh = [r for r in reqs if r.key not in have]
        print(f"reusing {len(have)} saved constructor responses, asking for {len(fresh)}", flush=True)
        responses = list(have.values()) + (con.generate(fresh) if fresh else [])
    else:
        responses = con.generate(reqs)
    retry = []
    for resp, code, chk in checked(responses, ""):
        lv, stem, p = by_key[resp.key]
        if chk.get("ok"):
            kept.append(record(lv, stem, code, chk, commit, "ok", None, resp.output["plan"]))
        else:
            retry.append((lv, stem, p, code, chk))
    if retry:
        reqs = [Request(f"port/{lv}/{stem}", PORT_SYSTEM, port_user(stem, p.read_text(), retry_note=json.dumps(chk)[:1500]), PORT_SCHEMA)
                for lv, stem, p, code, chk in retry]
        by_key = {f"port/{lv}/{stem}": (lv, stem, p) for lv, stem, p, _, _ in retry}
        for resp, code, chk in checked(con.generate(reqs), " (retry)"):
            lv, stem, p = by_key[resp.key]
            if chk.get("ok"):
                kept.append(record(lv, stem, code, chk, commit, "ok", None, resp.output["plan"]))
            else:
                excluded.append(record(lv, stem, code, chk, commit, "excluded", f"port_failed:{str(chk.get('reason') or chk.get('max_rel_err'))[:200]}", resp.output["plan"]))
    for r in kept:
        errs = validate("problem", r)
        if errs:
            raise SystemExit(f"{r['problem_id']}: {errs}")
        out = PROBLEMS / "kb" / f"l{r['source_ref']['kb_level']}" / f"{r['source_ref']['kb_stem']}.py"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r["ref_code"])
    write_jsonl(PROBLEMS / "problems.jsonl", kept, append=True)
    write_jsonl(PROBLEMS / "excluded.jsonl", excluded, append=True)
    print(f"ported {len(kept)}, excluded {len(excluded)} -> {PROBLEMS / 'problems.jsonl'}")


if __name__ == "__main__":
    main()
