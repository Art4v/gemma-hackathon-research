"""Stage 3, layer 3 of the contamination filter (PROCESS.md section 3): near-duplicate text against the 73 exam baseline files, and
within-set duplicates.

  python dataset/scripts/filter_neardup.py [--problems dataset/problems/problems.jsonl] [--dry-run]

Code is normalised before comparing: comments and docstrings dropped, numbers -> N, strings -> S, identifiers kept and lower-cased,
the `benchmark()` boilerplate and the `if __name__` block removed (every exam file carries them). Then token 5-gram Jaccard and
difflib ratio (autojunk off). Excludes at Jaccard >= 0.5 or ratio >= 0.8 (max over identifier-kept and identifier-abstracted token streams) (`contamination:neardup:<task>:j=..,r=..`), for the
reference and, when present, the wrapped human kernel (`source_ref.kernel_code` or a sibling `<path>.kernel.py`). Then two `ok`
problems with the same (op_chain, input_shapes, input_dtypes) keep the first by problem_id (`duplicate:<kept>`). Writes the top-3
nearest exam files per kept problem into signatures.jsonl.
"""
from __future__ import annotations

import argparse, ast, difflib, io, pathlib, sys, tokenize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import PROBLEMS, exam_files, now, read_jsonl, write_jsonl  # noqa: E402

JACCARD_BAR, RATIO_BAR, NGRAM = 0.5, 0.8, 5
DUP_JACCARD = 0.9  # within-set duplicate: same signature AND near-identical abstracted text
# Two token streams are scored: identifiers kept, and identifiers abstracted to ID (alpha-renaming invariant). A renamed copy of an
# exam file keeps ~0.8 5-gram Jaccard on the abstracted stream while scoring ~0.2 with identifiers kept; an unrelated module that only
# shares the JAXBench skeleton scores ~0.3 abstracted and ~0.7 difflib ratio (pilot, Nov 8: MoE router vs 12p_RMSNorm). So: exclude at
# max Jaccard >= 0.5, or max ratio >= 0.8; the ratio never excludes at the skeleton level alone.
import builtins, keyword
_KEEP = set(keyword.kwlist) | set(dir(builtins)) | {
    "jax", "jnp", "lax", "random", "numpy", "np", "key", "split", "normal", "uniform", "randint", "matmul", "dot", "dot_general", "sum",
    "mean", "max", "min", "exp", "log", "sqrt", "rsqrt", "astype", "reshape", "transpose", "float32", "bfloat16", "int32", "shape", "config",
    "create_inputs", "workload", "dtype", "axis", "keepdims", "nn", "relu", "gelu", "softmax", "sigmoid", "tanh", "where", "top_k", "square",
    "preferred_element_type", "einsum", "swapaxes", "var", "std", "abs", "clip", "maximum", "minimum", "concatenate", "stack", "zeros", "ones",
    "arange", "asarray", "array", "pad", "take", "argmax", "argmin", "cumsum", "logsumexp", "softplus", "silu", "swish", "mish", "erf", "power",
    "pallas_call", "pl", "pltpu", "blockspec", "grid", "in_specs", "out_specs", "out_shape", "shapedtypestruct", "vmem", "program_id"}


def alpha(tokens: list) -> list:
    """Identifiers outside the keyword/builtin/JAX-API set become ID, so renaming variables does not hide a copy."""
    return ["ID" if (t.isidentifier() and t not in _KEEP) else t for t in tokens]


def strip_boilerplate(code: str) -> str:
    """Remove `def benchmark(...)`, `def get_flops`, and the `if __name__ == '__main__'` block. Falls back to the input on a syntax error."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines(keepends=True)
    drop = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("benchmark", "get_flops"):
            drop.append(node)
        elif isinstance(node, ast.If) and "__name__" in (ast.get_source_segment(code, node.test) or ""):
            drop.append(node)
    for node in sorted(drop, key=lambda n: n.lineno, reverse=True):
        start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
        del lines[start:node.end_lineno]
    return "".join(lines)


def normalise(code: str) -> list:
    """Token stream: identifiers/keywords lower-cased, numbers N, strings S, operators kept; comments, docstrings, NL/INDENT dropped."""
    code = strip_boilerplate(code)
    out, prev_sig = [], None
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = []
        try:
            for t in tokenize.generate_tokens(io.StringIO(code).readline):
                toks.append(t)
        except Exception:
            pass
    for t in toks:
        tt = t.type
        if tt in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
            if tt == tokenize.NEWLINE:
                prev_sig = "NEWLINE"
            continue
        if tt == tokenize.STRING:
            # a string that is the first token of a logical line is a docstring (module, class or function): drop it
            if prev_sig in (None, "NEWLINE", ":"):
                prev_sig = "S"
                continue
            out.append("S")
        elif tt == tokenize.NUMBER:
            out.append("N")
        elif tt == tokenize.NAME:
            out.append(t.string.lower())
        else:
            out.append(t.string)
        prev_sig = t.string if tt == tokenize.OP else out[-1]
    return out


def ngrams(tokens: list, n: int = NGRAM) -> set:
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def similarity(a: list, b: list) -> tuple:
    """(5-gram Jaccard, difflib ratio) of two normalised token streams."""
    ga, gb = ngrams(a), ngrams(b)
    j = len(ga & gb) / len(ga | gb) if (ga or gb) else 0.0
    r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if (a and b) else 0.0
    return j, r


def exam_corpus() -> list:
    """[(task, level, normalised tokens)] for all 73 exam baseline files."""
    return [(task, level, normalise(p.read_text())) for task, level, p in exam_files()]


def nearest(tokens: list, corpus: list, k: int = 3) -> list:
    """Top-k exam files by similarity; jaccard/ratio are the max over the two token streams, with both kept for the audit trail."""
    rows, ta = [], alpha(tokens)
    for task, level, toks in corpus:
        j1, r1 = similarity(tokens, toks)
        j2, r2 = similarity(ta, alpha(toks))
        rows.append({"task": task, "level": level, "jaccard": round(max(j1, j2), 3), "ratio": round(max(r1, r2), 3),
                     "jaccard_ids": round(j1, 3), "jaccard_alpha": round(j2, 3), "ratio_ids": round(r1, 3), "ratio_alpha": round(r2, 3)})
    rows.sort(key=lambda x: (x["jaccard"], x["ratio"]), reverse=True)
    return rows[:k]


def hits(near: list) -> dict | None:
    for n in near:
        if n["jaccard"] >= JACCARD_BAR or n["ratio"] >= RATIO_BAR:
            return n
    return None


def kernel_text(rec: dict, problems_path: pathlib.Path) -> str | None:
    kc = rec.get("source_ref", {}).get("kernel_code")
    if kc:
        return kc
    rel = rec.get("source_ref", {}).get("path")
    if rel:
        for base in (problems_path.parent, problems_path.parent.parent):
            p = base / rel
            k = p.with_suffix(".kernel.py") if p.suffix == ".py" else p.parent / (p.name + ".kernel.py")
            if k.exists():
                return k.read_text()
    return None


def run(problems: list, corpus: list, problems_path: pathlib.Path, log=print) -> tuple:
    """(problems, excluded rows, signature updates, max similarity seen)."""
    excluded, sig_updates, worst = [], {}, (0.0, 0.0, None)
    for rec in problems:
        if rec.get("status") != "ok":
            continue
        pid = rec["problem_id"]
        texts = [("ref", rec["ref_code"])]
        kt = kernel_text(rec, problems_path)
        if kt:
            texts.append(("kernel", kt))
        near_all = {}
        hit = None
        for label, text in texts:
            near = nearest(normalise(text), corpus)
            near_all[label] = near
            if near and (near[0]["jaccard"], near[0]["ratio"]) > worst[:2]:
                worst = (near[0]["jaccard"], near[0]["ratio"], f"{pid}/{label}->{near[0]['task']}")
            h = hits(near)
            if h and not hit:
                hit = (label, h)
        if hit:
            label, h = hit
            rec["status"] = "excluded"
            rec["exclude_reason"] = f"contamination:neardup:{h['task']}:j={h['jaccard']},r={h['ratio']}" + ("" if label == "ref" else ":kernel")
            excluded.append({"problem_id": pid, "reason": rec["exclude_reason"], "at": now()})
            log(f"EXCLUDE {pid}: {rec['exclude_reason']}")
            continue
        sig_updates[pid] = {"nearest_exam_text": near_all}
    # within-set duplicates
    seen, seen_code = {}, {}
    for rec in sorted((p for p in problems if p.get("status") == "ok"), key=lambda p: p["problem_id"]):
        sig = rec.get("signature") or {}
        if "op_chain" not in sig:
            continue
        key = (tuple(sig["op_chain"]), str(sig.get("input_shapes")), str(sig.get("input_dtypes")))
        # The signature drops layout ops and axis parameters, so GroupNorm and BatchNorm at one shape share a key (pilot, Nov 8).
        # A duplicate also needs near-identical text on the identifier-abstracted stream.
        if key in seen and similarity(alpha(normalise(strip_boilerplate(rec["ref_code"]))), alpha(normalise(strip_boilerplate(seen_code[key]))))[0] >= DUP_JACCARD:
            rec["status"], rec["exclude_reason"] = "excluded", f"duplicate:{seen[key]}"
            excluded.append({"problem_id": rec["problem_id"], "reason": rec["exclude_reason"], "at": now()})
            sig_updates.pop(rec["problem_id"], None)
            log(f"EXCLUDE {rec['problem_id']}: {rec['exclude_reason']}")
        else:
            seen[key] = rec["problem_id"]
            seen_code[key] = rec['ref_code']
    return problems, excluded, sig_updates, worst


def merge_signatures(path: pathlib.Path, updates: dict) -> None:
    rows = {r["problem_id"]: r for r in read_jsonl(path)}
    for pid, fields in updates.items():
        rows.setdefault(pid, {"problem_id": pid}).update(fields)
    write_jsonl(path, [rows[k] for k in sorted(rows)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default=str(PROBLEMS / "problems.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    path = pathlib.Path(a.problems)
    problems = read_jsonl(path)
    if not problems:
        sys.exit(f"no problems in {path}")
    problems, excluded, sig_updates, worst = run(problems, exam_corpus(), path)
    kept = sum(1 for p in problems if p["status"] == "ok")
    print(f"kept {kept} of {len(problems)}; excluded {len(excluded)} (neardup {sum('neardup' in e['reason'] for e in excluded)}, "
          f"duplicate {sum(e['reason'].startswith('duplicate') for e in excluded)}); "
          f"max similarity seen: jaccard {worst[0]:.2f} ratio {worst[1]:.2f} ({worst[2]})")
    if a.dry_run:
        return
    write_jsonl(path, problems)
    write_jsonl(path.parent / "excluded.jsonl", excluded, append=True)
    merge_signatures(path.parent / "signatures.jsonl", sig_updates)


if __name__ == "__main__":
    main()
