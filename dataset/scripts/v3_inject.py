"""Dataset v3, part 1: error-injection repair pairs. Break each TPU-verified train kernel the way LoRA v1 breaks, ask the oracle's own
kernel check what is wrong, and emit [problem, broken kernel, the oracle's feedback verbatim, original kernel].

  python dataset/scripts/v3_inject.py [--limit N] [--python PY]

A mutation only becomes a row if the oracle really rejects the broken kernel. Edits are spliced into the source text at AST offsets, so
the rest of each broken kernel is byte-identical to its target. Writes dataset/v3/repairs.jsonl and repairs_report.json.
"""
from __future__ import annotations

import argparse, ast, collections, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import write_json, write_jsonl  # noqa: E402
from v3_common import ORACLE_SHA, V3, checked_in_batches, derived_row, train_singles  # noqa: E402


class Src:
    """Source text addressed by AST node. col_offset is in UTF-8 bytes, so all slicing is on bytes."""

    def __init__(self, code: str):
        self.b, self.tree = code.encode(), ast.parse(code)
        self.starts = [0]
        for line in self.b.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))

    def span(self, n) -> tuple:
        return self.starts[n.lineno - 1] + n.col_offset, self.starts[n.end_lineno - 1] + n.end_col_offset

    def text(self, n) -> str:
        a, b = self.span(n)
        return self.b[a:b].decode()

    def splice(self, edits: list) -> str:
        """`edits` are (start, end, replacement); applied from the end so offsets stay valid."""
        b = self.b
        for a, e, new in sorted(edits, reverse=True):
            b = b[:a] + new.encode() + b[e:]
        return b.decode()


def _name(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def pallas_call(s: Src):
    """The first pallas_call in source order, or None."""
    calls = [n for n in ast.walk(s.tree) if isinstance(n, ast.Call) and _name(n) == "pallas_call"]
    return min(calls, key=lambda n: (n.lineno, n.col_offset)) if calls else None


def _spec_kw(s: Src, call: ast.Call, name: str):
    """in_specs or out_specs, on the call or nested in grid_spec=PrefetchScalarGridSpec(...); a bare name resolves to its literal assignment."""
    v = _kw(call, name)
    if v is None and isinstance(_kw(call, "grid_spec"), ast.Call):
        v = _kw(_kw(call, "grid_spec"), name)
    if isinstance(v, ast.Name):
        found = [n.value for n in ast.walk(s.tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == v.id for t in n.targets)
                 and n.lineno < v.lineno and isinstance(n.value, (ast.List, ast.Tuple, ast.Call))]
        return found[-1] if found else None
    return v


def _specs(s: Src, call: ast.Call) -> list:
    """The BlockSpec calls of this pallas_call, in source order."""
    vals = [v for v in (_spec_kw(s, call, "in_specs"), _spec_kw(s, call, "out_specs")) if v is not None]
    return sorted((n for v in vals for n in ast.walk(v) if isinstance(n, ast.Call) and _name(n) == "BlockSpec"), key=lambda n: (n.lineno, n.col_offset))


def out_shape_int(s: Src, call: ast.Call, rng) -> str | None:
    v = _kw(call, "out_shape")
    if isinstance(v, (ast.List, ast.Tuple)) and v.elts:
        v = v.elts[0]
    if isinstance(v, ast.Call) and _name(v) == "ShapeDtypeStruct" and v.args and isinstance(v.args[0], ast.Tuple) and v.args[0].elts:
        return s.splice([(*s.span(v), s.text(v.args[0].elts[0]))])
    return None


def index_map_arity(s: Src, call: ast.Call, rng) -> str | None:
    lams = [n for sp in _specs(s, call) for n in ast.walk(sp) if isinstance(n, ast.Lambda) and n.args.args]
    if not lams:
        return None
    lam = rng.choice(lams)
    args = lam.args.args
    first, last = s.span(args[0])[0], s.span(args[-1])[1]
    if len(args) >= 2 and rng.random() < 0.5:  # one too few: the dropped index becomes 0, so the body still resolves
        edits = [(first, last, ", ".join(a.arg for a in args[:-1]))]
        edits += [(*s.span(n), "0") for n in ast.walk(lam.body) if isinstance(n, ast.Name) and n.id == args[-1].arg]
        return s.splice(edits)
    extra = next(c for c in ("k", "l", "m", "n", "p", "q", "r", "t") if c not in {a.arg for a in args})
    return s.splice([(first, last, ", ".join(a.arg for a in args) + f", {extra}")])


def in_specs_count(s: Src, call: ast.Call, rng) -> str | None:
    v = _spec_kw(s, call, "in_specs")
    if not isinstance(v, (ast.List, ast.Tuple, ast.Call)):
        return None
    if isinstance(v, (ast.List, ast.Tuple)) and len(v.elts) >= 2:
        return s.splice([(*s.span(v), "[" + ", ".join(s.text(e) for e in v.elts[:-1]) + "]")])
    e = v.elts[0] if isinstance(v, (ast.List, ast.Tuple)) and v.elts else v
    return s.splice([(*s.span(v), f"[{s.text(e)}, {s.text(e)}]")])


def block_rank(s: Src, call: ast.Call, rng) -> str | None:
    tups = [t for sp in _specs(s, call) for t in ([sp.args[0]] if sp.args else [_kw(sp, "block_shape")]) if isinstance(t, ast.Tuple) and t.elts]
    if not tups:
        return None
    t = rng.choice(tups)
    if len(t.elts) >= 2:
        return s.splice([(*s.span(t), "(" + ", ".join(s.text(e) for e in t.elts[:-1]) + ("," if len(t.elts) == 2 else "") + ")")])
    return s.splice([(*s.span(t), f"({s.text(t.elts[0])}, 1)")])


def invented_kwarg(s: Src, call: ast.Call, rng) -> str | None:
    specs = [sp for sp in _specs(s, call) if isinstance(sp.func, ast.Attribute)]
    v = _kw(call, "out_shape")
    if specs and (v is None or rng.random() < 0.4):
        e = s.span(rng.choice(specs).func)[1]
        return s.splice([(e - len("BlockSpec"), e, "BlockMap")])
    if v is None:
        return None
    return s.splice([(s.span(v)[1], s.span(v)[1], f", x_shape={s.text(v)}")])


def missing_workload(s: Src, call: ast.Call, rng) -> str | None:
    for n in s.tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "workload":
            a = s.span(n)[0] + s.b[s.span(n)[0]:].index(b"workload")
            return s.splice([(a, a + len("workload"), rng.choice(("run", "kernel", "main", "forward", "compute")))])
    return None


MUTATIONS = {"out_shape_int": out_shape_int, "index_map_arity": index_map_arity, "in_specs_count": in_specs_count, "block_rank": block_rank,
             "invented_kwarg": invented_kwarg, "missing_workload": missing_workload}


def mutants(code: str, seed_key: str) -> dict:
    s, out = Src(code), {}
    call = pallas_call(s)
    if call is None:
        return out
    for name, f in MUTATIONS.items():
        new = f(s, call, random.Random(f"{seed_key}/{name}"))
        if new is not None and new != code:
            out[name] = new
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    bases = train_singles()[: a.limit or None]
    items, refs, info, skipped = [], {}, {}, collections.Counter()
    for e in bases:
        task, target = e["task_name"], e["turns"][-1]["code"]
        refs[task] = e["turns"][0]["ref_code"]
        got = mutants(target, e["example_id"])
        skipped.update({m: 1 for m in MUTATIONS if m not in got})
        for m, broken in got.items():
            cid = f"{task}.inj.{m}"
            items.append({"id": cid, "task": task, "code": broken})
            info[cid] = (e, m, broken)
    print(f"{len(bases)} kernels, {len(items)} mutants; not applicable: {dict(skipped)}", flush=True)
    rows = checked_in_batches(items, refs, a.python, log=lambda m: print(m, flush=True))
    out, outcome = [], collections.Counter()
    for cid, (e, m, broken) in info.items():
        r = rows.get(cid)
        v = (r or {}).get("verdict", "no_row")
        outcome[f"{m}:{v}"] += 1
        if v not in ("compile", "not_a_kernel") or not r.get("feedback"):
            continue  # the oracle did not reject it, or it failed for a harness reason: not a usable repair pair
        turns = [e["turns"][0],
                 {"role": "assistant", "type": "kernel", "code": broken, "candidate_id": cid},
                 {"role": "user", "type": "oracle_feedback", "feedback": r["feedback"], "verdict": v,
                  "verdict_ref": {"job_id": "v3-static", "oracle_sha": ORACLE_SHA}},
                 e["turns"][-1]]
        out.append(derived_row(e, example_id=cid, turns=turns, kind="repair", origin="injected_repair", verification="tpu", extra={"mutation": m}))
    out.sort(key=lambda x: x["example_id"])
    V3.mkdir(parents=True, exist_ok=True)
    name = "repairs" if not a.limit else "repairs_sample"
    write_jsonl(V3 / f"{name}.jsonl", out)
    write_json(V3 / f"{name}_report.json", {"n_kernels": len(bases), "n_mutants": len(items), "n_rows": len(out), "not_applicable": dict(skipped),
                                            "outcome": dict(sorted(outcome.items())), "oracle_sha": ORACLE_SHA})
    print(f"{len(out)} repair rows of {len(items)} mutants")
    print(dict(sorted(outcome.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
