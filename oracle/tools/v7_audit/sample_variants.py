"""Pick surface-variant rows from a training file and write them as oracle items, so their text is run on a TPU.

  python3 -I sample_variants.py TRAIN.jsonl OUT_PREFIX N_PARTS [PER_STRATUM] --items ITEMS.jsonl [ITEMS.jsonl ...]

A variant is a generator-v7 row whose target or reference differs from its judged item's. Strata are (stem prefix, family,
transposed, first argument name of workload in the reference); PER_STRATUM rows are taken from each (default 2, 0 = all),
in file order, and dealt round-robin into N_PARTS item files.
"""
import ast, collections, importlib.util, json, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v", os.path.join(here, "verify_v7.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
STEM = re.compile(r"gen7[A-Z]?__[a-z_]+?__\d{5}")


def first_arg(ref):
    try:
        fn = next(n for n in ast.parse(ref).body if isinstance(n, ast.FunctionDef) and n.name == "workload")
        return fn.args.args[0].arg
    except (SyntaxError, StopIteration, IndexError):
        return "?"


def main():
    args = sys.argv[1:]
    cut = args.index("--items")
    train, prefix, n_parts = args[0], args[1], int(args[2])
    per = int(args[3]) if cut > 3 else 2
    items = {}
    for p in args[cut + 1:]:
        for it in v.read_jsonl(p)[0]:
            m = STEM.search(f"{it.get('task')} {it.get('id')}")
            if m:
                items[m.group(0)] = it
    k = v.bench_constants()
    strata = collections.defaultdict(list)
    n_var = 0
    for i, r in enumerate(v.read_jsonl(train)[0]):
        if r.get("teacher") != "generator-v7":
            continue
        m = STEM.search(f"{r.get('task')} {r.get('example_id')} {r.get('surface_of')} {r.get('variant_of')}")
        it = items.get(m.group(0)) if m else None
        code, ref = v.unfence(r["messages"][-1].get("content") or ""), v.ref_of(r, k)
        same_code = bool(code) and code.rstrip("\n") == (it.get("code") or "").rstrip("\n") if it else False
        if it is None or not code or not ref or (same_code and ref == it.get("ref_code")):
            continue
        n_var += 1
        key = (m.group(0).split("__")[0], it.get("family"), bool((it.get("layout") or {}).get("transposed")), first_arg(ref), "ref-only" if same_code else "renamed")
        strata[key].append({"id": f"gen7V/{r.get('example_id') or m.group(0) + '__v'}", "task": r.get("task"), "ref_code": ref, "code": code})
    picked = [x for key in sorted(strata, key=str) for x in (strata[key] if per == 0 else strata[key][:per])]
    assert len({x["id"] for x in picked}) == len(picked), "duplicate ids"
    for p in range(n_parts):
        part = picked[p::n_parts]
        open(f"{prefix}{'abcdefgh'[p]}.items.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in part))
    print(f"{n_var} variants in {len(strata)} strata; picked {len(picked)} into {n_parts} parts")
    for key in sorted(strata, key=str):
        print("  ", key, len(strata[key]))


if __name__ == "__main__":
    main()
