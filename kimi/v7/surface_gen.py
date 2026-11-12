"""Surface variants of the template-generated rows: the TPU-verified generator rows (e5_gen) and Aarav's layout items (gen7L), at zero TPU cost.

Why. The generator shows the policy kernel (K tiled, f32 accumulator, vmem_limit 124 MiB) under ONE surface: workload(a, b[, s][, v]), CONFIG
keys m/n/k, CONFIG name 'gen7_*' / 'gen7L_*', docstring 'Matmul -> ...'. No other row in v7 has that surface, so the cheapest thing the student
can learn is "gen7 surface -> policy kernel", and the exam's matmul tasks do not have that surface (they look like the pool's KernelBench
ports: batch_size / in_features / out_features, workload(x, weight, bias), `x = jnp.matmul(x, weight.T) + bias`). This script re-renders each
row under a random surface, so the policy kernel is tied to the operation and the sizes, not to the wording.

What changes, and what cannot. Only identifiers and wording:
  reference   argument / result / PRNG-key variable names, CONFIG key names and CONFIG name, how create_inputs reads the sizes, the docstring,
              `jnp.matmul(a, b)` vs `a @ b` (with `.T` / `jnp.swapaxes` kept as written), the bias (or scale and shift) on the matmul line or the
              next one.  Same PRNG keys, shapes, dtypes, scales, same ops in the same order.
  target      the same kernel with workload's parameters (and the kernel's matching *_ref names) renamed to the reference's new names.
Nothing here is a new kernel. check() proves it per row:
  1. target: renaming the new names back gives an AST identical to the original target's AST;
  2. reference: at a shrunken CONFIG the new create_inputs returns bit-identical arrays and the new workload a bit-identical output.
So every variant inherits its original's TPU verdict (same program, same inputs). The variant's text itself was not run on a TPU.

Sources (SRC = gen | layout | rowout; default gen):
  gen      v7_gen_rows.jsonl: the TPU-passed generator rows (aarav/dataset-v4 @ de8f43a)                     -> gen_surface/
  layout   layout_items.jsonl: aarav/dataset-v4 @ 098bb22 dataset/v7_gen_layout/items.jsonl (127 CPU-verified  -> layout_surface/
           items, not yet judged); variants are built for every item and only exported for rows the TPU passed

  python3 surface_gen.py build  [SRC]          -> <dir>/variants.jsonl
  python3 surface_gen.py check  [SRC]          -> <dir>/check.json   (the two proofs above; needs jax on CPU; small arrays, one process)
  python3 surface_neardup.py <aarav/dataset-v4 dataset/scripts dir> [SRC]   -> <dir>/neardup.json (the team's near-duplicate filter on the new text)
  python3 surface_gen.py export                -> v7_gen_rows_surf_add.jsonl      originals + variants of the train rows   (for assemble_v7 --gen)
                                                  v7_gen_rows_surf_replace.jsonl  each train row once: variant (75%) or original (25%)
                                                  dev_items_gen_surface.jsonl     the dev items under a new surface (a pre-exam transfer check)
  python3 surface_gen.py export-rowout [ROWS] [OUT]   gen7R rows (default v7_rowout_rows.jsonl) -> v7_rowout_rows_surf_add.jsonl; same conditions,
        plus Karthik's oracle verdict for the item (layout/gen7R-*.jsonl) must be pass at >= 0.7x; drops from a6r2/gen7R_drops.json
  python3 surface_gen.py export-layout ROWS [OUT]
        ROWS = the TPU-verified layout rows (any rows file; rows whose task starts with gen7L__ are used). A variant is written only if its row
        passed (verdict pass), is not on a6r2/aarav_drops.json's drop lists or the dev list, survived the near-duplicate filter, and the row's
        reference and target are byte-identical (up to trailing newlines) to the item the variant was built from.
        OUT (default v7_layout_rows_surf_add.jsonl) = those layout rows + their variants; also writes v7_gen_plus_layout_surf_add.jsonl =
        v7_gen_rows_surf_add.jsonl + OUT, one file for assemble_v7 --gen.
"""
import ast, hashlib, io, json, pathlib, random, re, sys, tokenize

V7 = pathlib.Path(__file__).resolve().parent
OUT = V7 / "gen_surface"
GEN = V7 / "v7_gen_rows.jsonl"
DEV = V7 / "dev_items_gen.jsonl"
LAYOUT_ITEMS = V7 / "layout_items.jsonl"
SOURCES = {"gen": V7 / "gen_surface", "layout": V7 / "layout_surface", "rowout": V7 / "rowout_surface"}
ITEMS = {"layout": LAYOUT_ITEMS, "rowout": V7 / "layout/items_gen7R.jsonl"}  # rowout: Karthik's gen7R items (oracle-results/karthik-gen7)
DROPS = V7 / "a6r2/aarav_drops.json"
JAXBENCH = pathlib.Path("/Users/hrudaynara/Projects/gemma-hackathon-research/benchmarks/jaxbench")
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
import bench  # noqa: E402  (conversation(): the eval's own prompt)

# (a, b, v) -> new names. None of these may collide with a name the kernels already use (see FORBIDDEN).
NAMES_2D = [(("x", "weight", "bias"), 30), (("x", "w", "b"), 22), (("inputs", "weights", "bias"), 10), (("h", "W", "b"), 10),
            (("x", "W", "b"), 8), (("inp", "weight", "bias"), 5), (("a", "b", "v"), 15)]
NAMES_3D = [(("x1", "x2", "bias"), 30), (("lhs", "rhs", "bias"), 25), (("p", "q", "b"), 15), (("a", "b", "v"), 30)]
# (s, v) -> new names when the reference has a per-feature scale and shift (layout items only; drawn from a separate RNG stream)
NAMES_SV = [(("scale", "shift"), 30), (("gamma", "beta"), 25), (("bn_scale", "bn_bias"), 20), (("scale", "bias"), 15), (("s", "v"), 10)]
KEYS_2D = [(("batch_size", "out_features", "in_features"), 40), (("m", "n", "k"), 15), (("rows", "cols", "inner"), 10),
           (("batch", "d_out", "d_in"), 15), (("n_samples", "n_out", "n_in"), 10), (("batch_size", "hidden_size", "input_size"), 10)]
KEYS_3D = [(("batch", "m", "n", "k"), 35), (("batch_size", "rows", "cols", "inner"), 30), (("num_heads", "seq_len", "d_out", "d_in"), 35)]
PRNG = [("k{i}", 40), ("named", 35), ("key_{i}", 25)]
STEMS = ["dense", "linear_head", "proj", "mlp_layer", "fc", "readout", "classifier_head", "embed_proj", "feature_map", "output_layer"]
FORBIDDEN = {"jax", "jnp", "pl", "pltpu", "lax", "VMEM_LIMIT", "_tile", "n", "pref", "t", "_kernel", "o_ref", "acc", "_", "y", "workload", "M", "K", "N", "B",
             "tm", "tn", "tk", "i", "j", "l", "b_i", "c", "CONFIG", "dtype", "create_inputs", "r", "ch"}  # r, ch: the gen7R kernels' row-slice loop
MATMUL_LINE = re.compile(r"^    y = jnp\.matmul\(a, (b|b\.T|jnp\.swapaxes\(b, -1, -2\))\)$")
BT = {"b": "none", "b.T": "T", "jnp.swapaxes(b, -1, -2)": "swap"}


def pick(rng, table):
    r = rng.uniform(0, sum(w for _, w in table))
    for v, w in table:
        r -= w
        if r <= 0:
            return v
    return table[-1][0]


def rename(code: str, mp: dict) -> str:
    """Rename NAME tokens (not attributes) simultaneously; formatting and comments are kept. A keyword argument named like a renamed name
    would be renamed too; the AST proof in check() would then fail, so such a row can never be exported."""
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    out, prev = [], None
    for t in toks:
        s = t.string
        if t.type == tokenize.NAME and s in mp and not (prev and prev.string == "."):
            s = mp[s]
        out.append(s); prev = t
    return _untok(code, toks, out)


def _untok(code: str, toks, strings) -> str:
    """Rebuild the source with replaced token strings, keeping every byte between tokens."""
    lines = code.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    pos = lambda rc: offs[rc[0] - 1] + rc[1]
    res, last = [], 0
    for t, s in zip(toks, strings):
        if t.type in (tokenize.ENDMARKER,) or t.string == "" and t.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        a, b = pos(t.start), pos(t.end)
        if a < last:
            continue
        res.append(code[last:a]); res.append(s); last = b
    res.append(code[last:])
    return "".join(res)


def parse_ref(ref: str) -> dict:
    """The generator's reference (v3_generate.ref_module, v7_gen --layout) back into its parts."""
    doc = re.match(r'"""(.*?)"""\n', ref, re.S).group(1)
    cfg = ast.literal_eval(re.search(r"CONFIG = (\{.*?\n\})", ref, re.S).group(1))
    dtype = re.search(r"def create_inputs\(dtype=(jnp\.\w+)\):", ref).group(1)
    nkeys = int(re.search(r"jax\.random\.split\(jax\.random\.key\(0\), (\d+)\)", ref).group(1))
    draws = re.findall(r"^    (\w+) = jax\.random\.normal\(k(\d+), (\(.*?\)), dtype\) \* ([\d.]+)$", ref, re.M)
    assert len(draws) == nkeys and [int(d[1]) for d in draws] == list(range(nkeys)), ref
    sig = re.search(r"def workload\((.*?)\):\n", ref).group(1)
    body = ref[ref.index("def workload("):].split("\n", 1)[1].rstrip("\n").split("\n")
    m = MATMUL_LINE.match(body[0])
    assert m and body[-1] == "    return y", body
    bt = BT[m.group(1)]
    vec = body[1] if len(body) > 1 and body[1] in ("    y = y + v", "    y = y * s + v") else None
    bias, scale = vec is not None, vec == "    y = y * s + v"
    names = ["a", "b"] + (["s"] if scale else []) + (["v"] if bias else [])
    assert sig == ", ".join(names) and [d[0] for d in draws] == names, (sig, draws, body)
    epi = body[2 if bias else 1:-1]
    assert all(l.startswith("    y = ") for l in epi), epi
    return {"doc": doc, "cfg": cfg, "dtype": dtype, "draws": [(d[0], d[2], d[3]) for d in draws], "bias": bias, "scale": scale, "bt": bt,
            "epi": epi, "batched": "batch" in cfg}


def jaxbench_names() -> set:
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    out = set()
    for p in JAXBENCH.glob("level*/*/baseline.py"):
        out.add(norm(p.parent.name))
        m = re.search(r"'name': '([^']+)'", p.read_text())
        if m:
            out.add(norm(m.group(1)))
    return out


def render(ref: str, code: str, rng, rng2, jb: set) -> dict:
    """One surface variant of (reference, kernel): {ref_code, code, surface}. `rng` makes every choice the generator rows have; `rng2` only
    the layout-only ones (scale/shift names, the transposed-weight wording), so the generator rows' variants do not change."""
    p = parse_ref(ref)
    A, Bn, Vn = pick(rng, NAMES_3D if p["batched"] else NAMES_2D)
    keys = pick(rng, KEYS_3D if p["batched"] else KEYS_2D)
    kmap = dict(zip(("batch", "m", "n", "k") if p["batched"] else ("m", "n", "k"), keys))
    res = pick(rng, [("y", 35), ("out", 15), ("z", 10), (A, 40)])  # A: the result is assigned back to the first argument
    at, fuse, style, prng = rng.random() < 0.5, rng.random() < 0.5, pick(rng, [("c", 35), ("unpack", 35), ("direct", 30)]), pick(rng, PRNG)
    Sn = None
    if p["scale"]:
        Sn, Vn = pick(rng2, NAMES_SV)
        if len({A, Bn, Sn, Vn}) < 4:
            Sn, Vn = ("scale", "shift") if len({A, Bn, "scale", "shift"}) == 4 else ("s", "v")
    names = {"a": A, "b": Bn, "v": Vn, **({"s": Sn} if Sn else {})}
    used = [names[d[0]] for d in p["draws"]]
    assert not (set(names.values()) - {"a", "b", "s", "v"}) & FORBIDDEN and len(set(names.values())) == len(names)
    # ---- CONFIG: the pool's KernelBench ports list batch, in, out; keep m, n, k order for the m/n/k scheme
    order = (["batch"] if p["batched"] else []) + (["m", "n", "k"] if keys[-3:] in (("m", "n", "k"), ("rows", "cols", "inner")) else ["m", "k", "n"])
    segs = re.split(r" -> ", re.sub(r" over the last axis\.$|\.$", "", p["doc"]))
    ops = [o for o in segs[1:] if o not in ("bias", "scale and shift")]
    head = ("bmm" if p["batched"] else "linear" if p["bias"] else "matmul")
    name = "_".join([head] + ops) if rng.random() < 0.5 else f"{rng.choice(STEMS)}_{rng.randrange(1000)}"
    if re.sub(r"[^a-z]", "", name.lower()) in jb or any(re.sub(r"[^a-z]", "", name.lower()) == re.sub(r"\d", "", x) for x in jb):
        name = f"{rng.choice(STEMS)}_{rng.randrange(1000)}"
    cfg = {"name": name, **{kmap[k]: p["cfg"][k] for k in order}}
    config = "CONFIG = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in cfg.items()) + "}\n"
    # ---- create_inputs: same key, same split, same shapes / dtype / scales; only names and the way sizes are read differ
    n = len(p["draws"])
    kn = {"k{i}": [f"k{i}" for i in range(n)], "key_{i}": [f"key_{i}" for i in range(n)],
          "named": [f"k_{x}" if len(x) < 3 else f"k{x[0]}" for x in used]}[prng]
    if len(set(kn)) < n:
        kn = [f"k{i}" for i in range(n)]
    U = {"batch": "B", "m": "M", "n": "N", "k": "K"}
    def shape(expr: str) -> str:
        f = {"c": lambda k: f"c['{kmap[k]}']", "direct": lambda k: f"CONFIG['{kmap[k]}']", "unpack": lambda k: U[k]}[style]
        return re.sub(r"c\['(\w+)'\]", lambda m: f(m.group(1)), expr)
    pre = {"c": "    c = CONFIG\n", "direct": "",
           "unpack": "".join(f"    {U[k]} = CONFIG['{kmap[k]}']\n" for k in order)}[style]
    draws = "".join(f"    {names[nm]} = jax.random.normal({kn[i]}, {shape(sh)}, dtype) * {sc}\n" for i, (nm, sh, sc) in enumerate(p["draws"]))
    ci = (f"def create_inputs(dtype={p['dtype']}):\n{pre}    {', '.join(kn)} = jax.random.split(jax.random.key(0), {n})\n{draws}"
          f"    return {', '.join(used)}\n")
    # ---- workload: same ops in the same order
    W = {"none": Bn, "T": f"{Bn}.T", "swap": f"jnp.swapaxes({Bn}, -1, -2)"}[p["bt"]]
    mm = f"{A} @ {W}" if at else f"jnp.matmul({A}, {W})"
    vexpr = f" * {Sn} + {Vn}" if p["scale"] else f" + {Vn}"
    lines = [f"    {res} = {mm}{vexpr}"] if (p["bias"] and fuse) else [f"    {res} = {mm}"] + ([f"    {res} = {res}{vexpr}"] if p["bias"] else [])
    lines += [rename(l + "\n", {"y": res}).rstrip("\n") for l in p["epi"]] + [f"    return {res}"]
    sig = ", ".join(used)
    # ---- docstring
    tail = " over the last axis" if p["doc"].endswith("over the last axis.") else ""
    first = ("Batched matmul" if p["batched"] else "Linear layer (matmul + bias)" if p["bias"] and not p["scale"] else
             "Linear layer (matmul, per-feature scale and shift)" if p["scale"] else "Matmul")
    if p["bt"] != "none":
        first += pick(rng2, [(" with the weight stored transposed", 40), (" with a (out, in) weight", 30), (" against the transposed weight", 30)])
    doc = pick(rng, [(p["doc"], 30), (f"{first}, then {', '.join(ops)}{tail}.", 25), (f"{first} followed by {' -> '.join(ops)}{tail}.", 25),
                     (f"{' + '.join([first] + ops)}{tail}.", 20)])
    new_ref = f'"""{doc}"""\nimport jax\nimport jax.numpy as jnp\nfrom jax import lax\n\n{config}\n\n{ci}\n\ndef workload({sig}):\n' + "\n".join(lines) + "\n"
    tmap = {"a": A, "b": Bn, "v": Vn, "a_ref": f"{A}_ref", "b_ref": f"{Bn}_ref", "v_ref": f"{Vn}_ref"}
    if Sn:
        tmap.update({"s": Sn, "s_ref": f"{Sn}_ref"})
    surface = {"names": used, "config_keys": list(keys), "config_name": name, "result": res, "matmul": "@" if at else "jnp.matmul",
               "bias_fused": bool(p["bias"] and fuse), "sizes": style, "prng": prng, "doc": doc}
    if p["bt"] != "none" or p["scale"]:
        surface.update(weight=p["bt"], scale=p["scale"])
    return {"ref_code": new_ref, "code": rename(code, tmap), "orig_ref": ref, "orig_code": code, "surface": surface, "tmap": tmap, "kmap": kmap}


def _ref_of(row: dict) -> str:
    u = row["messages"][1]["content"]
    return u[u.index("```python\n") + 10:u.rindex("```")]


def _target_of(row: dict) -> str:
    tgt = row["messages"][-1]["content"]
    assert tgt.startswith("```python\n") and tgt.endswith("\n```")
    return tgt[len("```python\n"):-len("\n```")]


def _variant(meta: dict, eid: str, surface_of: str, is_dev: bool, v: dict) -> dict:
    return {**meta, "example_id": eid, "surface_of": surface_of, "is_dev": is_dev, "surface": v["surface"],
            "messages": bench.conversation(v["ref_code"], [], None) + [{"role": "assistant", "content": "```python\n" + v["code"].rstrip("\n") + "\n```"}],
            "_proof": {"orig_ref": v["orig_ref"], "orig_code": v["orig_code"], "ref_code": v["ref_code"], "code": v["code"], "tmap": v["tmap"], "kmap": v["kmap"]}}


def build(src: str = "gen") -> None:
    d = SOURCES[src]; d.mkdir(exist_ok=True)
    dev = {json.loads(l)["example_id"] for l in open(DEV)}
    jb, out = jaxbench_names(), []
    if src == "gen":
        for l in open(GEN):
            r = json.loads(l)
            v = render(_ref_of(r), _target_of(r), random.Random("v7surface/" + r["example_id"]), random.Random("v7surface/L/" + r["example_id"]), jb)
            out.append(_variant({k: x for k, x in r.items() if k != "messages"}, r["example_id"] + "s", r["example_id"], r["example_id"] in dev, v))
    else:
        for l in open(ITEMS[src]):
            it = json.loads(l)
            v = render(it["ref_code"], it["code"], random.Random("v7surface/" + it["id"]), random.Random("v7surface/L/" + it["id"]), jb)
            meta = {"task": it["task"], "family": it["family"], "layout": it.get("layout"), "item_id": it["id"]}
            out.append(_variant(meta, it["id"] + "s", it["id"], False, v))  # example_id / surface_of are re-keyed to the verified row at export
    (d / "variants.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    import collections
    print(f"{len(out)} variants ({sum(x['is_dev'] for x in out)} of dev items) -> {d / 'variants.jsonl'}")
    for k in ("names", "config_keys", "result", "matmul", "sizes", "weight", "scale"):
        print(" ", k, dict(collections.Counter(str(x["surface"].get(k)) for x in out).most_common(8)))


class _Back(ast.NodeTransformer):
    def __init__(self, inv):
        self.inv = inv
    def visit_Name(self, n):
        n.id = self.inv.get(n.id, n.id); return n
    def visit_arg(self, n):
        n.arg = self.inv.get(n.arg, n.arg); return n


def check(src: str = "gen") -> int:
    """Proof 1 (AST): the variant's target is the original target under a consistent renaming. Proof 2 (numerics, CPU, shrunken CONFIG):
    create_inputs gives bit-identical arrays and workload a bit-identical output."""
    import numpy as np
    import jax  # noqa: F401
    d = SOURCES[src]
    rows = [json.loads(l) for l in open(d / "variants.jsonl")]
    bad = []
    SMALL = {"batch": 2, "m": 4, "n": 8, "k": 16}
    for r in rows:
        p = r["_proof"]
        # 1. target
        inv = {v: k for k, v in p["tmap"].items()}
        assert len(inv) == len(p["tmap"])
        new_names = set(p["tmap"].values()) - set(p["tmap"])
        orig_ids = {t.string for t in tokenize.generate_tokens(io.StringIO(p["orig_code"]).readline) if t.type == tokenize.NAME}
        if new_names & orig_ids:  # a new name already means something else in the kernel: the renaming would capture it
            bad.append((r["example_id"], "name capture", sorted(new_names & orig_ids))); continue
        back = ast.dump(_Back(inv).visit(ast.parse(p["code"])))
        if back != ast.dump(ast.parse(p["orig_code"])):
            bad.append((r["example_id"], "target AST differs")); continue
        # 2. reference
        outs = []
        for s, km in ((p["orig_ref"], {k: k for k in p["kmap"]}), (p["ref_code"], p["kmap"])):
            ns = {}
            exec(compile(s, "<ref>", "exec"), ns)
            for k, key in km.items():
                assert key in ns["CONFIG"], (key, ns["CONFIG"])
                ns["CONFIG"][key] = SMALL[k]
            ins = ns["create_inputs"]()
            outs.append(([np.asarray(x) for x in ins], np.asarray(ns["workload"](*ins))))
        (i0, o0), (i1, o1) = outs
        same = len(i0) == len(i1) and all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes() for a, b in zip(i0, i1)) \
            and o0.dtype == o1.dtype and o0.shape == o1.shape and o0.tobytes() == o1.tobytes()
        if not same:
            bad.append((r["example_id"], "reference differs"))
        # 3. the sizes are the original's
        c0 = ast.literal_eval(re.search(r"CONFIG = (\{.*?\n\})", p["orig_ref"], re.S).group(1)); c1 = ast.literal_eval(re.search(r"CONFIG = (\{.*?\n\})", p["ref_code"], re.S).group(1))
        if any(c0[k] != c1[key] for k, key in p["kmap"].items()) or len(c1) != len(c0):
            bad.append((r["example_id"], "CONFIG sizes differ"))
    print(f"{len(rows)} variants checked: {len(rows) - len({b[0] for b in bad})} proven identical to their original (target AST under renaming; "
          f"reference inputs and output bit-identical at a shrunken size)")
    if bad:
        print("FAILURES:", bad[:10])
    (d / "check.json").write_text(json.dumps({"variants": len(rows), "failures": bad}, indent=1))
    return 1 if bad else 0


def _load_variants(d: pathlib.Path) -> tuple:
    """({surface_of: variant or None if the near-duplicate filter excluded it}, [excluded ids]) after asserting check and neardup ran clean."""
    chk = json.load(open(d / "check.json"))
    assert chk["failures"] == [], f"run check first; it must be clean ({d})"
    near = {x["example_id"]: x for x in json.load(open(d / "neardup.json"))}
    var, dropped = {}, []
    for l in open(d / "variants.jsonl"):
        r = json.loads(l); r.pop("_proof")
        assert r["example_id"] in near, f"run surface_neardup.py after build ({d})"
        if near[r["example_id"]]["excluded"]:
            dropped.append(r["example_id"]); var[r["surface_of"]] = None
        else:
            var[r["surface_of"]] = r
    assert chk["variants"] == len(var)
    return var, dropped


def export() -> None:
    # the team's layer-3 contamination filter (near-duplicate text against the 73 exam files), run on the NEW reference text by
    # surface_neardup.py with aarav/dataset-v4's filter_neardup.py: a variant at or over its bar is dropped and the original is used
    var, dropped = _load_variants(SOURCES["gen"])
    print(f"variants dropped by the near-duplicate filter: {dropped}")
    orig = [json.loads(l) for l in open(GEN)]
    assert len(var) == len(orig)
    # read the dev set and the A6 rule-2 drops NOW, not at build time: both changed at 02:50Z (dev 27 -> 29 items, 33 generated rows dropped).
    # A variant has its own example_id, so assemble_v7's example_id holdouts would not catch it: a variant of a held-out row is never written.
    dev_now = {json.loads(l)["example_id"] for l in open(DEV)}
    r2g = set(json.load(open(DROPS))["drop_generated_training_example_ids"]) if DROPS.exists() else set()
    for r in orig:
        if var[r["example_id"]] is not None:
            var[r["example_id"]]["is_dev"] = r["example_id"] in dev_now
    add, rep, devrows = [], [], []
    for r in orig:
        v = var[r["example_id"]]
        if v is None or (r["example_id"] in r2g and not v["is_dev"]):
            add.append(r); rep.append(r)  # the original stays; assemble_v7 drops it by example_id (R2G); its variant is never written
            continue
        if v["is_dev"]:
            add.append(r); rep.append(r)  # the original stays so assemble_v7's --dev-gen holdout sees it; its variant is never a training row
            devrows.append({"example_id": v["example_id"], "surface_of": r["example_id"], "task": r["task"], "family": r["family"], "surface": v["surface"],
                            "ref_code": v["messages"][1]["content"][v["messages"][1]["content"].index("```python\n") + 10:v["messages"][1]["content"].rindex("```")],
                            "messages": v["messages"][:-1], "verified_target": v["messages"][-1]["content"]})
            continue
        v = {k: x for k, x in v.items() if k != "is_dev"}
        add += [r, v]
        rep.append(v if random.Random("v7surface/replace/" + r["example_id"]).random() < 0.75 else r)
    for name, rows in (("v7_gen_rows_surf_add.jsonl", add), ("v7_gen_rows_surf_replace.jsonl", rep), ("dev_items_gen_surface.jsonl", devrows)):
        body = "".join(json.dumps(x) + "\n" for x in rows)
        (V7 / name).write_text(body)
        print(f"{name}: {len(rows)} rows, sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]}")


def _export_rows(src: str, rows_path: str, out: str, prefix: str, drop_items: set, drop_rows: set, judged: dict = None) -> str:
    """Template rows (TPU-verified) + one variant each, keyed to the verified row. A variant is written only if the row passed, is not dropped,
    not dev, survived the near-duplicate filter, its reference and target equal the item's (up to trailing newlines), and, when `judged`
    (the oracle's rows by item id) is given, the oracle's verdict for that item is pass at >= 0.7x. Returns the written body."""
    import collections
    var, dropped = _load_variants(SOURCES[src])
    items = {json.loads(l)["task"]: json.loads(l) for l in open(ITEMS[src])}
    dev_now = {json.loads(l)["example_id"] for l in open(DEV)}
    rows = [json.loads(l) for l in open(rows_path)]
    rows = [r for r in rows if str(r.get("task", "")).startswith(prefix)]
    st, outrows = collections.Counter(), []
    strip = lambda s: s.rstrip("\n")
    for r in rows:
        it = items.get(r["task"])
        outrows.append(r)  # the original row is passed through as given; assemble_v7 applies its own holdouts and checks to it
        jv = (judged or {}).get(it["id"]) if it else None
        why = ("no_item" if it is None else "not_pass" if r.get("verdict", "pass") != "pass" else
               "not_judged_pass" if judged is not None and not (jv and jv.get("verdict") == "pass" and (jv.get("speed_vs_xla") or 0) >= 0.7) else
               "a6_rule2" if (it["id"] in drop_items or r["task"] in drop_items or r["example_id"] in drop_rows) else "dev" if r["example_id"] in dev_now else
               "neardup" if var.get(it["id"]) is None else
               "text_differs" if (strip(_ref_of(r)) != strip(it["ref_code"]) or strip(_target_of(r)) != strip(it["code"])) else None)
        if why:
            st["no_variant_" + why] += 1; continue
        v = {k: x for k, x in var[it["id"]].items() if k not in ("is_dev", "task", "family", "layout")}
        meta = {k: x for k, x in r.items() if k not in ("messages", "example_id")}
        outrows.append({**meta, **v, "example_id": r["example_id"] + "s", "surface_of": r["example_id"], "surface_of_item": it["id"]})
        st["variant"] += 1
    body = "".join(json.dumps(x) + "\n" for x in outrows)
    (V7 / out).write_text(body)
    print(f"{out}: {len(rows)} {src} rows + {st['variant']} variants = {len(outrows)} rows, sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]} | {dict(st)} "
          f"| near-duplicate exclusions among variants: {len(dropped)}")
    return body


def export_layout(rows_path: str, out: str = "v7_layout_rows_surf_add.jsonl") -> None:
    """Layout rows (TPU-verified) + one variant each, keyed to the verified row. See the module docstring for the conditions."""
    # the union of aarav_drops.json and aarav_drops_latest.json (the file assemble_v7's layout holdout reads; it had 3 more layout ids at 03:29Z)
    drop_items, drop_rows = set(), set()
    for f in (DROPS, V7 / "a6r2/aarav_drops_latest.json"):
        if f.exists():
            d = json.load(open(f))
            drop_items |= set(d.get("drop_layout_item_ids", [])); drop_rows |= set(d.get("drop_generated_training_example_ids", []))
    body = _export_rows("layout", rows_path, out, "gen7L__", drop_items, drop_rows)
    if out == "v7_layout_rows_surf_add.jsonl":
        both = (V7 / "v7_gen_rows_surf_add.jsonl").read_text() + body
        (V7 / "v7_gen_plus_layout_surf_add.jsonl").write_text(both)
        print(f"v7_gen_plus_layout_surf_add.jsonl: {both.count(chr(10))} rows, sha256 {hashlib.sha256(both.encode()).hexdigest()[:16]} (for assemble_v7 --gen)")


def export_rowout(rows_path: str = str(V7 / "v7_rowout_rows.jsonl"), out: str = "v7_rowout_rows_surf_add.jsonl") -> None:
    """Karthik's gen7R rows (aarav 4eab022 v7_rowout_rows.jsonl) + one variant each. Drops: a6r2/gen7R_drops.json (by task). Verdicts:
    layout/gen7R-*.jsonl (Karthik's oracle rows), pass at >= 0.7x required. f32_wholerow_n8192_scratch is never set on a variant
    (these rows replace the scratch-template rows; a variant inherits the row's own fields)."""
    drops = json.load(open(V7 / "a6r2/gen7R_drops.json"))["drops"]
    judged = {}
    for f in sorted((V7 / "layout").glob("gen7R-*.jsonl")):
        for l in open(f):
            if l.strip() and '"meta": true' not in l:
                x = json.loads(l); judged[x["id"]] = x
    body = _export_rows("rowout", rows_path, out, "gen7R__", set(drops), set(), judged)
    bad = [json.loads(l)["example_id"] for l in body.splitlines() if json.loads(l).get("f32_wholerow_n8192_scratch")]
    assert not bad, f"f32_wholerow_n8192_scratch set on {bad[:3]}"


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {"build": build, "check": check, "export": export, "export-layout": export_layout, "export-rowout": export_rowout}[cmd]
    sys.exit(fn(*args) or 0)
