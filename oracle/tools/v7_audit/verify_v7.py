"""Pre-flight checks for a frozen v7 LoRA training file. Read-only, standard library only.

python3 -I verify_v7.py TRAIN.jsonl [--dev DEV.jsonl] [--drops DROPS.json ...] [--verdicts FILE.jsonl ...]
    [--items FILE.jsonl ...] [--base A6_TRAIN.jsonl] [--allow-slow-matmul]

Prints "PASS name: detail" or "FAIL name: detail (first 5 offending ids)" per check, a rows-by-set/kind/teacher
table and the file's sha256; exits 1 if any check fails. Dataset code is parsed with ast, never executed.
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.dirname(os.path.dirname(HERE))
BENCH_PY = os.path.join(SCRATCH, "wt/v4-runs/control/bench.py")
LEVEL1 = os.path.join(SCRATCH, "repo/benchmarks/jaxbench/level1")
BASE_SETS = {"pool", "e1_readmit", "e2_kbkd", "e3_scale"}
TEACHERS = {"kimi-k3.1", "generator-v7"}
BANNED_TEACHER = re.compile(r"claude|opus|sonnet|fable|gpt|gemini", re.I)
REPAIR_TAIL = "\nReply with the corrected Python code block."
SPEED_BAR = 0.7
MAX_CHARS = 70_000
EXAM_CHAINS = {"matmul", "matmul relu", "matmul leaky_relu", "matmul sum_keep", "matmul mish mish", "matmul hardtanh gelu",
               "matmul sigmoid sum_keep", "matmul swish", "matmul softmax", "matmul gelu", "matmul swish tanh gelu hardtanh",
               "matmul gelu softmax", "matmul sigmoid"}
CHAIN_DROP = {"bias", "scale", "shift", "negate", "scale and shift"}
CHAIN_MAP = {"clamp": "hardtanh", "hard_sigmoid": "hardtanh", "mean_keep": "sum_keep", "log_softmax": "softmax"}
MATMUL_CALLS = {"matmul", "dot", "einsum", "dot_general", "linear", "dense", "Dense", "Linear", "batch_matmul", "bmm", "tensordot"}
CONV_CALLS = {"conv", "conv_general_dilated", "conv_transpose", "conv_with_general_padding", "Conv", "ConvTranspose"}

failed = False


def report(name, bad, ok_detail, bad_detail=None, soft=False):
    global failed
    if not bad:
        print(f"PASS {name}: {ok_detail}")
        return
    if soft:
        print(f"PASS {name}: REPORT ONLY, {bad_detail or ''} count={len(bad)} (first 5: {bad[:5]})")
        return
    failed = True
    print(f"FAIL {name}: {bad_detail or ''} count={len(bad)} ({bad[:5]})")


def bench_constants():
    tree = ast.parse(open(BENCH_PY).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t for t in node.targets]
            for t in names:
                if isinstance(t, ast.Name) and t.id == "SYSTEM":
                    out["SYSTEM"] = ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple) and [e.id for e in t.elts] == ["BEFORE_REF", "AFTER_REF"]:
                    out["BEFORE_REF"], out["AFTER_REF"] = ast.literal_eval(node.value)
    missing = {"SYSTEM", "BEFORE_REF", "AFTER_REF"} - out.keys()
    if missing:
        sys.exit(f"cannot read {missing} from {BENCH_PY}")
    return out


def read_jsonl(path):
    rows, bad = [], []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(f"line{i}")
    return rows, bad


def unfence(text):
    """Code inside exactly one ```python block making up the whole message, else None."""
    if text.startswith("```python\n") and text.endswith("\n```") and text.count("```") == 2:
        return text[len("```python\n"):-len("\n```")]
    return None


def ref_of(row, k):
    msgs = row.get("messages") or []
    if len(msgs) < 2:
        return None
    u = msgs[1].get("content", "")
    head, tail = k["BEFORE_REF"] + "\n```python\n", "\n```\n" + k["AFTER_REF"]
    if u.startswith(head) and u.endswith(tail):
        return u[len(head):-len(tail)]
    return None


def norm_chain(ref):
    """Docstring 'Matmul -> bias -> relu.' -> 'matmul relu'; None if there is no docstring."""
    try:
        doc = ast.get_docstring(ast.parse(ref))
    except SyntaxError:
        m = re.match(r'\s*"""(.*?)"""', ref, re.S)
        doc = m.group(1) if m else None
    if not doc:
        return None
    out = []
    for step in doc.strip().rstrip(".").split("->"):
        s = step.strip().lower()
        s = s.replace("with the weight stored (n, k)", "").replace("over the last axis", "")
        s = re.sub(r"\bbatched\b", "", s)
        s = " ".join(s.split())
        if not s or s in CHAIN_DROP:
            continue
        out.append(CHAIN_MAP.get(s, s))
    return " ".join(out)


def id_keys(s):
    """Raw id plus (family, number) so 'gen7__f__00000.gen7', 'gen7/f/00000', 'gen7/gen7__f__00000' all agree."""
    if not isinstance(s, str):
        return set()
    keys = {s}
    toks = [t for t in re.split(r"__|/", s) if t]
    if len(toks) >= 2:
        last = toks[-1].split(".")[0]
        if last.isdigit():
            keys.add((toks[-2], int(last)))
    return keys


def strings_in_lists(obj):
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, str):
                yield x
            elif isinstance(x, dict) and isinstance(x.get("id"), str):
                yield x["id"]  # drop-reason records ({"id": ..., "ktasks": ...}) count as drops too
                yield from strings_in_lists(x)
            else:
                yield from strings_in_lists(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings_in_lists(v)


def call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def first_heavy_op(ref):
    """'matmul', 'conv' or None: the first matmul/conv-like op reached from workload(), following local helpers."""
    try:
        tree = ast.parse(ref)
    except SyntaxError:
        return None
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    seen = set()

    def visit(fn):
        if fn.name in seen:
            return None
        seen.add(fn.name)
        nodes = [n for n in ast.walk(fn) if isinstance(n, (ast.Call, ast.BinOp))]
        nodes.sort(key=lambda n: (n.lineno, n.col_offset))
        for n in nodes:
            if isinstance(n, ast.BinOp):
                if isinstance(n.op, ast.MatMult):
                    return "matmul"
                continue
            name = call_name(n)
            if name in MATMUL_CALLS:
                return "matmul"
            if name in CONV_CALLS or (name or "").startswith("conv"):
                return "conv"
            if name in funcs and funcs[name] is not fn:
                got = visit(funcs[name])
                if got:
                    return got
        return None

    return visit(funcs["workload"]) if "workload" in funcs else None


def row_text(row):
    return "\n".join(m.get("content", "") or "" for m in row.get("messages") or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--dev")
    ap.add_argument("--drops", nargs="+", action="extend", default=[])
    ap.add_argument("--verdicts", nargs="+", action="extend", default=[])
    ap.add_argument("--items", nargs="+", action="extend", default=[])
    ap.add_argument("--base")
    ap.add_argument("--allow-slow-matmul", action="store_true", help="report check 13 without failing (fallback file)")
    a = ap.parse_args()
    k = bench_constants()
    rows, bad_lines = read_jsonl(a.train)
    rid = lambda r, i: r.get("example_id") or f"row{i}"
    gen = lambda r: r.get("set") not in BASE_SETS

    # 1. parse, unique ids and tasks
    report("parse", bad_lines, f"{len(rows)} rows parse")
    ids = collections.Counter(r.get("example_id") for r in rows)
    report("example_id_unique", [x for x, c in ids.items() if c > 1 or not x], f"{len(ids)} unique example_ids")
    tasks = collections.Counter(r.get("task") for r in rows)
    report("task_unique", [str(x) for x, c in tasks.items() if c > 1 or not x], f"{len(tasks)} unique tasks",
           "duplicated or missing task (trainer holds out by task)")

    # 2. message structure
    bad_roles, bad_sys, bad_user1, bad_repair, bad_fence = [], [], [], [], []
    for i, r in enumerate(rows):
        msgs = r.get("messages") or []
        roles = [m.get("role") for m in msgs]
        if len(roles) < 3 or roles[:3] != ["system", "user", "assistant"] or len(roles) % 2 != 1 or \
                any(roles[j:j + 2] != ["user", "assistant"] for j in range(3, len(roles), 2)):
            bad_roles.append(rid(r, i))
            continue
        if msgs[0].get("content") != k["SYSTEM"]:
            bad_sys.append(rid(r, i))
        if ref_of(r, k) is None:
            bad_user1.append(rid(r, i))
        if any(not (m.get("content") or "").endswith(REPAIR_TAIL) for m in msgs[3::2]):
            bad_repair.append(rid(r, i))
        if any(unfence(m.get("content") or "") is None for m in msgs[2::2]):
            bad_fence.append(rid(r, i))
    report("roles", bad_roles, "system,user,assistant(+user,assistant)*")
    report("system_text", bad_sys, "equals bench.SYSTEM")
    report("first_user", bad_user1, "BEFORE_REF + fenced ref + AFTER_REF")
    report("repair_user_tail", bad_repair, "later user turns end with the repair line")
    report("assistant_one_fence", bad_fence, "every assistant turn is one ```python block")

    # 3. final target
    bad_final = []
    for i, r in enumerate(rows):
        msgs = r.get("messages") or []
        code = unfence(msgs[-1].get("content") or "") if msgs and msgs[-1].get("role") == "assistant" else None
        try:
            ok = code is not None and "def workload(" in code and "pl.pallas_call" in code and ast.parse(code)
        except SyntaxError:
            ok = False
        if not ok:
            bad_final.append(rid(r, i))
    report("final_target", bad_final, "defines workload(, calls pl.pallas_call, parses")

    # 4. thinking off
    report("thinking_false", [rid(r, i) for i, r in enumerate(rows) if r.get("thinking") is not False or
                              any(m.get("reasoning") or m.get("reasoning_content") or m.get("thinking") for m in r.get("messages") or [])],
           "thinking is False, no reasoning fields")

    # 5. teachers
    report("teacher", [rid(r, i) for i, r in enumerate(rows) if r.get("teacher") not in TEACHERS or
                       BANNED_TEACHER.search(f"{r.get('teacher')} {r.get('source')}")], f"teacher in {sorted(TEACHERS)}")

    # 6. drops
    if a.drops:
        drop = set()
        for p in a.drops:
            for s in strings_in_lists(json.load(open(p))):
                drop |= id_keys(s)
        hit = [rid(r, i) for i, r in enumerate(rows)
               if (id_keys(r.get("example_id")) | id_keys(r.get("problem_id")) | id_keys(r.get("task"))) & drop]
        report("drops_absent", hit, f"none of {len(drop)} drop keys present")

    # items (by task) for checks 7 and 10
    items = {}
    for p in a.items:
        for it in read_jsonl(p)[0]:
            items.setdefault(it.get("task"), it)

    chains = {}
    for i, r in enumerate(rows):
        if gen(r):
            ref = ref_of(r, k)
            chains[rid(r, i)] = norm_chain(ref) if ref else None

    # 7. dev overlap
    if a.dev:
        dev, _ = read_jsonl(a.dev)
        dev_tasks = {d.get("task") for d in dev}
        report("dev_task_absent", [rid(r, i) for i, r in enumerate(rows) if r.get("task") in dev_tasks], f"{len(dev_tasks)} dev tasks absent")
        dev_chain, unresolved = {}, []
        for d in dev:
            ref = d.get("ref_code") or d.get("reference") or (items.get(d.get("task")) or {}).get("ref_code")
            c = norm_chain(ref) if ref else None
            if c is None:
                unresolved.append(d.get("task"))
            else:
                dev_chain.setdefault(c, d.get("task"))
        report("dev_chain_resolved", unresolved, f"{len(dev)} dev items have a docstring chain (via --items)",
               "dev item has no reference to read a chain from; pass --items")
        report("dev_chain_absent", [f"{x}={c}~{dev_chain[c]}" for x, c in chains.items() if c in dev_chain],
               f"no generated row shares one of {len(dev_chain)} dev chains")

    # 8. exam chains
    report("gen_chain_present", [x for x, c in chains.items() if c is None], f"{len(chains)} generated rows have a docstring chain")
    report("exam_chain_absent", [f"{x}={c}" for x, c in chains.items() if c in EXAM_CHAINS], "no generated row has an exam chain")

    # 9. exam names and baselines
    names = sorted({re.sub(r"^\d+[a-z]?_", "", d) for d in os.listdir(LEVEL1) if os.path.isdir(os.path.join(LEVEL1, d))})
    name_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9])")
    ws = lambda s: " ".join(s.split())
    baselines = {ws(open(os.path.join(LEVEL1, d, "baseline.py")).read()): d for d in os.listdir(LEVEL1)
                 if os.path.isfile(os.path.join(LEVEL1, d, "baseline.py"))}
    hits, generic = [], []
    for i, r in enumerate(rows):
        found = set(name_re.findall(row_text(r) + "\n" + str(r.get("task")) + " " + str(r.get("example_id"))))
        # ponytail: one-word names (GEMM, RMSNorm) are ordinary kernel vocabulary, so they are reported, not failed
        hits += [f"{rid(r, i)}:{n}" for n in sorted(found) if "_" in n]
        generic += [f"{rid(r, i)}:{n}" for n in sorted(found) if "_" not in n]
    report("exam_name_absent", hits, f"none of {len(names)} level-1 task names (multi-word) in any row")
    report("exam_oneword_name", generic, "no one-word level-1 name in any row", "one-word level-1 names", soft=True)
    report("exam_baseline_absent", [f"{rid(r, i)}={baselines[ws(ref_of(r, k))]}" for i, r in enumerate(rows)
                                    if ref_of(r, k) and ws(ref_of(r, k)) in baselines], f"no reference equals one of {len(baselines)} baselines")

    # 10. e5_gen targets match verified items
    if a.verdicts and a.items:
        verd = {}
        for p in a.verdicts:
            for v in read_jsonl(p)[0]:
                verd[v.get("id")] = v
        missing, mismatch = [], []
        for i, r in enumerate(rows):
            if r.get("set") != "e5_gen":
                continue
            it = items.get(r.get("task"))
            v = verd.get(it.get("id")) if it else None
            if it is None or v is None:
                missing.append(rid(r, i))
                continue
            target = unfence(r["messages"][-1].get("content") or "")
            # the prompt's reference must be the one the oracle timed, or the verdict says nothing about this row
            why = "code" if target != it.get("code") else "ref" if ref_of(r, k) != it.get("ref_code") else v.get("verdict")
            if why in ("code", "ref") or v.get("verdict") != "pass" or (v.get("speed_vs_xla") or 0) < SPEED_BAR:
                mismatch.append(f"{rid(r, i)}:{why}/{v.get('speed_vs_xla')}")
        report("e5_gen_matched", missing, "every e5_gen row has an item and a verdict", "no item or verdict")
        report("e5_gen_target_verified", mismatch, "every e5_gen target equals a passing item at >= 0.7x")

    # 11. non-base rows are fast passes
    report("nonbase_fast_pass", [rid(r, i) for i, r in enumerate(rows) if gen(r) and (
        r.get("kind") in ("correct_slow", "repair_slow") or r.get("verdict") != "pass" or (r.get("speed_vs_xla") or 0) < SPEED_BAR)],
           "rows outside v6 base sets are pass and >= 0.7x")

    # 12. base rows byte-equal to base file
    if a.base:
        base = {b.get("example_id"): b.get("messages") for b in read_jsonl(a.base)[0]}
        report("base_rows_equal", [rid(r, i) for i, r in enumerate(rows) if not gen(r) and base.get(r.get("example_id")) != r.get("messages")],
               f"base-set rows equal {os.path.basename(a.base)}")

    # 13. below-bar matmul rows in base sets
    slow_mm = [rid(r, i) for i, r in enumerate(rows) if not gen(r) and (r.get("speed_vs_xla") or 0) < SPEED_BAR
               and ref_of(r, k) and first_heavy_op(ref_of(r, k)) == "matmul"]
    report("base_slow_matmul", slow_mm, "no base-set row below 0.7x has a matmul as its first heavy op",
           "base-set rows below 0.7x whose first heavy op is a matmul", soft=a.allow_slow_matmul)

    # 14. length
    lens = sorted(((len(row_text(r)), rid(r, i)) for i, r in enumerate(rows)), reverse=True)
    report("length", [x for n, x in lens if n > MAX_CHARS], f"max {lens[0][0] if lens else 0} chars <= {MAX_CHARS}", f"> {MAX_CHARS} chars")
    print("longest:", ", ".join(f"{x}={n}" for n, x in lens[:3]))

    table = collections.Counter((str(r.get("set")), str(r.get("kind")), str(r.get("teacher"))) for r in rows)
    print(f"\n{'set':<14}{'kind':<14}{'teacher':<16}rows")
    for (s, kd, t), n in sorted(table.items()):
        print(f"{s:<14}{kd:<14}{t:<16}{n}")
    print(f"{'total':<44}{len(rows)}")
    print("sha256", hashlib.sha256(open(a.train, "rb").read()).hexdigest(), a.train)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
