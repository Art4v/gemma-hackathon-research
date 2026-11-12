"""Fidelity checks of a training jsonl against bench.py's conversation()/extract_code (parsed as text, nothing from rows executed)."""
import json, re, sys, collections
SYSTEM = ("You are a code model that writes TPU kernels in JAX Pallas. Given a reference JAX function, reply with one Python code block only. Define "
          "workload(*inputs) with the same signature and outputs as the reference, reading shapes from inputs at runtime. Run the main computation inside "
          "pl.pallas_call, using the imports from jax.experimental import pallas as pl and from jax.experimental.pallas import tpu as pltpu. The kernel runs on a "
          "TPU v6e at the reference's full size. Use the largest blocks fitting in memory to be as fast as the jax.jit reference.")
def extract_code(t):
    f = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", t or "", re.S | re.I); return f[-1] if f else ""
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
c = collections.Counter(); ex = collections.defaultdict(list)
maxans = maxprompt5 = maxtot = 0
for r in rows:
    m = r["messages"]; c[f"len{len(m)}"] += 1; c[f"thinking={r.get('thinking','MISSING')}"] += 1
    c["roles=" + ",".join(x["role"] for x in m)] += 1
    if m[0]["content"] != SYSTEM: c["system_differs"] += 1; ex["system_differs"].append(r.get("example_id"))
    u = m[1]["content"]
    if not (u.startswith("Here is the reference JAX function:\n```python\n") and u.endswith("\n```\nPlease convert this function into a Pallas TPU kernel.")):
        c["user0_format_differs"] += 1; ex["user0"].append(r.get("example_id"))
    if any(x.get("reasoning", "").strip() for x in m): c["has_reasoning"] += 1
    last = m[-1]["content"]
    if not re.fullmatch(r"```python\n.*\n```", last, re.S): c["final_not_bare_block"] += 1; ex["final"].append((r.get("example_id"), repr(last[:40]), repr(last[-40:])))
    ln = sum(len(x["content"]) for x in m[:-1])
    maxans = max(maxans, len(last)); maxtot = max(maxtot, ln + len(last))
    if len(m) == 5:
        maxprompt5 = max(maxprompt5, ln)
        a1 = m[2]["content"]
        code = extract_code(a1)
        bench_hist = f"```python\n{code}\n```"
        if a1 == bench_hist: c["a1_eq_bench_wrapper"] += 1
        else:
            c["a1_ne_bench_wrapper"] += 1
            if a1 == f"```python\n{code}```": c["a1_single_newline_close (bench gives \\n\\n)"] += 1
            ex["a1"].append((r.get("example_id"), repr(a1[-30:]), repr(bench_hist[-30:])))
        u2 = m[3]["content"]
        if not u2.endswith("\nReply with the corrected Python code block."): c["u2_suffix_differs"] += 1; ex["u2"].append(r.get("example_id"))
for k, v in sorted(c.items()): print(f"{k}: {v}")
print("rows", len(rows), "max answer chars", maxans, "~tok", round(maxans/3.2), "| max 5-msg prompt chars", maxprompt5, "~tok", round(maxprompt5/3.2), "| max total chars", maxtot, "~tok", round(maxtot/3.2))
for k, v in ex.items(): print(k, len(v), v[:3])
