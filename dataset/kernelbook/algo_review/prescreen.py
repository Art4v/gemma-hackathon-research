"""Stage 1 of the algorithm-level contamination review (rule 3): flag KernelBook/prod problems whose name or op pattern matches the
algorithm family of a JAXBench task. Over-inclusive on purpose; stage 2 (a careful read of each flagged reference) decides.
  python prescreen.py PROBLEMS.jsonl OUT.jsonl"""
import ast, collections, json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("batch") in ("kbk", "kbkc", "kbkd", "prod") and r.get("status") == "ok"]

NAME = {  # family -> (regex on module/task name, JAXBench tasks it could shadow)
    "attention": (r"att|mha|transformer|selfatt|crossatt|sdpa|flash|softmaxattention|scaled_dot", "1p 2p 3p 4p 5p 51p-54p 65p"),
    "paged_kv": (r"paged|kvcache|kv_cache|ragged.*att|rpa", "6p 7p 60p 61p 58p"),
    "cross_entropy": (r"crossentropy|cross_entropy|celoss|nll|logsoftmax.*loss|softmaxloss|labelsmooth|xent", "13p 64p"),
    "rmsnorm": (r"rmsnorm|rms_norm|rmslayer", "12p"),
    "glu_mlp": (r"swiglu|geglu|glu|gated.*mlp|gatedffn|feedforward|ffn|mlp", "9p"),
    "moe": (r"moe|expert|router|gating|switch", "10p 11p 59p"),
    "grouped_mm": (r"ragged|grouped.*(mm|matmul)|gmm|segment.*matmul|blocksparse", "11p 14p 59p 4p"),
    "sinkhorn": (r"sinkhorn|optimal.?transport|\bot\b", "56p"),
    "ssm_linear_rnn": (r"mamba|ssm|s4|retention|retnet|delta|gla|linear.?att|rwkv|scan", "15p 16p 62p 63p"),
    "triangle": (r"triangle|trimul|pairformer|evoformer", "17p"),
    "gemm": (r"^linear$|gemm|matmul|dense$|fullyconnected|fc$", "8p"),
    "quant": (r"quant|int8|int4|fp8", "65p"),
    "conv1d_gate": (r"causalconv|causal_conv|shortconv|conv1d.*gate", "62p"),
}


def ops(r):
    sig = r.get("signature") or {}
    ms = sig.get("op_multiset"); ch = sig.get("op_chain")
    ms = ast.literal_eval(ms) if isinstance(ms, str) else (ms or {})
    ch = ast.literal_eval(ch) if isinstance(ch, str) else (ch or [])
    return ms, ch


def op_flags(ms, ch, code):
    f = []
    c = " ".join(ch)
    n_dot = ms.get("dot_general", 0)
    softmaxish = ("exp" in ms and ("reduce_sum" in ms) and ("div" in ms or "mul" in ms))
    if n_dot >= 2 and softmaxish and re.search(r"dot_general.*(reduce_max|exp).*exp.*reduce_sum.*dot_general", c):
        f.append("attention")                     # softmax between two matmuls
    if "log" in ms and "exp" in ms and "reduce_sum" in ms and (re.search(r"take_along_axis|gather|one_hot|argmax", code) or "gather" in ms):
        f.append("cross_entropy")                 # log-softmax + target gather
    if "rsqrt" in ms and re.search(r"mean|reduce_sum", c) and not re.search(r"\bsub\b.*reduce_sum.*rsqrt", c):
        f.append("rmsnorm_like")                  # rsqrt of a mean square without centering (rough)
    if n_dot >= 3 and re.search(r"logistic|silu|sigmoid|gelu", c) and ms.get("mul", 0) >= 1:
        f.append("glu_mlp_like")
    if re.search(r"top_k|argsort|sort", c) and n_dot >= 1:
        f.append("moe_like")
    if re.search(r"while|scan|fori_loop", c) and "exp" in ms:
        f.append("iterative_exp (sinkhorn/ssm?)")
    if re.search(r"cumsum|cumprod|associative_scan|scan", c) and n_dot >= 1:
        f.append("scan+matmul (ssm/retention?)")
    if n_dot == 1 and set(ms) <= {"dot_general", "add", "convert_element_type", "reshape", "transpose", "broadcast_in_dim", "squeeze", "expand_dims"}:
        f.append("plain_gemm")
    return f


out = []
for r in rows:
    name = (r["source_ref"].get("module_name") or r["task_name"]).lower() if r.get("source_ref") else r["task_name"].lower()
    ms, ch = ops(r)
    nf = [fam for fam, (rx, _) in NAME.items() if re.search(rx, name)]
    of = op_flags(ms, ch, r["ref_code"])
    if nf or of:
        out.append({"problem_id": r["problem_id"], "task_name": r["task_name"], "batch": r["batch"], "module": name,
                    "name_flags": nf, "op_flags": of, "n_ops": sum(ms.values()) if ms else None})
json.dump(out, open(sys.argv[2], "w"), indent=0)
c = collections.Counter(f for o in out for f in o["name_flags"] + o["op_flags"])
print(f"{len(rows)} problems screened; {len(out)} flagged for stage 2")
for k, v in c.most_common(): print(f"  {k}: {v}")
