"""A6 rule 2, widened after Karthik's audit (Nov 12 02:50Z). A problem matches a k-task if ANY of these equals the k-task's chain read the same way:
  (a) whole chain minus linear + layout ops, collapsed (Aarav's test);
  (b) the same from the first dot_general on (the FLOP-dominant chain of a matmul-led problem; prologue ops on weights/inputs set aside);
  (c) (a) or (b) with 'log' removed on both sides (log_softmax read as softmax; unsure means out).
Also: matmul-led = first heavy op (dot_general / conv_general_dilated) is dot_general."""
import json, re, glob, collections
S = "/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/aarav_ds"
exam = json.load(open(f"{S}/dataset/problems/exam_signatures.json"))
LIN = {"add", "add_any", "sub", "mul", "div", "neg", "copy"}; LAYOUT = ("convert", "broadcast", "reshape", "transpose", "squeeze", "expand_dims", "copy")
def strip(ch, nolog=False):
    out = []
    for o in ch:
        if o in LIN or o.startswith(LAYOUT) or (nolog and o == "log"): continue
        if not out or out[-1] != o: out.append(o)
    return tuple(out)
def from_dot(ch): return ch[ch.index("dot_general"):] if "dot_general" in ch else ch
def views(ch): return {strip(ch), strip(from_dot(ch)), strip(ch, True), strip(from_dot(ch), True)}
KT = collections.defaultdict(set)
for t, s in exam.items():
    if re.match(r"^\d+k_", t) and "op_chain" in s:
        for v in views(s["op_chain"]): KT[v].add(t)
def match(ch): return sorted({t for v in views(ch) for t in KT.get(v, ())})
def mm_led(ch):
    h = [o for o in ch if o in ("dot_general", "conv_general_dilated")]
    return bool(h) and h[0] == "dot_general"
