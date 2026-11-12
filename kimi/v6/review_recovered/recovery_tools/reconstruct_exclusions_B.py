"""Reconstruct the class-B rows of the lost review/exclusions.jsonl (built by filter-audit L372 build_table.py).

Field sources (for each class-B problem):
  class, b_strength, closest_jaxbench_task, family, justification  <- scripts/labels.py (verbatim, filter-audit L361/365/368)
  in_kimi_pool_now  <- B_list.txt as printed by the adversary (L73 + L76), which shows a P/- flag for every class-B row
  source            <- build_table.py's first-add order (steps 1-9), with each lost input replaced by its printed copy:
                         step 3 algo reviews: labels.py section 'algo review kbk + prod (88)'; kbkc/kbkd lists printed at filter-audit L250
                         step 4 score filters: karthik rows printed with batch at filter-audit L266; full ordered score list at L278/L282/L286
                         steps 5/6: the on-disk problems.jsonl files build_table read (mtimes predate the build)
                         step 7/8: labels.py sections; prod batch membership from prod_selection.jsonl (corroboration)
`rule` is NOT reconstructed (it needs the lost algo_review_*.jsonl verdict strings and exclude_reason texts); the field is omitted.
A/C rows are NOT written (their in_kimi_pool_now and rule cannot be recovered exactly); verdicts.py reads only class B.

Every derived field is checked against independent transcript printouts; any mismatch aborts.
Run: python3 -I reconstruct_exclusions_B.py
"""
import ast, collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
WF = ("/Users/hrudaynara/.claude/projects/-Users-hrudaynara-Projects-gemma-hackathon-research/"
      "2c5e8742-e04c-4af6-a34a-5943a12b6e62/subagents/workflows/wf_49512077-aae")
ADV = WF + "/agent-a155bcc295ec0cc51.jsonl"
AUD = WF + "/agent-a42ffff3745e37008.jsonl"
DS_CLAUDE = "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/problems/problems.jsonl"
DS_GEN = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/problems/problems.jsonl"
BENCH = "/Users/hrudaynara/Projects/gemma-hackathon-research/benchmarks/jaxbench/level1"
PROD_SEL = "/Users/hrudaynara/Projects/gemma-work/tpu/ops1/repo/dataset/kernelbook/prod_selection.jsonl"
PROD_EXC = "/Users/hrudaynara/Projects/gemma-work/tpu/ops1/repo/dataset/kernelbook/prod_excluded_by_algorithm.jsonl"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
def check(cond, msg):
    say(("PASS " if cond else "FAIL ") + msg)
    if not cond: raise SystemExit("aborting: " + msg)


def result(path, lineno):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i == lineno:
                d = json.loads(line)
                rs = [p for p in d["message"]["content"] if p.get("type") == "tool_result"]
                assert len(rs) == 1
                c = rs[0]["content"]
                return "".join(x.get("text", "") for x in c) if isinstance(c, list) else c


def counter(text):
    return dict(ast.literal_eval(text.strip()[len("Counter("):-1]))


# ---- labels.py, executed section by section with a logging wrapper around a() ----
src = open(os.path.join(OUT, "scripts/labels.py"), encoding="utf-8").read()
parts = re.split(r"(?m)^(# -{8,} .*)$", src)
ns = {"__name__": "labels"}
exec(compile(parts[0], "labels.py:header", "exec"), ns)
orig_a = ns["a"]; section = {"name": "header"}; calls = collections.defaultdict(list)
def logged_a(pid, *rest):
    calls[section["name"]].append(pid); orig_a(pid, *rest)
ns["a"] = logged_a
for k in range(1, len(parts), 2):
    section["name"] = parts[k]
    exec(compile(parts[k + 1], "labels.py:" + parts[k], "exec"), ns)
L = ns["L"]
sec = {name: list(dict.fromkeys(v)) for name, v in calls.items()}
def sec_named(prefix):
    m = [n for n in sec if n.startswith("# ---------------- " + prefix)]
    assert len(m) == 1, (prefix, m); return sec[m[0]]
S_KBK = sec_named("algo review kbk + prod (88)")
S_KBKC = sec_named("algo review kbkc (25)")
S_KBKD = sec_named("algo review kbkd (28)")
S_PRODC = sec_named("prod candidates dropped before authoring")
S_V4 = sec_named("v4 rule-3 families")
B = sorted(p for p, v in L.items() if v[0] == "B")
say("labels.py: entries", len(L), "| class counts", dict(collections.Counter(v[0] for v in L.values())))
check(len(B) == 273, f"labels.py has 273 class-B problems (got {len(B)})")
check(len(S_KBK) == 88 and len(S_KBKC) == 25 and len(S_KBKD) == 28, "algo-review sections hold 88 / 25 / 28 problems")
cc = lambda s: dict(sorted(collections.Counter(L[p][0] for p in s).items()))
check(cc(S_KBK) == {"A": 17, "B": 71} and cc(S_KBKC) == {"A": 4, "B": 21} and cc(S_KBKD) == {"A": 9, "B": 19},
      "section class mix equals filter-audit L377 layer stats (kbk+prod A17/B71, kbkc A4/B21, kbkd A9/B19)")

# ---- E3: algo review kbkc / kbkd excluded lists printed at filter-audit L250 ----
t = result(AUD, 250); cur = None; algo = collections.defaultdict(list)
for ln in t.split("\n"):
    m = re.match(r"^######## (kbkc|kbkd)$", ln)
    if m: cur = m.group(1); continue
    m = re.match(r"^(\S+) \[(\w+)\] (\S+): ", ln)
    if m and cur: algo[cur].append(m.group(1))
check(len(algo["kbkc"]) == 25 and len(algo["kbkd"]) == 28, "L250 prints 25 kbkc + 28 kbkd algo-review excludes")
check(set(algo["kbkc"]) == set(S_KBKC) and set(algo["kbkd"]) == set(S_KBKD), "L250 lists == labels.py kbkc/kbkd sections")

# ---- prod batch (corroboration from the on-disk prod_selection.jsonl) ----
prod_ids = {r["src"] + "/" + r["stem"] for r in map(json.loads, open(PROD_SEL))}
PROD_IN_KBK = [p for p in S_KBK if p in prod_ids]
check(len(PROD_IN_KBK) == 11, f"11 of the kbk+prod section are prod-batch per prod_selection.jsonl (got {len(PROD_IN_KBK)}; L258: prod contamination:algorithm 11)")

# ---- E4/E5: score-filter rows ----
t = result(AUD, 266)
karthik = [(m.group(1), m.group(2)) for m in re.finditer(r"(?m)^(kbk|kbkd|kbkc|prod) (\S+) contamination:", t)]
hdr = []
for n in (278, 282, 286):
    hdr += re.findall(r"(?m)^## (\S+) \| ", result(AUD, n))
score_ids = list(dict.fromkeys(hdr))
check(len(karthik) == 86, f"L266 prints 86 karthik/kbk score-filter rows (53 extra + 8 opchain + 7 neardup kbk, 16 + 2 kbkd per L258) (got {len(karthik)})")
check(len(score_ids) == 122, f"L278/L282/L286 print all 122 score_ids.txt entries (L274 wc) (got {len(score_ids)})")
check(score_ids[:86] == [p for _, p in karthik], "first 86 score ids == L266 karthik rows, same order")
aarav = [("kbkc", p) for p in score_ids[86:]]
check(len(aarav) == 36, "remaining 36 == aarav/kbk-c kbkc rows (L262: extra 28 + neardup 4 + opchain 4)")

# ---- steps 5/6: persistent problems.jsonl files ----
for f in (DS_CLAUDE, DS_GEN):
    say("mtime", f, os.path.getmtime(f))
v1 = [r["problem_id"] for r in map(json.loads, open(DS_CLAUDE)) if r["status"] == "excluded" and (r.get("exclude_reason") or "").startswith("contamination")]
gen = [r["problem_id"] for r in map(json.loads, open(DS_GEN)) if r["status"] == "excluded" and (r.get("exclude_reason") or "").startswith("contamination")]

# ---- step 1: KernelBench L2 exam stems ----
ktask = {}
for d in sorted(os.listdir(BENCH)):
    if re.match(r"\d+k_", d):
        m = re.search(r"'name':\s*'([^']+)'", open(f"{BENCH}/{d}/baseline.py").read())
        ktask["kb/l2/" + m.group(1)] = d

# ---- simulate build_table.py's add() order; first add fixes `source` ----
first = collections.OrderedDict()
def add(pid, source):
    first.setdefault(pid, source)
for pid in ktask: add(pid, "kernelbench_l2 (never ported)")
for pid in [p for p in L if p.startswith("kb/l1/") and not p.startswith("kb/l1/50_")]: add(pid, "kernelbench_l1 (never ported)")
for pid in S_KBK: add(pid, "prod (github/authored-from-paper)" if pid in prod_ids else "kernelbook (kbk)")
for pid in algo["kbkc"]: add(pid, "kernelbook (kbkc)")
for pid in algo["kbkd"]: add(pid, "kernelbook (kbkd)")
for b, pid in karthik + aarav: add(pid, f"kernelbook ({b})")
for pid in v1: add(pid, "kernelbench_l2 port (dataset-v1)")
for pid in gen: add(pid, "gemini-era pool (gemma-dataset-gen)")
for pid in S_PRODC: add(pid, "prod candidate (never authored)")
for pid in S_V4: add(pid, "authored/scraped (dataset-v1) or m1 child")
prod_exc = {"prod/" + json.loads(l)["stem"] for l in open(PROD_EXC)}
check(set(S_PRODC) == prod_exc, "labels.py prod-candidate section == on-disk prod_excluded_by_algorithm.jsonl (25 stems)")
missing = [p for p in B if p not in first]
check(not missing, f"every class-B problem gets a source from steps 1-8 (missing: {missing[:5]})")

# ---- E1: B_list.txt (adversary L72 writes it; L73 prints lines 1-120, L76 prints 120-306) ----
t73 = result(ADV, 73).split("\n"); t76 = result(ADV, 76).split("\n")
check(t73[0].strip().startswith("306 "), "B_list.txt had 306 lines")
blist = [x for x in t73[1:] if x != ""] + [x for x in t76 if x != ""][1:]
check(len(blist) == 306, f"recovered all 306 B_list.txt lines (got {len(blist)})")
bl, task = {}, None
for ln in blist:
    m = re.match(r"^== (\S+) (\d+)$", ln)
    if m: task = m.group(1); continue
    m = re.match(r"^   ([nf]) ([P-]) (\S+) \| (.*)$", ln)
    assert m, ln
    bl[m.group(3)] = dict(task=task, st=m.group(1), pool=m.group(2) == "P", j150=m.group(4))
check(sorted(bl) == B, "B_list.txt problem set == labels.py class-B set (273)")
check(all(bl[p]["task"] == L[p][2] for p in B), "closest_jaxbench_task matches B_list.txt for all 273")
check(all(bl[p]["st"] == L[p][1][0] for p in B), "b_strength matches B_list.txt for all 273")
check(all(bl[p]["j150"] == L[p][4][:150].rstrip("\n") for p in B), "justification[:150] matches B_list.txt for all 273")

# ---- cross-checks on source and pool flags ----
t = result(ADV, 283)
e2 = {}
for ln in t.strip().split("\n"):
    m = re.match(r"^(.{12}) (\S+) (True|False)$", ln)
    if m: e2[m.group(2)] = (m.group(1), m.group(3) == "True")
mine = {p: (first[p][:12], bl[p]["pool"]) for p in B if first[p].startswith("gemini") or first[p].startswith("kernelbench_l2 port") or "kbkd" in first[p]}
check(len(e2) == 52 and e2 == mine, "per-problem source + pool flag == adversary L283 for all 52 gemini/kb-l2-port/kbkd rows")
c57 = counter(result(ADV, 57).strip().split("\n")[-1])
mine57 = dict(collections.Counter(first[p] for p in B))
check(c57 == mine57, f"source histogram over 273 == adversary L57 {c57}")
t381 = result(AUD, 381).split("\n")
i = [k for k, x in enumerate(t381) if x.startswith("B NOT in Kimi pool:")][0]
c381 = counter(t381[i + 1])
mine381 = dict(collections.Counter(first[p] for p in B if not bl[p]["pool"]))
check(c381 == mine381 and t381[i].startswith("B NOT in Kimi pool: 186 "), f"not-in-pool source histogram (186) == filter-audit L381 {c381}")
fam381 = counter(t381[i + 2].split(":", 1)[1].split(" ", 2)[2])
mine_fam = dict(collections.Counter(L[p][3] for p in B if bl[p]["pool"]))
check(fam381 == mine_fam and sum(mine_fam.values()) == 87, f"in-pool family histogram (87) == filter-audit L381 {fam381}")
check(set(PROD_IN_KBK) <= set(B), "all 11 prod-batch problems are class B")

rows = []
for p in B:
    cls, st, tsk, fam, why = L[p]
    rows.append({"problem_id": p, "source": first[p], "closest_jaxbench_task": tsk, "class": cls, "b_strength": st or None,
                 "family": fam, "in_kimi_pool_now": bl[p]["pool"], "justification": why})
txt = "".join(json.dumps(r) + "\n" for r in rows)
for rel in ("exclusions_classB_reconstructed.jsonl", "regen/review/exclusions.jsonl"):
    with open(os.path.join(OUT, rel), "w", encoding="utf-8", newline="") as f:
        f.write(txt)
say("wrote", len(rows), "class-B rows")
with open(os.path.join(OUT, "checks/reconstruct_exclusions_B.log"), "w") as f:
    f.write("\n".join(log) + "\n")
