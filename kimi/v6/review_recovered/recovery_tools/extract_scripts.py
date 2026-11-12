"""Extract the A6 review's scripts verbatim from the workflow transcripts (tool_use inputs).

Read-only on the transcripts. Writes into ../scripts/ and ../regen/ (replay commands).
Run: python3 -I extract_scripts.py
"""
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
WF = ("/Users/hrudaynara/.claude/projects/-Users-hrudaynara-Projects-gemma-hackathon-research/"
      "2c5e8742-e04c-4af6-a34a-5943a12b6e62/subagents/workflows/wf_49512077-aae")
ADV = WF + "/agent-a155bcc295ec0cc51.jsonl"   # contamination-adversary (the A6 review)
AUD = WF + "/agent-a42ffff3745e37008.jsonl"   # filter-audit (built exclusions.jsonl)
OLD_ADV_DIR = ("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/"
               "2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv")
NEW_ADV_DIR = OUT + "/regen/review/adv"


def tool_use(path, lineno):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i == lineno:
                d = json.loads(line)
                uses = [p for p in d["message"]["content"] if p.get("type") == "tool_use"]
                assert len(uses) == 1, (path, lineno, len(uses))
                assert uses[0]["name"] == "Bash", uses[0]["name"]
                return uses[0]["id"], uses[0]["input"]["command"]
    raise SystemExit(f"line {lineno} not found in {path}")


def heredocs(cmd):
    """Bodies of every heredoc in a shell command, in order, exactly as `cat` would write them."""
    out, body, inside = [], None, False
    for ln in cmd.split("\n"):
        if inside:
            if ln == "EOF":
                out.append("".join(x + "\n" for x in body)); inside = False
            else:
                body.append(ln)
        elif re.search(r"<<\s*'?EOF'?", ln):
            inside, body = True, []
    assert not inside, "unterminated heredoc"
    return out


def dq_snippets(cmd):
    """Code of every `python3 -c "..."` in a shell command, with bash double-quote unescaping."""
    res, i = [], 0
    while True:
        j = cmd.find('python3 -c "', i)
        if j < 0:
            return res
        k = j + len('python3 -c "'); buf = []; expands = False
        while cmd[k] != '"':
            if cmd[k] == "\\" and cmd[k + 1] in '"\\$`':
                buf.append(cmd[k + 1]); k += 2
            else:
                expands |= cmd[k] in "$`"
                buf.append(cmd[k]); k += 1
        res.append(None if expands else "".join(buf)); i = k + 1  # None: shell expands it, keep only the .sh


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def w(rel, text):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return rel


prov = {"transcripts": {"adversary": [ADV, sha(ADV)], "filter_audit": [AUD, sha(AUD)]}, "files": {}}

# 1. filter-audit: labels.py (cat > at L361, cat >> at L365 and L368) and build_table.py (L372)
lab = []
for n in (361, 365, 368):
    tid, cmd = tool_use(AUD, n)
    w(f"scripts/filter_audit_L{n}.sh", cmd)
    hd = heredocs(cmd); assert len(hd) == 1
    lab.append(hd[0]); prov["files"][f"scripts/filter_audit_L{n}.sh"] = [AUD, n, tid]
assert "cat > $S/review/labels.py" in tool_use(AUD, 361)[1]
assert all("cat >> $S/review/labels.py" in tool_use(AUD, n)[1] for n in (365, 368))
w("scripts/labels.py", "".join(lab)); prov["files"]["scripts/labels.py"] = [AUD, [361, 365, 368], "heredoc bodies concatenated (cat >, cat >>, cat >>)"]
tid, cmd = tool_use(AUD, 372)
w("scripts/filter_audit_L372.sh", cmd)
hd = heredocs(cmd); assert len(hd) == 1
w("scripts/build_table.py", hd[0]); prov["files"]["scripts/build_table.py"] = [AUD, 372, tid]

# 2. adversary: every Bash step that writes or reads the three files
steps = {286: "verdicts.py written + run", 290: "print safe list (pre-patch)", 310: "pool_relu_stacks.json written",
         314: "patch verdicts.py + rerun + check", 348: "pool_contamination.json written", 352: "final_safe_list.txt written",
         356: "group counts + ls sizes", 360: "print final_safe_list.txt", 364: "print keep-excluded reasons"}
with open(ADV, encoding="utf-8") as f:   # every other Bash step from L286 to the StructuredOutput (analysis only)
    for i, line in enumerate(f, 1):
        if 286 <= i <= 366 and i not in steps:
            content = (json.loads(line).get("message") or {}).get("content")
            if isinstance(content, list) and any(p.get("type") == "tool_use" and p.get("name") == "Bash" for p in content):
                steps[i] = "analysis step (does not write any of the three files)"
steps = dict(sorted(steps.items()))
for n, what in steps.items():
    tid, cmd = tool_use(ADV, n)
    w(f"scripts/adv_L{n}.sh", cmd); prov["files"][f"scripts/adv_L{n}.sh"] = [ADV, n, tid, what]
    for k, body in enumerate(heredocs(cmd)):
        rel = w(f"scripts/adv_L{n}_heredoc{k + 1}.py", body); prov["files"][rel] = [ADV, n, tid, f"heredoc #{k + 1}"]
    for k, code in enumerate(dq_snippets(cmd)):
        if code is None:
            prov["files"][f"scripts/adv_L{n}.sh"].append(f'python3 -c #{k + 1} uses shell expansion; see the .sh only'); continue
        rel = w(f"scripts/adv_L{n}_python_c{k + 1}.py", code); prov["files"][rel] = [ADV, n, tid, f'python3 -c "..." #{k + 1}']
    # replay command for the new run directory: same command, new cwd, python3 -> python3 -I
    if n in (286, 290, 314, 352, 356, 360, 364):
        assert cmd.count(OLD_ADV_DIR) >= 1
        rep = cmd.replace(OLD_ADV_DIR, NEW_ADV_DIR)
        rep = re.sub(r"\bpython3 ", "python3 -I ", rep)
        assert "/private/tmp" not in rep
        w(f"regen/replay_L{n}.sh", rep)
w("scripts/verdicts.py", heredocs(tool_use(ADV, 286)[1])[0])
prov["files"]["scripts/verdicts.py"] = [ADV, 286, "heredoc body as first written (before the L314 patch)"]
w("checks/provenance.json", json.dumps(prov, indent=1) + "\n")
print("\n".join(sorted(prov["files"])))
