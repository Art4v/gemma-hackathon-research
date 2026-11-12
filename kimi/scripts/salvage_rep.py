"""Re-parse repair-round rows that failed only because Kimi answered "code only" with a bare module (no JSON, no fence): the saved
text is the module, as render_chat.unfence accepts. Rewrites ok/code/parsed_from in place; nothing is regenerated.
  python3 salvage_rep.py out/rep2"""
import glob, json, pathlib, sys
d = pathlib.Path(sys.argv[1]); n = 0
for f in glob.glob(str(d / "k31_rep*_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    for r in rows:
        t = r.get("text_if_unparsed") or ""
        if not r["ok"] and "def workload" in t and "```" not in t and not t.lstrip().startswith("{"):
            r.update(ok=True, code=t.strip("\n") + "\n", plan="", parsed_from="raw_module", error=None); n += 1
    pathlib.Path(f).write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"salvaged {n} bare-module answers")
