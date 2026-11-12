"""After the v7 Kimi fixes finish: oracle step 1 on CPU (tpu_oracle.py --static-only, the same oracle files, sha c9a07d9298a1, JAX 0.10.2), then TPU
items for every static_ok fix (ids kimiKN1/<pid>, attempt 2 of the scaled problem), split N ways for the fleet.
  python3 fix_step1.py N      ->  out/fix1/cpu_scored.jsonl, ~/Projects/gemma-work/tpu/fleet/kimi-kn1-s<K>.items.jsonl"""
import glob, json, os, pathlib, subprocess, sys, tempfile, time, zlib
V7 = pathlib.Path(__file__).resolve().parent
D = V7 / "out/fix1"; F = pathlib.Path.home() / "Projects/gemma-work/tpu/fleet"
PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"
ORACLE = "/Users/hrudaynara/Projects/gemma-hackathon-research/oracle/tpu_oracle.py"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
while not all(any(m in open(f).read() for m in ("FIX GEN DONE", "CAP REACHED")) for f in glob.glob(str(D / "gen_*.log"))):
    time.sleep(60)
items = {}
for f in ("kn_items_existing.jsonl", "kn_items_existing_8.jsonl", "kn_items_existing_p.jsonl"):
    for l in open(V7 / f):
        x = json.loads(l); items[x["id"]] = x
fixes = {}
for f in glob.glob(str(D / "k31_fix1_*.jsonl")):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok") and d.get("code") and d["id"] in items:
            fixes[d["id"]] = d
print(len(fixes), "fixes with code", flush=True)
rows = []
ids = sorted(fixes)
for i in range(0, len(ids), 40):
    batch = ids[i:i + 40]
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        (t / "pack.json").write_text(json.dumps({"refs": {items[x]["task"]: items[x]["ref_code"] for x in batch},
                                                 "items": [{"id": x, "task": items[x]["task"], "code": fixes[x]["code"]} for x in batch]}))
        subprocess.run([PY, ORACLE, str(t / "pack.json"), str(t / "static.jsonl"), "--static-only"], env={**os.environ, "JAX_PLATFORMS": "cpu"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        st = {}
        if (t / "static.jsonl").exists():
            for l in (t / "static.jsonl").read_text().splitlines():
                if '"meta": true' not in l:
                    x = json.loads(l); st[x["id"]] = x
    for x in batch:
        s = st.get(x, {"verdict": "harness_no_row", "feedback": "static check produced no row"})
        rows.append({"id": x, "task": items[x]["task"], "static": s.get("verdict"), "static_feedback": (s.get("feedback") or "")[:400],
                     "oracle_sha": s.get("oracle_sha")})
    print(f"step 1: {len(rows)}/{len(ids)} scored, static_ok so far {sum(r['static'] == 'static_ok' for r in rows)}", flush=True)
(D / "cpu_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
ok = [r for r in rows if r["static"] == "static_ok"]
groups = {k: [] for k in range(1, N + 1)}
for r in ok:
    x = r["id"]; pid = x.split("/", 1)[1]
    groups[zlib.crc32(r["task"].encode()) % N + 1].append({"id": "kimiKN1/" + pid, "task": r["task"], "ref_code": items[x]["ref_code"],
                                                          "code": fixes[x]["code"]})
for k, g in groups.items():
    (F / f"kimi-kn1-s{k}.items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in g))
print(f"STEP1 DONE: {len(ok)} of {len(rows)} fixes pass step 1 -> TPU items per VM", {k: len(g) for k, g in groups.items()}, flush=True)
