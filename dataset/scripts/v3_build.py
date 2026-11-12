"""Dataset v3: assemble what exists under dataset/v3/ into training rows and a manifest. Safe to run on a partial build, so a first slice can ship.

  python dataset/scripts/v3_build.py [--python PY]

Writes dataset/v3/final/{train.jsonl, val.jsonl, verification.jsonl, MANIFEST.json}.

Rows are in the training format: `messages` from control/bench.py's own conversation(ref, history, None), then the assistant target as a fenced
block with "reasoning": "", and "thinking": false, task, example_id, kind (single, repair, repair_injected, generated, shape_variant, correct_slow)
and source. dataset-v1's 26 val rows are val, unchanged. Nothing new is derived from a val family, so no val problem leaks into train.
"""
from __future__ import annotations

import argparse, collections, importlib.util, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import FINAL, REPO, now, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402
from v3_common import ORACLE_SHA, V3, checked_in_batches  # noqa: E402

CHARS_PER_TOKEN = 2.44  # dataset-v1: median target 2,985 chars, 1,221 tokens by Hruday's tokenizer run on issue 5
MAX_TARGET_TOKENS = 8192
TEACHER = {"repair_injected": "injected", "generated": "generator"}  # who wrote the target or the failure; everything else is a Claude-written kernel


def bench():
    spec = importlib.util.spec_from_file_location("bench", REPO / "control" / "bench.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def history_of(turns: list) -> list:
    """[{code, feedback}] for every failed attempt in a row's turns: exactly what bench.conversation takes."""
    return [{"code": turns[i]["code"], "feedback": turns[i + 1]["feedback"]} for i in range(1, len(turns) - 1, 2)]


def training_row(e: dict, kind: str, conversation) -> dict:
    target = e["turns"][-1]["code"]
    msgs = conversation(e["turns"][0]["ref_code"], history_of(e["turns"]), None)
    msgs.append({"role": "assistant", "content": f"```python\n{target}\n```", "reasoning": ""})
    return {"messages": msgs, "thinking": False, "task": e["task_name"], "example_id": e["example_id"], "kind": kind, "source": e["source"], "teacher": TEACHER.get(kind, "claude")}


PARTS = (("repairs.jsonl", "repair_injected"), ("generated.jsonl", "generated"), ("shape_variants.jsonl", "shape_variant"), ("correct_slow.jsonl", "correct_slow"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    conv = bench().conversation
    v1_train, v1_val = read_jsonl(FINAL / "train.jsonl"), read_jsonl(FINAL / "val.jsonl")
    val_families = {e["family_id"] for e in v1_val}
    rows, verif, present = [], [], []
    for e in v1_train:
        rows.append(training_row(e, e["kind"], conv))
        verif.append({"example_id": e["example_id"], "kind": e["kind"], "verification": "tpu", "final_verdict": e["final_verdict"], "oracle_job_ids": e["oracle"]["job_ids"]})
    # the static verdict of every distinct target that has none yet: dataset-v1 targets the repair pairs reuse
    extra_rows = {}
    for fn, kind in PARTS:
        if (V3 / fn).exists():
            extra_rows[kind] = read_jsonl(V3 / fn)
            present.append(kind)
    need, refs = {}, {}
    for e in v1_train + extra_rows.get("repair_injected", []) + extra_rows.get("correct_slow", []):
        t = e["turns"][-1]
        need.setdefault(sha256(t["code"]), {"id": sha256(t["code"]), "task": e["task_name"], "code": t["code"]})
        refs[e["task_name"]] = e["turns"][0]["ref_code"]
    print(f"static check of {len(need)} distinct targets", flush=True)
    static = checked_in_batches(list(need.values()), refs, a.python, log=lambda m: print(m, flush=True))
    bad = [k for k, r in static.items() if r.get("verdict") != "static_ok"]
    if bad:
        print(f"{len(bad)} targets are not static_ok: {[need[k]['task'] for k in bad[:5]]}")
    for kind, part in extra_rows.items():
        for e in part:
            if e["family_id"] in val_families or e["split"] != "train":
                sys.exit(f"{e['example_id']}: comes from a val family; v3 never trains on those")
            rows.append(training_row(e, kind, conv))
            m = e["meta"]["v3"]
            v = {"example_id": e["example_id"], "kind": kind, "verification": m["verification"], "final_verdict": e["final_verdict"]}
            if kind in ("repair_injected", "correct_slow"):
                v["target_static_verdict"] = static[sha256(e["turns"][-1]["code"])]["verdict"]
            if kind == "repair_injected":
                v.update(mutation=m["mutation"], broken_verdict=e["turns"][2]["verdict"], feedback=e["turns"][2]["feedback"])
            for k in ("target_static_verdict", "numerics", "lowered", "ref_precision", "nearest_exam"):
                if k in m:
                    v[k] = m[k]
            verif.append(v)
    val = [training_row(e, e["kind"], conv) for e in v1_val]
    ids = collections.Counter(r["example_id"] for r in rows + val)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        sys.exit(f"{len(dup)} duplicate example ids, e.g. {dup[:3]}")
    out = V3 / "final"
    write_jsonl(out / "train.jsonl", rows)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "verification.jsonl", verif)
    lens = [len(r["messages"][-1]["content"]) for r in rows]
    total = [sum(len(m["content"]) for m in r["messages"]) for r in rows]
    fin = lambda xs: int(max(xs) / CHARS_PER_TOKEN)
    files = ["train.jsonl", "val.jsonl", "verification.jsonl"]
    manifest = {"created_at": now(), "complete": len(present) == len(PARTS), "parts_present": present, "oracle_sha": ORACLE_SHA, "n_train": len(rows), "n_val": len(val),
                "train_by_kind": dict(collections.Counter(r["kind"] for r in rows)), "by_verification": dict(collections.Counter(v["verification"] for v in verif)),
                "by_final_verdict": dict(collections.Counter(v["final_verdict"] for v in verif)),
                "broken_verdicts": dict(collections.Counter(v["broken_verdict"] for v in verif if "broken_verdict" in v)),
                "mutations": dict(collections.Counter(v["mutation"] for v in verif if "mutation" in v)),
                "n_targets_static_checked": len(need), "n_targets_not_static_ok": len(bad),
                "tokens_ESTIMATED": {"how": f"chars / {CHARS_PER_TOKEN}: not a tokenizer run. Confirm with build_rows' tokenizer before training.",
                                     "max_target": fin(lens), "max_row": fin(total), "max_target_limit": MAX_TARGET_TOKENS},
                "files": {f: sha256((out / f).read_bytes()) for f in files}}
    write_json(out / "MANIFEST.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("complete", "parts_present", "n_train", "n_val", "train_by_kind", "by_verification", "n_targets_not_static_ok", "tokens_ESTIMATED")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
