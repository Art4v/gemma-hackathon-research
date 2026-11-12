"""Aarav's assembled dataset (dataset/final/{train,val}.jsonl, structured `turns`) -> chat rows for training/train_lora.py.

Every prompt is built by control/bench.py's own conversation(), the function that builds the benchmark's prompts, so a training prompt is
byte-identical to what the student sees at evaluation (bare arm: no docs). Thinking is off, as in the benchmark, and rows carry no reasoning.
Only the last assistant turn is graded; in a repair chain the earlier, failing kernels and the oracle's feedback are prompt only.

  python training/build_rows.py --src REF_OR_DIR --out training/data/pilot   # writes train.jsonl, val.jsonl, MANIFEST.json
  REF_OR_DIR: a git ref holding dataset/final (e.g. origin/dataset-gen) or a local dataset/final directory.
"""
import argparse, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))
from bench import conversation  # noqa: E402  (the benchmark's prompt builder, not a copy)


def read_split(src: str, split: str) -> tuple[list, str]:
    p = pathlib.Path(src) / f"{split}.jsonl"
    if p.exists():
        text = p.read_text()
    else:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{src}:dataset/final/{split}.jsonl"], capture_output=True, text=True,
                              check=True).stdout
    return [json.loads(l) for l in text.splitlines() if l.strip()], hashlib.sha256(text.encode()).hexdigest()


def to_messages(turns: list) -> list:
    """turns: user(ref_code), assistant(code), [user(feedback), assistant(code)]*. The last turn must be the assistant's final kernel."""
    assert turns and turns[0]["role"] == "user" and "ref_code" in turns[0], "first turn must be the reference"
    assert turns[-1]["role"] == "assistant" and turns[-1].get("code"), "last turn must be the graded kernel"
    body = turns[1:-1]
    assert len(body) % 2 == 0, "repair turns must come in (assistant code, user feedback) pairs"
    history = []
    for a, u in zip(body[::2], body[1::2]):
        assert a["role"] == "assistant" and u["role"] == "user" and "feedback" in u, (a.get("role"), u.get("role"))
        history.append({"code": a["code"], "feedback": u["feedback"]})
    messages = conversation(turns[0]["ref_code"], history, None)
    messages.append({"role": "assistant", "content": f"```python\n{turns[-1]['code']}\n```", "reasoning": ""})
    return messages


def convert(row: dict) -> dict:
    assert row["final_verdict"] == "pass", f"{row['example_id']}: only passing kernels are training targets"
    return {"task": row["problem_id"], "example_id": row["example_id"], "kind": row["kind"], "n_repairs": row["n_repairs"],
            "source": row["source"], "speed_vs_xla": row["speed_vs_xla"], "thinking": False, "teacher": "claude",
            "messages": to_messages(row["turns"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="origin/dataset-gen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"src": args.src, "splits": {}}
    if not pathlib.Path(args.src).exists():
        manifest["src_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", args.src], capture_output=True, text=True).stdout.strip()
    tasks = {}
    for split in ("train", "val"):
        rows, sha = read_split(args.src, split)
        conv = [convert(r) for r in rows]
        tasks[split] = {r["task"] for r in conv}
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in conv))
        manifest["splits"][split] = {"rows": len(conv), "repairs": sum(r["n_repairs"] > 0 for r in conv), "src_sha256": sha}
    assert not tasks["train"] & tasks["val"], f"tasks in both splits: {tasks['train'] & tasks['val']}"
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
