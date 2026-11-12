"""Stage 10: the reference chat renderer. One place defines how a problem, a kernel and an oracle feedback line look as chat turns,
so the constructor agent, the colleague's trainer and (if its owner adopts it) the eval harness all show the same text.

  python dataset/scripts/render_chat.py --check FINAL_DIR     round-trips every example: the fenced code equals `code` byte for byte
  python dataset/scripts/render_chat.py --show FINAL_DIR [N]  prints example N rendered

render(example) -> [{"role": "user"|"assistant", "content": str}, ...]. Targets are code only: the assistant content is exactly one
```python fence around the module. The user turn is a fixed instruction plus the fenced reference; a feedback turn is
"Oracle verdict: <feedback>" plus a fixed retry instruction. Nothing else is added, so the colleague's chat template is the only
other layer between these strings and the model.
"""
import argparse, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import read_jsonl  # noqa: E402

PROMPT_VERSION = "render-1"

INSTRUCTION = (
    "Write a Pallas TPU kernel for the JAX reference below. Reply with one Python module that defines `workload` with the same "
    "arguments as the reference and does its computation inside `pl.pallas_call`. Outside the kernel use only layout and dtype "
    "operations (reshape, transpose, pad, astype). Take sizes from the input shapes, not from literals. The module must import "
    "everything it uses. Target TPU v6e with JAX 0.10.2. Reply with the code only."
)
RETRY_INSTRUCTION = (
    "Fix the kernel. Reply with the complete corrected module, code only."
)
FENCE = re.compile(r"```(?:python)?\n(.*?)\n```", re.S)


def fence(code: str) -> str:
    return "```python\n" + code.rstrip("\n") + "\n```"


def unfence(text: str) -> str | None:
    """The single fenced Python block in `text`, or the whole text if it has no fence and parses as a module; else None."""
    blocks = FENCE.findall(text)
    if len(blocks) == 1:
        return blocks[0] + "\n"
    if not blocks and "def workload" in text:
        return text.strip("\n") + "\n"
    return None


def render_problem(ref_code: str) -> str:
    return INSTRUCTION + "\n\n" + fence(ref_code)


def render_feedback(feedback: str) -> str:
    return "Oracle verdict: " + feedback + "\n\n" + RETRY_INSTRUCTION


def render(example: dict) -> list:
    out = []
    for t in example["turns"]:
        if t["type"] == "problem":
            out.append({"role": "user", "content": render_problem(t["ref_code"])})
        elif t["type"] == "kernel":
            out.append({"role": "assistant", "content": fence(t["code"])})
        elif t["type"] == "oracle_feedback":
            out.append({"role": "user", "content": render_feedback(t["feedback"])})
        else:
            raise ValueError(f"unknown turn type {t['type']!r}")
    return out


def check(final_dir: pathlib.Path) -> list:
    """Round-trip failures ("<example_id>: <why>") over train.jsonl and val.jsonl."""
    bad = []
    for name in ("train.jsonl", "val.jsonl"):
        for ex in read_jsonl(final_dir / name):
            msgs = render(ex)
            kernels = [t for t in ex["turns"] if t["type"] == "kernel"]
            assistant = [m for m in msgs if m["role"] == "assistant"]
            if len(kernels) != len(assistant):
                bad.append(f"{ex['example_id']}: {len(kernels)} kernels but {len(assistant)} assistant turns")
                continue
            for k, m in zip(kernels, assistant):
                back = unfence(m["content"])
                if back is None or back.rstrip("\n") != k["code"].rstrip("\n"):
                    bad.append(f"{ex['example_id']}: fenced code does not round-trip")
            if msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
                bad.append(f"{ex['example_id']}: must start with user and end with assistant")
            if any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])):
                bad.append(f"{ex['example_id']}: roles must alternate")
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", metavar="FINAL_DIR")
    ap.add_argument("--show", nargs="+", metavar=("FINAL_DIR", "N"))
    a = ap.parse_args(argv)
    if a.check:
        bad = check(pathlib.Path(a.check))
        print("\n".join(bad) if bad else "render check: ok")
        sys.exit(1 if bad else 0)
    if a.show:
        d, n = pathlib.Path(a.show[0]), int(a.show[1]) if len(a.show) > 1 else 0
        ex = read_jsonl(d / "train.jsonl")[n]
        for m in render(ex):
            print(f"--- {m['role']}\n{m['content']}\n")
    if not (a.check or a.show):
        ap.print_help()


if __name__ == "__main__":
    main()
