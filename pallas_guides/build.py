"""Build teacher_docs.md, the Pallas documentation the teacher (gemini-4-argon) gets, from upstream JAX docs at the pinned tag.

  python dataset/teacher_docs/build.py          -> writes teacher_docs.md and checks upstream/ against SOURCES.json

upstream/ holds unmodified copies of jax-ml/jax files at jax-v0.10.2 (the oracle's JAX pin), Apache-2.0 (upstream/LICENSE).
The quickstart keeps its TPU material only: paragraphs that start "On GPU", the two notes about the GPU backends, and the
"# GPU version" code cell are dropped, matched by content so a changed upstream fails here instead of passing through.
The other three files are used whole. Nothing here is written by an agent except this script and the source lines; the header
module (jaxbench/teacher_header.py) carries the agent-written guidance.
"""
import hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
UP = HERE / "upstream"
ORDER = ("quickstart.md", "details.rst", "matmul.md", "sparse.md")
GPU_LEAD = "On GPUs, we can call the kernel directly like so:"
GPU_NOTES = ("> Note: when using the Mosaic GPU backend", "> Note: there also exists a Triton backend on GPU")


def blocks(text: str) -> list:
    return re.split(r"\n\s*\n", text)


def quickstart_tpu(text: str) -> str:
    """The quickstart without its front matter and without its GPU-only blocks; each dropped block must exist exactly once."""
    body = text.split("\n---\n", 1)[1] if text.startswith("---") else text
    out, dropped = [], {"gpu_paragraphs": 0, "gpu_notes": 0, "gpu_cell": 0}
    for b in blocks(body):
        s = b.lstrip()
        if s.startswith("On GPU"):
            dropped["gpu_paragraphs"] += 1
        elif s.startswith(GPU_NOTES):
            dropped["gpu_notes"] += 1
        elif s.startswith("```{code-cell}") and "\n# GPU version\n" in s:
            dropped["gpu_cell"] += 1
        else:
            out.append(b)
    if dropped["gpu_cell"] != 1 or dropped["gpu_notes"] < 1 or dropped["gpu_paragraphs"] < 1:
        raise SystemExit(f"upstream quickstart changed shape: dropped {dropped}")
    text = "\n\n".join(out).strip() + "\n"
    if text.count(GPU_LEAD + "\n") != 1:  # the line that introduced the dropped GPU cell
        raise SystemExit("upstream quickstart changed shape: the GPU cell's lead-in line is not there exactly once")
    return text.replace(GPU_LEAD + "\n", "")


def main():
    sources = json.loads((HERE / "SOURCES.json").read_text())
    for name, want in sources["files"].items():
        got = hashlib.sha256((UP / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"upstream/{name} differs from SOURCES.json ({got[:12]} != {want[:12]})")
    parts = []
    for name in ORDER:
        text = (UP / name).read_text()
        text = quickstart_tpu(text) if name == "quickstart.md" else text
        parts.append(f"===== {sources['path'][name]} (jax-ml/jax @ {sources['tag']}) =====\n\n{text.strip()}\n")
    out = "\n\n".join(parts)
    (HERE / "teacher_docs.md").write_text(out)
    print(f"teacher_docs.md: {len(out)} characters, about {len(out) // 4} tokens")


if __name__ == "__main__":
    sys.exit(main())
