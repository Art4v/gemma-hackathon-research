"""The five pipeline diagrams. Counts come from data.json (see extract.py); every other fact is quoted from the
script named in ../README.md beside the figure, so each box can be checked against its source.

  uv run --with matplotlib python paper/figures/src/diagrams.py
"""
import json

from style import EDGE, HERE, arrow, box, canvas, group, key, note, save

D = json.loads((HERE / "data.json").read_text())
T = D["train"]
ROWS, KIMI_N, GEN_N = T["rows"], T["by_teacher"]["kimi-k3.1"], T["by_teacher"]["generator-v7"]
SLOW, PASSED = T["by_verdict"]["correct_slow"], T["by_verdict"]["pass"]
ONE_TURN = T["by_messages"]["3"]
CHAINS = ROWS - ONE_TURN
assert min(T["speed"]["generator-v7"]) >= 0.7, "the diagrams say every below-bar target is a teacher row"
V = D["candidates"]["gen7V"]["pass"]
N = lambda x: f"{x:,}"


def chain(ax, boxes):
    for l, r in zip(boxes, boxes[1:]):
        arrow(ax, [(l.r, l.cy), (r.l, r.cy)])


def overview():
    fig, ax = canvas(7.2, 3.5)
    top, h, w = 2.98, 1.15, 1.24
    spec = [("Reference problems", "Plain JAX functions from KernelBook, KernelBench, authored and GitHub sources, plus three generated families at exam scale", "data"),
            ("Candidate kernels", "The teacher, Kimi K3.1, writes a Pallas kernel per problem; a template generator instantiates tuned kernels", "step"),
            ("Training set", f"{N(ROWS)} conversations: {N(KIMI_N)} from the teacher, {N(GEN_N)} generated. Every target was judged on a TPU", "data"),
            ("LoRA fine-tune", "Gemma 4 E4B, rank 64, 2 epochs. The adapter is folded into the 4-bit model that is served", "model"),
            ("JAXBench", "64 tasks at full size, 58 in the primary set. Up to 3 attempts per task, with oracle feedback", "data")]
    b = [box(ax, 0.1 + i * 1.44, top, w, t, body, kind, h=h) for i, (t, body, kind) in enumerate(spec)]
    chain(ax, b)
    y = top + 0.2
    ax.plot([b[2].cx, b[2].cx, b[4].cx, b[4].cx], [top, y, y, top], color=EDGE["gate"], lw=0.9, ls=(0, (3, 2)))
    note(ax, (b[2].cx + b[4].cx) / 2, y + 0.11, "disjoint: no training row shares a computation chain with a benchmark task", ha="center")

    o = box(ax, b[1].l, 1.3, b[4].r - b[1].l, "TPU oracle", "One TPU v6e chip, JAX 0.10.2, 3 seeds. The first failing step is the verdict.", "oracle", h=1.08)
    steps = [("1 Kernel check", "traced, nothing run"), ("2 Compile and run", "on every seed"),
             ("3 Correctness", "vs a float32 reference"), ("4 Speed", "at 0.7x of jax.jit")]
    chain(ax, [box(ax, o.l + 0.12 + i * 1.37, 0.9, 1.22, t, body, "step", h=0.52, fs=6.6) for i, (t, body) in enumerate(steps)])
    for bx, down in ((b[1], "kernel"), (b[4], "attempt")):
        arrow(ax, [(bx.cx - 0.16, bx.b), (bx.cx - 0.16, o.t)])
        arrow(ax, [(bx.cx + 0.16, o.t), (bx.cx + 0.16, bx.b)])
        note(ax, bx.cx - 0.22, (bx.b + o.t) / 2, down, ha="right")
        note(ax, bx.cx + 0.22, (bx.b + o.t) / 2, "verdict and\nfeedback")
    for i, k in enumerate(("data", "step", "model", "oracle")):
        key(ax, 0.14, 1.18 - i * 0.2, [k])
    save(fig, "fig01_overview")


def dataset():
    fig, ax = canvas(7.2, 4.9)
    w, h = 1.08, 1.5
    xs = [0.2 + i * 1.24 for i in range(4)]

    group(ax, 0.1, 4.8, 5.0, 2.2, f"Teacher lane (Kimi K3.1): {N(KIMI_N)} rows, {SLOW} with a correct target below 0.7x XLA")
    spec = [("Reference problems", "KernelBook, KernelBench, authored and GitHub modules as plain JAX; matmul-heavy ones also at exam size", "data"),
            ("Teacher writes a kernel", "Kimi K3.1 returns one Pallas kernel per reference. Rows store it under the benchmark's prompt", "step"),
            ("TPU oracle", "Kernel check, compile and run, correctness on 3 seeds, then speed against jax.jit on a TPU v6e", "oracle"),
            ("Rows kept", "One-turn rows (reference, then kernel) and repair chains (failed attempts with their feedback, then the fix)", "data")]
    k = [box(ax, x, 4.5, w, t, body, kind, h=h) for x, (t, body, kind) in zip(xs, spec)]
    chain(ax, k)
    arrow(ax, [(k[2].cx, k[2].b), (k[2].cx, 2.85), (k[1].cx, 2.85), (k[1].cx, k[1].b)])
    note(ax, (k[1].cx + k[2].cx) / 2, 2.72, "not a pass: the attempt and the oracle's feedback line go back to the teacher, up to 3 attempts", ha="center")

    group(ax, 0.1, 2.45, 5.0, 2.2, f"Generator lane (templates): {N(GEN_N)} rows, {T['variants']} of them exam-wording variants")
    spec = [("Kernel templates", "Three families: matmul with an epilogue, row-reduction matmul, and batched matmul with an epilogue", "step"),
            ("Instantiate at exam scale", "A sampled op chain and matmul shapes up to M 4,096, K 8,192, N 8,192 give a reference and its kernel", "step"),
            ("TPU oracle", "Every generated kernel is run. Only a pass at 0.7x XLA or faster becomes a row", "oracle"),
            ("Exam-wording variant", f"Each kept row again in the benchmark's identifier names; alpha-equivalent to the judged kernel, {V} of {V} sampled pass", "data")]
    g = [box(ax, x, 2.15, w, t, body, kind, h=h) for x, (t, body, kind) in zip(xs, spec)]
    chain(ax, g)
    note(ax, 0.2, 0.45, "Tile sizes and the VMEM limit were fixed by TPU sweeps before generation (tile-sweep figure).")

    f1 = box(ax, 5.3, 4.8, 1.8, "Leakage and holdout filters", "A row is dropped when its computation chain equals a JAXBench task's (amendment A6, rule 2). "
             "29 generated rows are held out by chain as the dev set.", "gate")
    f2 = box(ax, 5.3, f1.b - 0.3, 1.8, "Independent verifier", "A second script re-checks the assembled file: prompt format, drop lists, a TPU pass behind every generated row, "
             "alpha-equivalence of variants.", "step")
    f3 = box(ax, 5.3, f2.b - 0.3, 1.8, "Training file", f"{N(ROWS)} conversations: {N(ONE_TURN)} one-turn rows and {CHAINS} repair chains. "
             f"{N(PASSED)} targets at 0.7x XLA or faster, {SLOW} correct but slower.", "data")
    arrow(ax, [(k[3].r, 4.2), (f1.l, 4.2)])
    arrow(ax, [(g[3].r, g[3].cy), (5.2, g[3].cy), (5.2, f1.b + 0.15), (f1.l, f1.b + 0.15)])
    arrow(ax, [(f1.cx, f1.b), (f2.cx, f2.t)])
    arrow(ax, [(f2.cx, f2.b), (f3.cx, f3.t)])
    save(fig, "fig02_dataset_pipeline")


def oracle():
    fig, ax = canvas(7.2, 3.3)
    top, h, w = 3.15, 1.52, 1.15
    i0 = box(ax, 0.1, top, 1.0, "Inputs", "A reference JAX function, a candidate workload(), and seeds 0, 1, 2 at the task's full shapes", "data", h=h)
    spec = [("1 Kernel check", "Traced at full shapes, nothing run. Needs a pallas_call the output depends on and no heavy work outside it.", "not_a_kernel", ""),
            ("2 Compile and run", "On one TPU v6e chip, on every seed, within 480 s per solution.", "compile, vmem, mosaic, timeout, runtime", ""),
            ("3 Correctness", "On every seed, against the reference at float32, matmul precision highest: allclose with rtol = atol = 1e-2, "
             "or error within 2x XLA's own.", "wrong", ""),
            ("4 Speed", "Seed 0: median of 20 calls after 3 warm-ups. speed = t_xla / t_kernel, against jax.jit of the reference.", "correct_slow", "below 0.7x")]
    s = [box(ax, 1.25 + i * 1.28, top, w, t, body, "step", h=h) for i, (t, body, _f, _fb) in enumerate(spec)]
    chain(ax, [i0, *s])
    for bx, (_t, _b, fail, fbody) in zip(s, spec):
        f = box(ax, bx.l, bx.b - 0.3, w, fail, fbody, "fail")
        arrow(ax, [(bx.cx, bx.b), (bx.cx, f.t)])
        note(ax, bx.cx + 0.06, bx.b - 0.15, "fails")
    p = box(ax, 6.42, s[3].cy + 0.27, 0.68, "pass", "0.7x XLA or faster", "oracle")
    arrow(ax, [(s[3].r, p.cy), (p.l, p.cy)])
    box(ax, 0.1, 0.62, 7.0, "", "Each verdict carries one feedback line, at most 400 characters of the chip's own error text, which the next repair turn sees. "
        "harness_oom and too_large_for_chip are failures of the harness, not of the kernel, and end the task. Oracle code sha c9a07d9298a1, JAX 0.10.2.",
        "data", fs=6.8)
    save(fig, "fig03_tpu_oracle")


def training():
    fig, ax = canvas(7.2, 3.25)
    top, h = 2.08, 1.5
    d = box(ax, 0.1, top, 1.2, "Training file", f"{N(ROWS)} conversations in the benchmark's own prompt format, thinking off", "data", h=h)
    t = box(ax, 1.5, top, 1.5, "LoRA training", "Base: the 4-bit weights dequantised bit for bit, frozen. Rank 64, alpha 128 on the attention and MLP projections. "
            "Loss on the final assistant turn only. 2 epochs, lr 1e-4, accumulation 4. One A100 80 GB, 89 min.", "step", h=h)
    a = box(ax, 3.2, top, 0.95, "Adapter", "558 MB of LoRA weights, saved at the end of each epoch", "data", h=h)
    f = box(ax, 4.35, top, 1.45, "Fold", "For each of 258 tensors: q8_0(dequant(Q4_0) + (alpha / r) B A). Every other tensor is copied byte for byte.", "step", h=h)
    s = box(ax, 6.0, top, 1.1, "Served model", "Ollama 0.40.14 on a Modal A10G. Its digest is checked on every generated row.", "model", h=h)
    chain(ax, [d, t, a, f, s])
    base = box(ax, 1.5, 3.15, 4.3, "Official 4-bit model", "gemma4:e4b-it-qat GGUF (Q4_0), 5.15 GB: the exact weights the base arm runs. Training and the fold both start from them.", "model")
    for b, label in ((t, "dequantised base"), (f, "base tensors")):
        arrow(ax, [(b.cx, base.b), (b.cx, b.t)])
        note(ax, b.cx + 0.07, (base.b + b.t) / 2, label)
    box(ax, 0.1, 0.45, 7.0, "", "sha256 chain: training file 10345461..., adapter a1259bb6..., folded GGUF 997a7145..., Ollama model digest 42b12500...", "data", fs=6.8)
    save(fig, "fig04_training_serving")


def evaluation():
    fig, ax = canvas(7.2, 2.55)
    top, h = 2.45, 1.2
    p = box(ax, 0.1, top, 1.35, "Prompt", "The system prompt, the task's reference JAX function, and the request to convert it into a Pallas TPU kernel", "data", h=h)
    m = box(ax, 1.65, top, 1.3, "Fine-tuned Gemma 4 E4B", "Greedy decoding, thinking off, 65,536-token context, at most 8,192 output tokens", "model", h=h)
    c = box(ax, 3.15, top, 1.15, "CPU kernel check", "Oracle step 1 on the code block of the reply", "step", h=h)
    o = box(ax, 4.5, top, 1.25, "TPU oracle", "The full oracle on one TPU v6e chip, 3 seeds", "oracle", h=h)
    v = box(ax, 5.95, top, 1.15, "Solved", "pass: correct on all 3 seeds at 0.7x XLA or faster", "oracle", h=h)
    chain(ax, [p, m, c, o, v])
    y = p.b - 0.3
    ax.plot([c.cx, c.cx], [c.b, y], color="#52514e", lw=0.9)
    arrow(ax, [(o.cx, o.b), (o.cx, y), (p.cx, y), (p.cx, p.b)])
    note(ax, (p.cx + o.cx) / 2, y - 0.13, "not solved: the attempt and the oracle's one-line feedback are appended to the conversation (turns 0, 1, 2)", ha="center")
    box(ax, 0.1, 0.5, 7.0, "", "Metric: pass_within_3, the number of tasks with a pass by turn 2, reported on the 58-task primary set; 64 tasks are run. "
        "A task also stops at harness_oom or too_large_for_chip, which no retry can fix.", "data", fs=6.8)
    save(fig, "fig05_evaluation_protocol")


if __name__ == "__main__":
    for f in (overview, dataset, oracle, training, evaluation):
        f()
