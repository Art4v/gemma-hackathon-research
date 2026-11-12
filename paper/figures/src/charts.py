"""The data figures, drawn from data.json alone (see extract.py).

  uv run --with matplotlib python paper/figures/src/charts.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from style import AXIS, GEN, GRID, HERE, INK, INK2, KIMI, MUTED, VERDICT, VERDICT_LABEL, VERSION, save

D = json.loads((HERE / "data.json").read_text())
T, R = D["train"], D["results"]
BAR = 0.7  # the oracle's speed bar
DASH = (0, (3, 2))
CPU_STOP, CPU_LABEL = "#dddcd5", "stopped at the CPU kernel check"
cls = lambda v: {"mosaic": "compile"}.get(v, v if v in VERDICT else "other")


def composition():
    s, total = T["by_set"], T["rows"]
    groups = [("Kimi: base set (v6)", ("pool", "e1_readmit", "e2_kbkd", "e3_scale"), KIMI),
              ("Kimi: exam-scale copies, verified or fixed", ("e4_kn", "e4_fix", "e4_fix2"), KIMI),
              ("Kimi: round-3 repairs", ("r3_gap",), KIMI),
              ("Generator: three families", ("e5_gen",), GEN),
              ("Generator: layout variants", ("e5_gen_layout",), GEN),
              ("Generator: row-output template", ("e5_gen7r",), GEN)]
    vals = [sum(s[k] for k in ks) for _, ks, _ in groups]
    assert sum(vals) == total, vals
    props = [("Repair chains", total - T["by_messages"]["3"]), ("Template-generated", T["by_teacher"]["generator-v7"]),
             ("Exam-wording variants", T["variants"]), ("Target correct, below 0.7x XLA", T["by_verdict"]["correct_slow"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    a.barh(range(len(vals)), vals, height=0.62, color=[c for *_, c in groups])
    a.set_yticks(range(len(vals)), [g[0] for g in groups])
    for i, v in enumerate(vals):
        a.text(v + 18, i, f"{v:,}", va="center", fontsize=7.5)
    a.set_xlim(0, 1380)
    a.set_xlabel("training rows")
    a.set_title(f"Where the {total:,} training rows come from")
    fig.legend(handles=[Patch(color=KIMI, label=f"target written by the teacher, Kimi K3.1 ({T['by_teacher']['kimi-k3.1']:,} rows)"),
                        Patch(color=GEN, label=f"target written by the template generator ({T['by_teacher']['generator-v7']:,} rows)")],
               ncols=2, loc="upper left", bbox_to_anchor=(0.0, 0.0))
    b.barh(range(len(props)), [total] * len(props), height=0.62, color="#f0efec")
    b.barh(range(len(props)), [v for _, v in props], height=0.62, color=INK2)
    b.set_yticks(range(len(props)), [p[0] for p in props])
    for i, (_, v) in enumerate(props):
        b.text(v + 30, i, f"{v:,} ({v / total:.0%})", va="center", fontsize=7.5)
    b.set_xlim(0, total)
    b.set_xlabel(f"rows, out of {total:,}")
    b.set_title("Properties that cut across sources")
    for ax in (a, b):
        ax.invert_yaxis()
        ax.tick_params(length=0)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    save(fig, "fig06_dataset_composition", tight=True)


def candidates():
    c = D["candidates"]
    rows = [("Kimi K3.1: third-attempt repairs", "kimi_r3"), ("Generator: three families", "gen7"), ("Generator: layout variants", "gen7L"),
            ("Generator: row-output template", "gen7R"), ("Generator: exam-wording variants (sample)", "gen7V")]
    fig, ax = plt.subplots(figsize=(7.0, 2.25), layout="constrained")
    seen = []
    for i, (_, k) in enumerate(rows):
        n, left = sum(c[k].values()), 0.0
        for v in VERDICT:
            x = 100 * c[k].get(v, 0) / n
            if not x:
                continue
            seen.append(v)
            ax.barh(i, x, left=left, height=0.62, color=VERDICT[v], edgecolor="white", linewidth=1.2)
            if x >= 4:
                ax.text(left + x / 2, i, str(c[k][v]), ha="center", va="center", fontsize=7, color=INK if v == "correct_slow" else "white")
            left += x
    ax.set_yticks(range(len(rows)), [f"{label} (n = {sum(c[k].values())})" for label, k in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of candidates judged on a TPU (%)")
    ax.tick_params(length=0)
    ax.set_title("TPU verdicts of the v7 training candidates, by source")
    ax.legend(handles=[Patch(color=VERDICT[v], label=VERDICT_LABEL[v]) for v in VERDICT if v in seen], ncols=3, loc="upper left", bbox_to_anchor=(0, -0.22))
    save(fig, "fig07_candidate_verdicts", tight=True)


def target_speed():
    lo, hi = 0.05, 20
    bins = np.concatenate([np.geomspace(lo, BAR, 24)[:-1], np.geomspace(BAR, hi, 30)])  # the bar is a bin edge
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.0), sharex=True, layout="constrained")
    for ax, (t, name, col) in zip(axs, (("kimi-k3.1", "Teacher Kimi K3.1", KIMI), ("generator-v7", "Template generator", GEN))):
        s = np.array(T["speed"][t])
        ax.hist(np.clip(s, lo, hi), bins=bins, color=col, edgecolor="white", linewidth=0.4)
        ax.axvline(BAR, color=INK, lw=0.9, ls=DASH)
        ax.axvline(1.0, color=AXIS, lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_ylabel("rows")
        ax.tick_params(length=2)
        ax.set_title(f"{name}: {len(s):,} rows, median {np.median(s):.2f}x, {int((s < BAR).sum())} below the bar")
    top = axs[0].get_ylim()[1]
    axs[0].text(BAR * 0.96, top * 0.92, "speed bar 0.7x", ha="right", va="top", fontsize=7, color=INK2)
    axs[0].text(1.04, top * 0.92, "as fast as XLA", ha="left", va="top", fontsize=7, color=INK2)
    ticks = [0.05, 0.1, 0.2, 0.5, 0.7, 1, 2, 5, 10, 20]
    axs[1].set_xticks(ticks, ["<=0.05", "0.1", "0.2", "0.5", "0.7", "1", "2", "5", "10", ">=20"])
    axs[1].minorticks_off()
    axs[1].set_xlim(lo, hi)
    axs[1].set_xlabel("speed of the training target relative to jax.jit of its reference (log scale)")
    save(fig, "fig08_target_speed", tight=True)


def tile_sweep():
    by = {}
    for r in D["sweep"]:
        if r["m"] == 2048:
            by.setdefault(tuple(r["tiles"]), []).append(r)
    order = sorted(by, key=lambda k: -max(r["speed"] for r in by[k]))
    fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
    for i, k in enumerate(order):
        xs = [r["speed"] for r in by[k]]
        ax.plot([min(xs), max(xs)], [i, i], color=AXIS, lw=1.2, zorder=1)
        for r in by[k]:
            ok = r["verdict"] == "pass"
            ax.scatter(r["speed"], i, s=30, marker="o" if ok else "s", color=VERDICT[r["verdict"]], edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(BAR, color=INK, lw=0.9, ls=DASH)
    ax.set_yticks(range(len(order)), [f"{tm} x {tn}, K tile {tk}" for tm, tn, tk in order])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("speed relative to jax.jit")
    ax.set_ylabel("output block (rows x columns)")
    ax.tick_params(length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Block shape decides the verdict")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=VERDICT["pass"], label="pass"),
                       Line2D([], [], marker="s", ls="", color=VERDICT["correct_slow"], label="correct, below 0.7x")],
              ncols=2, loc="upper left", bbox_to_anchor=(-0.55, -0.2), handletextpad=0.2)
    save(fig, "fig09_tile_sweep", tight=True)


def similarity():
    S = D["similarity"]
    groups = [("kimi", "Teacher rows", KIMI), ("gen", "Generated rows, generator wording", GEN), ("gen_variant", "Generated rows, exam wording", GEN),
              (None, "For scale, each JAXBench reference against its closest other JAXBench reference", MUTED)]
    bins = np.linspace(0, 1, 51)
    fig, axs = plt.subplots(len(groups), 1, figsize=(7.0, 4.1), sharex=True, layout="constrained")
    for ax, (g, name, col) in zip(axs, groups):
        v = [r["sim"] for r in S["rows"] if r["group"] == g] if g else S["between_benchmark_tasks"]
        ax.hist(v, bins=bins, color=col, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{name}: n = {len(v):,}, median {np.median(v):.2f}, max {max(v):.2f}", fontsize=7.8)
        ax.set_ylabel("rows" if g else "tasks")
        ax.tick_params(length=2)
    axs[-1].set_xlim(0, 1)
    axs[-1].set_xlabel("line similarity of a prompt's reference to the closest JAXBench reference (difflib ratio over lines)")
    save(fig, "fig10_prompt_similarity", tight=True)


def attempts():
    def within(m):
        tasks = R["models"][m]["tasks"].values()
        return [sum(1 for a in tasks if any(a.get(str(t), {}).get("verdict") == "pass" for t in range(k + 1))) for k in range(3)]
    lines = [(f"LoRA {m}", within(m), VERSION[m], "-") for m in ("v7", "v5", "v4")]
    lines += [(name, ys, INK2 if any(ys) else MUTED, DASH if any(ys) else "-") for name, ys in R["earlier_arms_within_k_of_64"].items()]
    fig, ax = plt.subplots(figsize=(7.0, 2.9), layout="constrained")
    ends = {}
    for name, ys, col, ls in lines:
        ax.plot([1, 2, 3], ys, color=col, lw=2, ls=ls, marker="o", ms=6.5, mec="white", mew=1.6, zorder=3, clip_on=False)
        ends.setdefault(ys[-1], []).append(name)
    for y, names in ends.items():
        ax.annotate(f"{', '.join(names)}: {y} of 64", (3, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.set_xticks([1, 2, 3], ["1 attempt", "2 attempts", "3 attempts"])
    ax.set_xlim(0.85, 5.6)
    ax.set_ylim(-0.6, 10.5)
    ax.set_ylabel("JAXBench tasks passed, of 64")
    ax.tick_params(length=0)
    ax.hlines(range(0, 11, 2), 0.85, 3.1, color=GRID, lw=0.6, zorder=0)
    ax.spines["bottom"].set_bounds(0.85, 3.1)
    ax.set_title("Tasks passed within k attempts, with the oracle's feedback between attempts")
    save(fig, "fig11_success_vs_attempts", tight=True)


def funnel():
    stages = [("b", "kernel check\npassed"), ("c", "compiles and\nruns on TPU"), ("d", "correct on\n3 seeds"), ("e", "pass at\n0.7x or faster")]
    fig, ax = plt.subplots(figsize=(3.6, 2.6), layout="constrained")
    width = 0.26
    for mi, m in enumerate(VERSION):
        f = R["models"][m]["funnel_primary58"]
        xs = [si + (mi - 1) * width for si in range(len(stages))]
        ax.bar(xs, [f[k] for k, _ in stages], width=width - 0.03, color=VERSION[m], label=f"LoRA {m}")
        for x, (k, _) in zip(xs, stages):
            ax.text(x, f[k] + 0.4, str(f[k]), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(stages)), [s for _, s in stages])
    ax.set_ylabel("tasks, of the 58 in the primary set")
    ax.set_ylim(0, 30)
    ax.tick_params(length=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Tasks reaching each oracle stage in 3 attempts")
    ax.legend(loc="upper right")
    save(fig, "fig12_results_funnel", tight=True)


def results_grid():
    tasks, models, n = R["matmul_k_tasks"], ("v4", "v5", "v7"), len(R["matmul_k_tasks"])
    fig, ax = plt.subplots(figsize=(7.0, 0.19 * n + 1.1))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for mi, m in enumerate(models):
        x0, passed = mi * 3.6, 0
        for ti, task in enumerate(tasks):
            for turn, cell in sorted(R["models"][m]["tasks"][task].items()):
                x = x0 + int(turn)
                ax.add_patch(Rectangle((x + 0.05, ti + 0.1), 0.9, 0.8, fc=VERDICT[cls(cell["verdict"])] if cell["tpu"] else CPU_STOP, ec="none"))
                if cell["verdict"] == "pass":
                    passed += 1
                    ax.text(x + 0.5, ti + 0.52, f"{cell['speed']:.2f}x", ha="center", va="center", fontsize=6.3, color="white", fontweight="bold")
        ax.text(x0 + 1.5, -1.25, f"LoRA {m}", ha="center", va="center", fontsize=8.5, fontweight="bold")
        for turn in range(3):
            ax.text(x0 + turn + 0.5, -0.4, str(turn + 1), ha="center", va="center", fontsize=7, color=INK2)
        ax.text(x0 + 1.5, n + 0.55, f"{passed} of {n} pass", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(-0.25, -0.4, "attempt", ha="right", va="center", fontsize=7, color=INK2)
    for ti, task in enumerate(tasks):
        ax.text(-0.25, ti + 0.5, task.replace("_", " "), ha="right", va="center", fontsize=7, color=INK2)
    handles = [Patch(color=VERDICT[v], label=VERDICT_LABEL[v]) for v in VERDICT if v != "other"] + [Patch(color=CPU_STOP, label=CPU_LABEL)]
    ax.legend(handles=handles, ncols=3, loc="upper center", bbox_to_anchor=(0.62, 0.085), fontsize=7)
    ax.set_xlim(-7.4, 10.9)
    ax.set_ylim(n + 3.9, -1.9)
    ax.axis("off")
    save(fig, "fig13_results_by_task", tight=True)


def outcome_by_type():
    """Final outcome per task (the first pass, else the last attempt), by task type and adapter."""
    is_k = lambda t: t.split("_")[0].endswith("k")
    types = [("Matmul-led k-tasks", R["matmul_k_tasks"]), ("Conv-led k-tasks", [t for t in R["other_tasks"] if is_k(t)]),
             ("p-tasks", [t for t in R["other_tasks"] if not is_k(t)])]
    assert all(t.split("_")[1].startswith("Conv") for t in types[1][1]), types[1][1]
    order = [*[v for v in VERDICT if v != "other"], "other", "cpu"]
    color, labels = {**VERDICT, "cpu": CPU_STOP}, {**VERDICT_LABEL, "cpu": CPU_LABEL}
    models = list(VERSION)

    def final(a):
        cell = next((c for _, c in sorted(a.items()) if c["verdict"] == "pass"), a[max(a)])
        return cls(cell["verdict"]) if cell["tpu"] else "cpu"

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.95), sharey=True, layout="constrained")
    seen = set()
    for ax, (name, tasks) in zip(axs, types):
        for i, m in enumerate(models):
            c = {k: 0 for k in order}
            for t in tasks:
                c[final(R["models"][m]["tasks"][t])] += 1
            left = 0
            for k in order:
                if c[k]:
                    seen.add(k)
                    ax.barh(i, c[k], left=left, height=0.62, color=color[k], edgecolor="white", linewidth=1.2)
                    if c[k] / len(tasks) >= 0.07:
                        ax.text(left + c[k] / 2, i, str(c[k]), ha="center", va="center", fontsize=7, color=INK if k in ("correct_slow", "cpu") else "white")
                    left += c[k]
        ax.set_title(f"{name} ({len(tasks)})", fontsize=8)
        ax.set_xlim(0, len(tasks))
        ax.set_xlabel("tasks")
        ax.tick_params(length=0)
    axs[0].set_yticks(range(len(models)), [f"LoRA {m}" for m in models])
    axs[0].invert_yaxis()
    fig.legend(handles=[Patch(color=color[k], label=labels[k]) for k in order if k in seen], ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig14_outcome_by_task_type", tight=True)


def training_loss():
    L, W = D["loss"], 100
    y, gen = np.array(L["loss"]), np.array(L["generated"], bool)
    x = np.arange(1, len(y) + 1)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 3.3), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [4, 1]})
    for mask, name, col in ((~gen, "teacher rows", KIMI), (gen, "generated rows", GEN)):
        a.plot(x[mask][W - 1:], np.convolve(y[mask], np.ones(W) / W, mode="valid"), color=col, lw=1.6, label=f"training loss, {name} (mean of the last {W})")
    dev = np.array(L["dev_loss"])
    a.plot(dev[:, 0].clip(1), dev[:, 1], ls="", marker="D", ms=5, color=INK, mec="white", mew=0.8, clip_on=False, zorder=4,
           label="dev loss (29 held-out generated rows)")
    a.set_yscale("log")
    a.set_ylim(1e-4, 2)
    a.set_ylabel("cross-entropy loss (log scale)")
    a.grid(axis="y")
    a.set_axisbelow(True)
    a.legend(loc="upper right")
    a.set_title("Training loss by who wrote the target")
    b.plot(x, np.array(L["lr"]) * 1e4, color=INK2, lw=1.4)
    b.set_ylabel("lr (1e-4)")
    b.set_ylim(0, 1.1)
    b.set_xlim(0, len(y))
    b.set_xlabel("training examples seen (4 per optimiser step)")
    for ax in (a, b):
        ax.axvline(len(y) / 2, color=AXIS, lw=0.8, zorder=0)
        ax.tick_params(length=2)
    for dx, ha, label in ((-40, "right", "epoch 1"), (40, "left", "epoch 2")):
        a.text(len(y) / 2 + dx, 1.4e-4, label, ha=ha, va="bottom", fontsize=7, color=INK2)
    save(fig, "fig15_training_loss", tight=True)


if __name__ == "__main__":
    for f in (composition, candidates, target_speed, tile_sweep, similarity, attempts, funnel, results_grid, outcome_by_type, training_loss):
        f()
