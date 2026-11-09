"""Paper figures (static PNG + SVG, light surface). Reference palette from the dataviz skill, validated light: slots 1-4 pass; text uses
text tokens, every mark is direct-labeled (two slots sit below 3:1 contrast on the surface, so labels are required).
  python paper/make_figures.py RESULTS_DIR OUT_DIR
RESULTS_DIR holds round-1 oracle rows as res_<job>.jsonl (b1-r1, b2-r1, k1-r1).
"""
import json, math, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def rows(d: pathlib.Path, job: str) -> list:
    p = d / f"res_{job}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def save(fig, out: pathlib.Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_attempts(out: pathlib.Path, arms: dict) -> None:
    """Tasks passed (of 64) within k attempts. bare/docs from the control run (0 at every k), flash from control/stats.py."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for (name, ys), color in zip(arms.items(), (S1, S2, S3, S4)):
        ax.plot([1, 2, 3], ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{name}: {ys[-1]}/64", (3, ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK2)
    ax.set_xticks([1, 2, 3], ["1 attempt", "2", "3"]); ax.set_xlim(0.8, 3.9)
    ax.set_ylabel("JAXBench tasks passed (of 64)"); ax.set_ylim(-0.5, max(max(v) for v in arms.values()) + 2)
    ax.set_title("Tasks passed within k attempts (oracle feedback between attempts)", loc="left", fontsize=10, color=INK)
    save(fig, out, "fig1_success_vs_attempts")


def fig_speed(out: pathlib.Path, rs: list) -> None:
    """Distribution of speed_vs_xla over the correct round-1 kernels (log2 x)."""
    sp = [r["speed_vs_xla"] for r in rs if r.get("verdict") in ("pass", "correct_slow") and r.get("speed_vs_xla")]
    lo, hi = math.floor(math.log2(min(sp))), math.ceil(math.log2(max(sp)))
    bins = [2 ** (lo + i / 4) for i in range(int((hi - lo) * 4) + 1)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(sp, bins=bins, color=S1, edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log", base=2)
    for x, lab, dx, ha in ((0.7, "pass bar 0.7x", -4, "right"), (1.0, "XLA 1.0x", 4, "left")):
        ax.axvline(x, color=INK2, linewidth=1, linestyle="--")
        ax.annotate(lab, (x, ax.get_ylim()[1] * 0.92), xytext=(dx, 0), textcoords="offset points", ha=ha, color=INK2, fontsize=9)
    ticks = [2.0 ** k for k in range(lo, hi + 1, 2)]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    med = sorted(sp)[len(sp) // 2]
    ax.set_xlabel("speed vs XLA (log scale)"); ax.set_ylabel("kernels")
    ax.set_title(f"Correct constructor kernels, round 1: n={len(sp)}, median {med:.2f}x", loc="left", fontsize=10, color=INK)
    save(fig, out, "fig2_speed_distribution")


def fig_by_type(out: pathlib.Path, rs: list) -> None:
    """Round-1 outcome by problem type: share pass / correct but slow / failed (stacked, 2px gaps)."""
    def cat(r):
        src = "authored" if r["task"].startswith("authored__") else "KernelBench" if r["task"].startswith("kb__") else None
        return None if src is None else f"{src}, {'conv' if 'onv' in r['task'] else 'non-conv'}"
    groups = {}
    for r in rs:
        c = cat(r)
        if c:
            groups.setdefault(c, []).append(r["verdict"])
    order = [c for c in ("authored, non-conv", "authored, conv", "KernelBench, non-conv", "KernelBench, conv") if c in groups]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for i, c in enumerate(order):
        v, n = groups[c], len(groups[c])
        parts = [("pass", sum(x == "pass" for x in v), S1), ("correct but slow", sum(x == "correct_slow" for x in v), S4),
                 ("failed", sum(x not in ("pass", "correct_slow") for x in v), GRID)]
        left = 0
        for lab, k, color in parts:
            if k:
                ax.barh(i, k / n * 100, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.6)
                if k / n >= 0.08:
                    ax.text(left + k / n * 50, i, f"{100 * k / n:.0f}%", ha="center", va="center", color=INK, fontsize=9)
                left += k / n * 100
        ax.text(101, i, f"n={n}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(order)), order); ax.invert_yaxis(); ax.set_xlim(0, 110); ax.set_xlabel("% of problems")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S4, GRID)], labels=["pass", "correct but slow", "failed"],
              frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig3_outcome_by_type")


if __name__ == "__main__":
    d, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    r1 = rows(d, "b1-r1") + rows(d, "b2-r1") + rows(d, "k1-r1")
    # bare and docs are identical (0 at every k), so one labeled line; the LoRA arm is added after the eval
    fig_attempts(out, {"Gemini Flash + docs": [1, 4, 8], "Gemma 4 E4B (bare, + docs, + LoRA v1)": [0, 0, 0]})
    fig_speed(out, r1)
    fig_by_type(out, r1)
    print("wrote", sorted(p.name for p in out.iterdir()))
