"""Two extra result figures (fig16, fig17), computed from the raw JAXBench runs through paper/analysis/jaxbench.py.

  uv run --with matplotlib python extra_charts.py

The expected counts are asserted, so the script fails if the run data changes under it.
"""
import math, pathlib, statistics, sys, textwrap

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "paper" / "analysis"), str(REPO / "paper" / "figures" / "src")]
import jaxbench as jb
import style as st
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

plt = st.plt
OUT = pathlib.Path(__file__).resolve().parent.parent
PASS_BAR = 0.7
OURS, FLASH = "Gemma 4 E4B + our LoRA", "Gemini Flash + docs"
C_OURS, C_FLASH, C_BOTH, C_NEITHER = "#104281", "#b77c00", st.INK2, "#ecebe6"


def outcomes(arm):
    per, _ = jb.load(*jb.ARMS[arm][:2])
    return {t: jb.task_outcome(v) for t, v in per.items() if jb.is_primary(t)}


def order(nums):
    return sorted(nums, key=lambda n: int(n[:-1]))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", OUT / name)


def swarm(xs_in, dia):
    """Vertical offsets (inches) so that dots of diameter dia at horizontal positions xs_in do not overlap."""
    placed, ys = [], {}
    for i in sorted(range(len(xs_in)), key=lambda i: xs_in[i]):
        for k in range(200):
            y = (k + 1) // 2 * dia * (1 if k % 2 else -1)
            if all(math.hypot(xs_in[i] - px, y - py) >= dia for px, py in placed):
                break
        placed.append((xs_in[i], y))
        ys[i] = y
    return [ys[i] for i in range(len(xs_in))]


def fig16(oc):
    rows = [(FLASH, "flash_docs", 36, 7, 0.54), (OURS, "ours", 9, 9, 1.37)]
    W, H, L, R, B, T = 6.5, 2.4, 1.42, 1.3, 0.42, 0.26
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    speeds = {arm: [o["best_correct_speed"] for o in oc[arm].values() if o["best_correct_speed"]] for _, arm, *_ in rows}
    lo, hi = min(map(min, speeds.values())), max(map(max, speeds.values()))
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.5, hi * 1.5)
    ax.set_ylim(-0.62, 1.5)
    in_per_decade = (W - L - R) / math.log10(hi * 1.5 / (lo / 1.5 ))
    in_per_row = (H - B - T) / 2.12
    ax.axvline(PASS_BAR, color=st.INK2, lw=0.9, zorder=1)
    ax.axvline(1.0, color=st.AXIS, lw=0.9, zorder=1)
    tr = ax.get_xaxis_transform()
    ax.text(PASS_BAR / 1.04, 1.02, "pass bar (0.7x)", transform=tr, ha="right", va="bottom", fontsize=7, color=st.INK2)
    ax.text(1.0 * 1.04, 1.02, "XLA", transform=tr, ha="left", va="bottom", fontsize=7, color=st.MUTED)
    report = {}
    for y0, (label, arm, n_exp, p_exp, med_exp) in zip((1, 0), rows):
        xs = speeds[arm]
        n, p, med = len(xs), sum(x >= PASS_BAR for x in xs), statistics.median(xs)
        assert (n, p, round(med, 2)) == (n_exp, p_exp, med_exp), (arm, n, p, med)
        assert p == sum(o["pass"] for o in oc[arm].values()), arm
        report[arm] = (n, p, med)
        dy = swarm([math.log10(x) * in_per_decade for x in xs], 0.068)
        ax.scatter(xs, [y0 + d / in_per_row for d in dy], s=17, lw=0.4, edgecolors="white", zorder=3,
                   c=[st.VERDICT["pass" if x >= PASS_BAR else "correct_slow"] for x in xs])
        # below the dots, so the tick never hides one
        ax.plot([med, med], [y0 - 0.54, y0 - 0.41], color=st.INK, lw=1.4, zorder=4, solid_capstyle="butt")
        ax.text(1.025, y0, f"{n} correct, {p} pass\nmedian {med:.2f}x", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.5, color=st.INK, linespacing=1.3)
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=8)
    ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if lo / 1.5 <= t <= hi * 1.5]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}x" for t in ticks])
    ax.set_xlabel("Speed of the fastest correct kernel relative to XLA (log scale); black tick = median")
    ax.spines["left"].set_visible(False)
    save(fig, "fig16_speed_of_correct_kernels")
    return report


def fig17(oc):
    ours, flash = oc["ours"], oc["flash_docs"]
    tasks = sorted(ours)
    assert len(tasks) == 58 and set(tasks) == set(flash), (len(tasks), len(flash))
    W, H, X0, BW, BH = 6.5, 2.6, 0.2, 6.1, 0.28
    fig, ax = st.canvas(W, H)
    cats = [("Both", C_BOTH, "white"), ("Only " + OURS, C_OURS, "white"), ("Only " + FLASH, C_FLASH, "white"), ("Neither", C_NEITHER, st.INK2)]
    x = X0
    for label, c, _ in cats:
        ax.add_patch(Rectangle((x, H - 0.24), 0.13, 0.1, fc=c, ec="none"))
        ax.text(x + 0.18, H - 0.19, label, va="center", ha="left", fontsize=7.2, color=st.INK2)
        x += 0.18 + len(label) * 7.2 * st.CHAR + 0.28
    report = {}
    bars = [("pass", "Pass: correct and at least 0.7x XLA", 1.93, (1, 8, 6, 43)), ("correct", "Correct at any speed", 0.9, (8, 1, 28, 21))]
    for key, title, top, expected in bars:
        groups = [order(jb.num(t) for t in tasks if (ours[t][key], flash[t][key]) == pat)
                  for pat in ((True, True), (True, False), (False, True), (False, False))]
        counts = tuple(map(len, groups))
        assert counts == expected, (key, counts, expected)
        report[key] = {"counts": counts, "only_ours": groups[1], "only_flash": groups[2]}
        ax.text(X0, top + 0.21, title, va="bottom", ha="left", fontsize=8.5, fontweight="bold", color=st.INK)
        x = X0
        for i, (g, (_, c, tc)) in enumerate(zip(groups, cats)):
            w = len(g) * BW / 58
            ax.add_patch(Rectangle((x, top - BH), w, BH, fc=c, ec="white", lw=0.8))
            if len(g) >= 3:
                ax.text(x + w / 2, top - BH / 2, str(len(g)), ha="center", va="center", fontsize=8, fontweight="bold", color=tc)
            else:  # too narrow for a number inside: write it above the segment
                ax.text(x + w / 2, top + 0.03, str(len(g)), ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=c)
            if i in (1, 2):
                per_line = max(1, int(w / (4.3 * 6.2 * st.CHAR)))
                per_line = math.ceil(len(g) / math.ceil(len(g) / per_line))  # balanced lines
                for k, j in enumerate(range(0, len(g), per_line)):
                    ax.text(x + w / 2, top - BH - 0.13 - 0.115 * k, " ".join(g[j:j + per_line]), ha="center", va="baseline", fontsize=6.2, color=c)
            x += w
    ax.text(X0, 0.08, "58 tasks, 3 attempts each. Numbers under a segment are the JAXBench task ids won by one model only.",
            ha="left", va="bottom", fontsize=6.6, color=st.MUTED)
    save(fig, "fig17_paired_outcomes")
    return report


if __name__ == "__main__":
    oc = {arm: outcomes(arm) for arm in ("ours", "flash_docs")}
    for arm, (n, p, med) in fig16(oc).items():
        print(f"{arm}: {n} correct, {p} pass, median best correct speed {med:.3f}x")
    for key, r in fig17(oc).items():
        print(key, "both/ours/flash/neither", r["counts"], "| only ours:", " ".join(r["only_ours"]), "| only Flash:", " ".join(r["only_flash"]))
