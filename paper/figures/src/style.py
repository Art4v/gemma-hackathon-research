"""Shared style for the paper figures: colours, fonts, saving, and the box and arrow primitives the diagrams use.

The colour sets come from one reference palette and were each checked with its validator on a white surface
(teacher pair and verdict set as categorical palettes on adjacent pairs, the version ramp as an ordinal ramp).
Three verdict colours are below 3:1 contrast against white, so every figure that uses them labels its marks.
"""
import pathlib, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent  # paper/figures

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KIMI, GEN = "#2a78d6", "#eb6834"  # who wrote a training target: the same two colours in every figure
VERDICT = {"pass": "#1baf7a", "correct_slow": "#eda100", "wrong": "#e34948", "vmem": "#6250d6", "compile": "#e87ba4",
           "other": "#898781"}
VERDICT_LABEL = {"pass": "pass (0.7x XLA or faster)", "correct_slow": "correct, slower than 0.7x", "wrong": "wrong output",
                 "vmem": "VMEM overflow", "compile": "compile or Mosaic error", "other": "runtime or harness error"}
VERSION = {"v4": "#86b6ef", "v5": "#3987e5", "v7": "#104281"}  # successive adapters: one hue, light to dark
FILL = {"data": "#f4f3ef", "step": "#ffffff", "model": "#dbe9fb", "oracle": "#d7f1e6", "gate": "#fdf1d4", "fail": "#ffffff"}
EDGE = {"data": "#a9a79f", "step": "#52514e", "model": "#2a78d6", "oracle": "#12855c", "gate": "#b77c00", "fail": "#a9a79f"}
KIND_LABEL = {"data": "data", "step": "process", "model": "model", "oracle": "TPU oracle", "gate": "filter"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.frameon": False, "legend.fontsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6, "text.color": INK,
})

CHAR = 0.0069  # inches per character per point of font size: Arial averages about half an em
PAD = 0.07


class Box:
    """Geometry of a drawn node, for attaching arrows. (x, y) is the bottom-left corner, in inches."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.l, self.r, self.b, self.t, self.cx, self.cy = x, x + w, y, y + h, x + w / 2, y + h / 2


def canvas(w, h):
    """A figure whose data units are inches, so box sizes and font sizes can be reasoned about together."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _wrap(text, width_in, fs):
    return textwrap.wrap(text, max(6, int(width_in / (fs * CHAR)))) or [""]


def box(ax, x, top, w, title, body="", kind="step", h=None, fs=7.2):
    """A rounded node with its top-left corner at (x, top). It is as tall as its wrapped text needs unless h is given;
    text that does not fit a given h is an error, never an overflow."""
    lh = fs * 1.3 / 72
    tl = _wrap(title, w - 2 * PAD, fs * 1.1) if title else []  # bold runs wider
    bl = [l for para in body.split("\n") for l in _wrap(para, w - 2 * PAD, fs)] if body else []
    gap = 0.035 if tl and bl else 0
    need = 2 * PAD + lh * (len(tl) + len(bl)) + gap
    h = h or need
    assert need <= h + 1e-6, f"text of {title!r} needs {need:.2f} in, the box has {h:.2f}"
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=FILL[kind], ec=EDGE[kind],
                                lw=0.9, ls="--" if kind == "fail" else "-"))
    y = top - PAD
    for l in tl:
        ax.text(x + PAD, y, l, va="top", ha="left", fontsize=fs, fontweight="bold", color=INK)
        y -= lh
    y -= gap
    for l in bl:
        ax.text(x + PAD, y, l, va="top", ha="left", fontsize=fs, color=INK2)
        y -= lh
    return Box(x, top - h, w, h)


def group(ax, x, top, w, h, label):
    """A lane behind several nodes, labelled at its top-left."""
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="#fbfbf9", ec=GRID, lw=0.9, zorder=0))
    ax.text(x + 0.1, top - 0.08, label, va="top", ha="left", fontsize=7.2, fontweight="bold", color=INK2)
    return Box(x, top - h, w, h)


def arrow(ax, pts, dashed=False, color=INK2):
    """An arrow along the polyline pts; the head is at the last point."""
    ls = (0, (3, 2)) if dashed else "-"
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=0.9, ls=ls, solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color, ls=ls, shrinkA=0, shrinkB=0))


def note(ax, x, y, text, ha="left", va="center", fs=6.4, color=INK2, **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.25, **kw)


def key(ax, x, y, kinds):
    """A one-line legend of node kinds, starting at (x, y)."""
    for k in kinds:
        ax.add_patch(FancyBboxPatch((x, y - 0.045), 0.16, 0.09, boxstyle="round,pad=0,rounding_size=0.02", fc=FILL[k], ec=EDGE[k], lw=0.8))
        ax.text(x + 0.21, y, KIND_LABEL[k], va="center", ha="left", fontsize=6.4, color=INK2)
        x += 0.3 + len(KIND_LABEL[k]) * 6.4 * CHAR + 0.12


def save(fig, name, tight=False):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", **({"bbox_inches": "tight", "pad_inches": 0.04} if tight else {}))
    plt.close(fig)
    print("wrote", name)
