"""Schematics / cartoons for slides 2, 9, 10, 11 -> presentation/assets/.

Built with matplotlib patches for precise placement and palette consistency,
saved as scalable SVG (crisp in print-PDF) plus a PNG fallback. No data, no
pipeline code; pure illustration.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, FancyBboxPatch,
                                FancyArrowPatch, Rectangle, Wedge)

from style import apply_style, ASSET_DIR, C

apply_style()


def save_asset(fig, name):
    # SVG (kept for reference) + a high-res PNG that rasterizes reliably in
    # Chrome's print/PDF path (some SVGs silently drop out of the print export).
    fig.savefig(os.path.join(ASSET_DIR, f"{name}.svg"), transparent=True)
    fig.savefig(os.path.join(ASSET_DIR, f"{name}.png"), transparent=True,
                dpi=300)
    plt.close(fig)
    print(f"  wrote assets/{name}.svg + {name}.png (300 dpi)")


def blank_ax(fig, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    return ax


def box(ax, x, y, w, h, label, fc, ec=None, tc="white", fs=20, lw=2,
        rounding=0.08):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold", zorder=5)


def arrow(ax, x0, y0, x1, y1, color="#444444", lw=3.2, style="-|>", ms=22):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=ms, color=color, lw=lw,
                 shrinkA=0, shrinkB=0, zorder=1))


# =====================================================================
# SLIDE 2 (left): curvature sensing cartoon
# =====================================================================
def curvature_cartoon():
    fig = plt.figure(figsize=(9.2, 6.2))
    ax = blank_ax(fig)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 6.2)

    def liposome(cx, cy, r, n_prot, label):
        # membrane
        ax.add_patch(Circle((cx, cy), r, fc="#eaf4ff", ec=C.blue, lw=5,
                            zorder=2))
        # bound proteins as green BAR "bananas" tangent to the membrane
        for k in range(n_prot):
            th = 2 * np.pi * k / n_prot
            px, py = cx + r * np.cos(th), cy + r * np.sin(th)
            ax.add_patch(Ellipse((px, py), 0.42, 0.17,
                         angle=np.degrees(th) + 90,
                         fc=C.green, ec="#04111a", lw=1.0, zorder=4))
        ax.text(cx, cy - r - 0.55, label, ha="center", va="top", fontsize=20,
                color=C.black, fontweight="bold")

    liposome(2.15, 3.4, 0.85, 14, "small, highly curved\n→ densely bound")
    liposome(6.5, 3.4, 1.6, 7, "large, ~flat\n→ sparsely bound")

    # protein legend glyph
    ax.add_patch(Ellipse((1.0, 5.7), 0.42, 0.17, angle=20, fc=C.green,
                 ec="#04111a", lw=1.0))
    ax.text(1.35, 5.7, "= curvature-sensing protein", va="center",
            fontsize=18, color=C.black)
    save_asset(fig, "curvature_sensing")


# =====================================================================
# SLIDE 9: architecture comparison
# =====================================================================
def architectures():
    fig = plt.figure(figsize=(15, 7.4))
    ax = blank_ax(fig)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.4)

    # ---- left: U-Net hourglass (pooling) ----
    ax.text(3.6, 7.0, "U-Net (today)", ha="center", fontsize=24,
            fontweight="bold", color=C.vermillion)
    # encoder boxes shrinking, decoder growing -> hourglass
    heights = [2.4, 1.6, 0.9, 1.6, 2.4]
    xs = [0.6, 1.9, 3.2, 4.5, 5.8]
    for x, h in zip(xs, heights):
        ax.add_patch(Rectangle((x, 3.7 - h / 2), 1.0, h, fc="#fbe3d4",
                     ec=C.vermillion, lw=2.5))
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 1.0, 3.7, xs[i + 1], 3.7, color=C.vermillion,
              lw=2.4, ms=16)
    ax.annotate("pooling shrinks the image →\nsmall, dim spots lose precision",
                xy=(3.7, 3.25), xytext=(3.6, 1.5), ha="center", fontsize=17,
                color="#9a3a00",
                arrowprops=dict(arrowstyle="->", color=C.vermillion, lw=2))

    # divider
    ax.plot([7.4, 7.4], [0.7, 6.6], color=C.lightgrey, lw=2, ls="--")

    # ---- right: resolution-preserving parallel streams ----
    ax.text(11.2, 7.0, "Resolution-preserving (to test)", ha="center",
            fontsize=24, fontweight="bold", color=C.green)
    for j, yy in enumerate((4.7, 3.7, 2.7)):
        ax.add_patch(FancyBboxPatch((8.2, yy - 0.32), 5.6, 0.64,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     fc="#dff3ec", ec=C.green, lw=2.4))
        arrow(ax, 13.8, yy, 14.4, yy, color=C.green, lw=2.2, ms=14)
    ax.text(11.0, 4.7, "full resolution kept throughout", ha="center",
            va="center", fontsize=16, color="#0b5c43")
    ax.text(11.0, 1.75, "DeepLabv3+  ·  HRNet  ·  Transformer", ha="center",
            fontsize=19, color=C.green, fontweight="bold")

    # ---- bottom strip: balanced data ----
    box(ax, 0.6, 0.05, 13.8, 1.15,
        "Balanced training data: curvature α randomized per spot\n"
        "→ no learnable prior for the network to imprint",
        fc=C.blue, fs=17)
    save_asset(fig, "architectures")


# =====================================================================
# SLIDE 10: DLS size-prior conditioning
# =====================================================================
def dls_conditioning():
    fig = plt.figure(figsize=(15, 6.0))
    ax = blank_ax(fig)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.0)

    def mini_hist(x0, y0, w, h, peak, color, title):
        d = np.linspace(0, 1, 24)
        y = np.exp(-((d - peak) ** 2) / (2 * 0.12 ** 2))
        y = y / y.max() * h
        bw = w / len(d)
        for i, yi in enumerate(y):
            ax.add_patch(Rectangle((x0 + i * bw, y0), bw * 0.9, yi, fc=color,
                         ec="none"))
        ax.plot([x0, x0 + w], [y0, y0], color=C.black, lw=2)
        ax.plot([x0, x0], [y0, y0 + h], color=C.black, lw=2)
        ax.text(x0 + w / 2, y0 + h + 0.22, title, ha="center", fontsize=18,
                color=C.black)
        ax.text(x0 + w / 2, y0 - 0.32, "diameter d", ha="center", fontsize=14,
                color=C.grey)

    mini_hist(0.5, 2.4, 3.0, 1.7, 0.32, C.skyblue, "Simulated  N(d)\n(number)")
    arrow(ax, 3.9, 3.2, 5.2, 3.2)
    ax.text(4.55, 3.7, r"$\times\, d^{6}$", ha="center", fontsize=26,
            color=C.vermillion, fontweight="bold")
    ax.text(4.55, 2.65, "Rayleigh\nweighting", ha="center", fontsize=14,
            color=C.grey)

    mini_hist(5.4, 2.4, 3.0, 1.7, 0.66, C.blue,
              "DLS-matched\n(intensity-weighted)")
    arrow(ax, 8.8, 3.2, 10.1, 3.2)

    box(ax, 10.2, 2.1, 4.3, 2.3,
        "Detector\n(size prior via\nFiLM  or  concat)", fc=C.green, fs=21)
    arrow(ax, 12.35, 2.1, 12.35, 1.2, color=C.green)
    ax.text(12.35, 0.95, "knows expected sizes", ha="center", va="top",
            fontsize=16, color="#0b5c43")

    ax.text(7.5, 5.4, r"DLS reports $I(d)\;\propto\;N(d)\,\cdot\,d^{6}$",
            ha="center", fontsize=24, color=C.black)
    save_asset(fig, "dls_conditioning")


# =====================================================================
# SLIDE 11: full pipeline + Spotiflow benchmark
# =====================================================================
def pipeline():
    fig = plt.figure(figsize=(15.5, 6.4))
    ax = blank_ax(fig)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.4)

    stages = [
        ("Calibrate\nmicroscope", C.blue),
        ("Generate\nlabeled images", C.skyblue),
        ("Train\ndetector", C.green),
        ("Detect on\nreal images", C.orange),
        (r"$\alpha\ \pm$ error bar", C.vermillion),
    ]
    w, h, gap = 2.5, 1.7, 0.45
    x = 0.4
    y = 3.6
    centers = []
    for label, color in stages:
        box(ax, x, y, w, h, label, fc=color, fs=20)
        centers.append(x + w / 2)
        x += w + gap
    for i in range(len(stages) - 1):
        arrow(ax, centers[i] + w / 2, y + h / 2,
              centers[i + 1] - w / 2, y + h / 2)

    # green check that calibration is DONE
    ax.text(centers[0], y - 0.45, "✓ done", ha="center", fontsize=18,
            color=C.green, fontweight="bold")
    ax.text((centers[1] + centers[2]) / 2, y - 0.45, "in progress",
            ha="center", fontsize=16, color=C.grey)

    # Spotiflow benchmark callout under "Detect"
    box(ax, centers[3] - 2.1, 1.05, 4.2, 1.25,
        "Benchmark vs Spotiflow\n(under-detects our small liposomes)",
        fc="#555555", fs=15)
    arrow(ax, centers[3], y, centers[3], 2.3, color="#555555", lw=2.4, ms=16)

    # end-goal banner
    box(ax, 0.4, 5.1, 14.7, 0.95,
        "Goal: unbiased curvature number α with uncertainty  +  "
        "a reusable recipe any lab can apply", fc=C.black, fs=18)
    save_asset(fig, "pipeline")


# =====================================================================
# NEW SLIDE A: how the SLiC measurement works (real-data pipeline)
# =====================================================================
def slic_measurement():
    fig = plt.figure(figsize=(16, 5.0))
    ax = blank_ax(fig)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)

    yc = 3.0
    bh = 1.6

    # --- step 1: two-channel confocal image icon -----------------------
    ix, iy, isz = 0.5, yc - 0.9, 1.8
    ax.add_patch(Rectangle((ix, iy), isz, isz, fc="#0d0018", ec=C.black, lw=2))
    rng = np.random.default_rng(7)
    for _ in range(46):
        px = ix + rng.uniform(0.14, isz - 0.14)
        py = iy + rng.uniform(0.14, isz - 0.14)
        col = C.blue if rng.random() < 0.5 else C.green
        ax.add_patch(Circle((px, py), 0.05, fc=col, ec="none", alpha=0.95))
    ax.text(ix + isz / 2, iy + isz + 0.26, "Two-channel\nconfocal image",
            ha="center", va="bottom", fontsize=17, fontweight="bold",
            color=C.black)
    ax.add_patch(Circle((ix + 0.18, iy - 0.34), 0.085, fc=C.blue, ec="none"))
    ax.text(ix + 0.36, iy - 0.34, "lipid  →  size", va="center", fontsize=13,
            color=C.black)
    ax.add_patch(Circle((ix + 0.18, iy - 0.72), 0.085, fc=C.green, ec="none"))
    ax.text(ix + 0.36, iy - 0.72, "protein  →  binding", va="center",
            fontsize=13, color=C.black)

    # --- step 2: detect each liposome ----------------------------------
    arrow(ax, 2.45, yc, 3.15, yc)
    box(ax, 3.2, yc - bh / 2, 2.3, bh, "Detect each\nliposome",
        fc=C.blue, fs=18)

    # --- step 3: per-vesicle measurement -------------------------------
    arrow(ax, 5.55, yc, 6.2, yc)
    box(ax, 6.25, yc - bh / 2 - 0.15, 4.0, bh + 0.3,
        "Per vesicle:\nlipid intensity ≈ size\nprotein intensity ≈ amount bound",
        fc=C.skyblue, fs=15)

    # --- step 4: log-log sorting curve -> slope = alpha ----------------
    arrow(ax, 10.3, yc, 11.0, yc)
    px0, py0, pw, ph = 11.5, 1.55, 3.9, 2.9
    ax.plot([px0, px0], [py0, py0 + ph], color=C.black, lw=2.2)
    ax.plot([px0, px0 + pw], [py0, py0], color=C.black, lw=2.2)
    pr = np.random.default_rng(2)
    t = pr.uniform(0, 1, 130)
    sx = px0 + 0.35 + t * (pw - 0.6)
    sy = py0 + ph - 0.45 - t * (ph - 0.9) + pr.normal(0, 0.16, t.size)
    ax.scatter(sx, sy, s=10, color=C.blue, alpha=0.28, edgecolor="none",
               zorder=2)
    ax.plot([px0 + 0.35, px0 + pw - 0.35], [py0 + ph - 0.45, py0 + 0.55],
            color=C.vermillion, lw=4, zorder=3)
    # annotation in the empty top-right corner (line runs top-left→bottom-right)
    ax.text(px0 + pw - 0.2, py0 + ph - 0.3, "slope = α − 2", ha="right",
            va="top", fontsize=20, color=C.vermillion, fontweight="bold")
    ax.text(px0 + pw / 2, py0 - 0.32, "liposome size  (log)", ha="center",
            va="top", fontsize=13, color=C.grey)
    ax.text(px0 - 0.28, py0 + ph / 2, "protein density  (log)", ha="center",
            va="center", rotation=90, fontsize=13, color=C.grey)
    ax.text(px0 + pw / 2, py0 + ph + 0.22, "Sorting curve", ha="center",
            va="bottom", fontsize=16, fontweight="bold", color=C.black)
    save_asset(fig, "slic_measurement")


# =====================================================================
# NEW SLIDE B: the forward model (physics chain that GENERATES an image)
# =====================================================================
def forward_model():
    fig = plt.figure(figsize=(16.5, 5.2))
    ax = blank_ax(fig)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 5.2)

    yc = 3.05
    h = 2.35
    y0 = yc - h / 2

    steps = [
        ("Sample sizes\n+ place at\ncalibrated density", C.skyblue, 2.7, 16),
        ("Assign intensities\nlipid $\\propto d^{2}$\nprotein $\\propto d^{\\alpha}$",
         C.blue, 2.7, 16),
        ("Optical blur\n(PSF)", C.orange, 2.4, 17),
        ("Detector physics:\nPoisson shot noise →\ngain × excess-noise →\n"
         "read noise + bias →\nclip to 12-bit", "#555555", 3.1, 13.5),
    ]
    x = 0.45
    gap = 0.42
    centers = []
    for label, color, w, fs in steps:
        box(ax, x, y0, w, h, label, fc=color, fs=fs)
        centers.append((x, x + w))
        x += w + gap

    # payoff box (emphasized): realistic image WITH known ground truth
    pw = 3.0
    arrow(ax, centers[-1][1], yc, x, yc)
    ax.add_patch(FancyBboxPatch((x - 0.12, y0 - 0.12), pw + 0.24, h + 0.24,
                 boxstyle="round,pad=0.02,rounding_size=0.1",
                 fc="#d7f0e6", ec=C.green, lw=2.5, zorder=0))
    box(ax, x, y0, pw, h,
        "Realistic image\n+ known\nground truth", fc=C.green, fs=18)
    ax.text(x + pw / 2, y0 - 0.34, "← why we can validate", ha="center",
            va="top", fontsize=14, color="#0b5c43", fontweight="bold")

    # arrows between process steps
    for i in range(len(centers) - 1):
        arrow(ax, centers[i][1], yc, centers[i + 1][0], yc)

    # trustworthy-parameters footnote
    ax.text(centers[0][0], 0.42,
            "Calibrated, trustworthy parameters:  "
            "PSF widths · lipid brightness · spot density",
            ha="left", va="center", fontsize=14, color=C.black, style="italic")
    save_asset(fig, "forward_model")


curvature_cartoon()
architectures()
dls_conditioning()
pipeline()
slic_measurement()
forward_model()
print("done: schematics")
