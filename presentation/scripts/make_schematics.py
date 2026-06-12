"""Schematics / cartoons for slides 2, 9, 10, 11 -> presentation/assets/.

Built with matplotlib patches for precise placement and palette consistency,
saved as scalable SVG (crisp in print-PDF) plus a PNG fallback. No data, no
pipeline code; pure illustration.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (Circle, Ellipse, FancyBboxPatch,
                                FancyArrowPatch, Rectangle, Wedge)
import tifffile

from style import apply_style, ASSET_DIR, REPO_ROOT, C

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

    # Box labels carry the STRUCTURE only; the crisp LaTeX equation objects
    # (rendered separately, placed under each box on the slide) carry the math.
    steps = [
        ("Sample sizes\n+ place at\ncalibrated density", C.skyblue, 2.7, 16),
        ("Assign\nintensities", C.blue, 2.7, 18),
        ("Optical blur\n(PSF)", C.orange, 2.4, 17),
        ("Detector\nphysics", "#555555", 3.1, 18),
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

    save_asset(fig, "forward_model")


# =====================================================================
# NEW SLIDE 3: SLiC real-data flow  (image -> detections -> linear -> log)
# Built from a REAL synthetic image with KNOWN alpha + its ground truth.
# =====================================================================
def slic_flow():
    from sorting import compute_curve

    aset = "real_cmp_alpha_1p25"
    true_alpha = 1.25
    img_id = "000000"
    base = os.path.join(REPO_ROOT, "external_export", aset)

    im = tifffile.imread(os.path.join(base, "tiffs", f"img_{img_id}.tif"))
    prot, lip = im[0].astype(float), im[1].astype(float)   # ch0=protein, ch1=lipid

    xs, ys, Lf, Pf = [], [], [], []
    with open(os.path.join(base, "ground_truth.csv")) as f:
        for row in csv.DictReader(f):
            if row["image_id"] == img_id:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
                Lf.append(float(row["lipid_flux"]))
                Pf.append(float(row["protein_flux"]))
    xs, ys = np.array(xs), np.array(ys)
    Lf, Pf = np.array(Lf), np.array(Pf)

    # Same definition as the deck's sorting curve (sorting.compute_curve):
    #   R (size) ∝ sqrt(lipid_flux) ∝ d ;  density = protein/(4 pi R^2) ∝ d^(alpha-2)
    #   => log-log slope = alpha - 2  (consistent with slide 2).
    c = compute_curve(Lf, Pf)
    print(f"  slic_flow: true_alpha={true_alpha} slope={c['slope']:.3f} "
          f"(=alpha-2) recovered_alpha={c['alpha']:.3f} r2={c['r2']:.3f}")

    # protein = green, lipid = purple/magenta  (distinct, fluorescence look)
    cmap_prot = LinearSegmentedColormap.from_list(
        "prot", ["#02150c", "#0a8f54", "#86ffbf"])
    cmap_lip = LinearSegmentedColormap.from_list(
        "lip", ["#190019", "#a81f95", "#ff9be8"])
    PROT_C, LIP_C = "#0a8f54", "#b3199c"

    def vlim(a):
        return tuple(np.percentile(a, [1.0, 99.5]))
    pv, lv = vlim(prot), vlim(lip)

    fig = plt.figure(figsize=(18, 5.0))
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_axis_off()
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)

    # ---- panel 1: two channels stacked (kept clear of the header) ----
    axp = fig.add_axes([0.025, 0.45, 0.135, 0.37])
    axp.imshow(prot, cmap=cmap_prot, vmin=pv[0], vmax=pv[1])
    axp.set_axis_off()
    axp.set_title("protein", color=PROT_C, fontsize=18, fontweight="bold",
                  pad=3)
    axl = fig.add_axes([0.025, 0.05, 0.135, 0.37])
    axl.imshow(lip, cmap=cmap_lip, vmin=lv[0], vmax=lv[1])
    axl.set_axis_off()
    axl.set_title("lipid", color=LIP_C, fontsize=18, fontweight="bold", pad=3)

    # ---- panel 2: detections on lipid --------------------------------
    ax2 = fig.add_axes([0.255, 0.10, 0.20, 0.78])
    ax2.imshow(lip, cmap=cmap_lip, vmin=lv[0], vmax=lv[1])
    ax2.scatter(xs, ys, s=24, facecolors="none", edgecolors="#f2ff45",
                lw=0.7, alpha=0.9)
    ax2.set_xlim(0, lip.shape[1])
    ax2.set_ylim(lip.shape[0], 0)
    ax2.set_axis_off()
    ax2.set_title("detect on lipid", fontsize=18, color=C.black,
                  fontweight="bold", pad=6)

    # ---- panel 3: linear axes ----------------------------------------
    R, dens = c["R"], c["density"]
    xhi, yhi = np.percentile(R, 99), np.percentile(dens, 99)
    ax3 = fig.add_axes([0.555, 0.18, 0.165, 0.62])
    ax3.scatter(R, dens, s=10, color=C.blue, alpha=0.20, edgecolor="none")
    rr = np.linspace(max(R.min(), 1), xhi, 120)
    ax3.plot(rr, 10 ** (c["slope"] * np.log10(rr) + c["intercept"]),
             color=C.grey, lw=2.2, ls="--", alpha=0.8)
    ax3.set_xlim(0, xhi * 1.05)
    ax3.set_ylim(0, yhi * 1.12)
    ax3.set_title("linear axes", fontsize=18, fontweight="bold")
    ax3.set_xlabel("liposome size  (from lipid)", fontsize=14)
    ax3.set_ylabel("protein density", fontsize=14)
    ax3.tick_params(labelsize=11)

    # ---- panel 4: log-log axes ---------------------------------------
    ax4 = fig.add_axes([0.805, 0.18, 0.165, 0.62])
    ax4.scatter(R, dens, s=10, color=C.blue, alpha=0.20, edgecolor="none")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    Rl = np.array([c["R_min"], c["R_max"]])
    ax4.plot(Rl, 10 ** (c["slope"] * np.log10(Rl) + c["intercept"]),
             color=C.vermillion, lw=4, zorder=5)
    ax4.set_xlim(np.percentile(R, 1) * 0.9, np.percentile(R, 99) * 1.1)
    ax4.set_ylim(np.percentile(dens, 1) * 0.7, np.percentile(dens, 99) * 1.6)
    ax4.set_title("log–log axes  →  slope = α − 2", fontsize=18,
                  fontweight="bold")
    ax4.set_xlabel("liposome size  (from lipid)", fontsize=14)
    # explicit, sparse ticks so the log labels don't crowd at this panel width
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax4.xaxis.set_major_locator(FixedLocator([20, 40, 80]))
    ax4.xaxis.set_minor_locator(NullLocator())
    ax4.xaxis.set_major_formatter(ScalarFormatter())
    ax4.tick_params(labelsize=11)
    # annotation in the empty upper-right (line runs top-left → bottom-right)
    ax4.text(0.96, 0.96,
             f"slope = α − 2 = {c['slope']:.2f}\n"
             rf"recovered $\alpha$ = {c['alpha']:.2f}  (true {true_alpha:.2f})",
             transform=ax4.transAxes, fontsize=13.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C.vermillion,
                       lw=1.6))

    # header: we KNOW the answer (synthetic)
    ov.text(0.025, 0.97, f"synthetic data — true α = {true_alpha:.2f}",
            fontsize=17, color=C.black, fontweight="bold", va="top")

    # arrows between panels
    for x0, x1 in [(0.175, 0.245), (0.470, 0.545), (0.730, 0.795)]:
        ov.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>",
                     mutation_scale=22, color="#444444", lw=3.0,
                     shrinkA=0, shrinkB=0))
    save_asset(fig, "slic_flow")


# =====================================================================
# Forward-model equations rendered as crisp transparent LaTeX-style PNGs.
# Placed as SEPARATE objects on slide 5 so the presenter can animate a
# one-by-one build-up by hand (python-pptx cannot set animations).
# =====================================================================
def _save_eq(fig, name):
    out = os.path.join(ASSET_DIR, f"{name}.png")
    fig.savefig(out, transparent=True, dpi=300, bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote assets/{name}.png (equation)")


def render_equations():
    # HERO: assign intensities. protein∝d^α is the physics the talk measures,
    # so it is larger + accent-coloured.
    fig = plt.figure(figsize=(5, 2.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.74, r"$A_{\mathrm{lipid}}\ \propto\ d^{2}$",
            ha="center", va="center", fontsize=30, color=C.black)
    ax.text(0.5, 0.26, r"$A_{\mathrm{protein}}\ \propto\ d^{\alpha}$",
            ha="center", va="center", fontsize=42, color=C.vermillion,
            fontweight="bold")
    _save_eq(fig, "eq_assign")

    # optical blur: convolution with the PSF
    fig = plt.figure(figsize=(5, 1.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.5, r"$I\ =\ \mathrm{signal}\ \ast\ \mathrm{PSF}$",
            ha="center", va="center", fontsize=30, color=C.black)
    _save_eq(fig, "eq_blur")

    # detector physics: one compact line + tiny degeneracy caption
    fig = plt.figure(figsize=(7, 2.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.80,
            r"$I\ \sim\ \mathrm{clip}\!\left(g\,F\cdot\mathrm{Pois}"
            r"(S{+}D{+}C)\ +\ \mathcal{N}(b,\sigma_{\mathrm{read}}^{2})\right)$",
            ha="center", va="center", fontsize=25, color=C.black)
    ax.text(0.5, 0.36, r"$g\!\cdot\!F$ = gain × excess-noise",
            ha="center", va="center", fontsize=18, color="#555555")
    ax.text(0.5, 0.13, "(degenerate — only the product is constrained)",
            ha="center", va="center", fontsize=15, color="#555555")
    _save_eq(fig, "eq_detector")


curvature_cartoon()
architectures()
dls_conditioning()
pipeline()
slic_measurement()
forward_model()
slic_flow()
render_equations()
print("done: schematics")
