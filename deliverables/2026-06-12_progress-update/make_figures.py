#!/usr/bin/env python3
"""Build the 2026-06-12 advisor progress-update figures from saved CSVs.

CPU only, no inference, no detector runs. Reads the self-contained snapshot in
figure_data/ and writes PNGs to figures/. Re-runnable:

    python make_figures.py

Dependencies: numpy, matplotlib (no pandas — uses the stdlib csv reader).
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "figure_data")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# ---- house style: clean, professional, colorblind-safe -------------------
plt.rcParams.update({
    # crisp when projected or zoomed
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # consistent professional sans-serif everywhere
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 13,
    "axes.labelpad": 6,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    # thin axes, no top/right spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    # light, subtle grid behind the data
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#B8B8B8",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "lines.solid_capstyle": "round",
})

# Okabe-Ito colorblind-safe palette — colors are consistent across all figures:
# "our method" is always blue, "standard method" always vermillion,
# EGFP always teal-green, endophilin always purple.
C_OURS = "#0072B2"        # blue        — our method
C_CLASSICAL = "#D55E00"   # vermillion  — standard method
C_EGFP = "#009E73"        # teal-green  — EGFP (control)
C_ENDO = "#9467BD"        # purple      — endophilin (sensor)
C_TRUTH = "#444444"       # grey        — reference lines (truth / perfect)
C_REGIME = "#F6D77A"      # warm sand   — curvature-sensing regime shading
C_FOOT = "#555555"        # caption / footnote grey


def read_csv(name):
    """Return a dict of column-name -> list of strings."""
    path = os.path.join(DATA, name)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    return cols, rows


def to_float(xs):
    return np.array([float(x) for x in xs], dtype=float)


# ==========================================================================
# Figure 1 — the measurement (headline): protein vs lipid, EGFP vs endophilin
# ==========================================================================
def figure1():
    _, rows = read_csv("real_perspot.csv")
    panels = [
        ("300nM_EGFP", "EGFP — negative control", C_EGFP),
        ("300nM_endophilin", "Endophilin — curvature sensor", C_ENDO),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
    for ax, (sample, title, color) in zip(axes, panels):
        sub = [r for r in rows if r["sample"] == sample]
        lx = to_float([r["log_lipid"] for r in sub])
        ly = to_float([r["log_protein"] for r in sub])
        # OLS fit of log(protein) vs log(lipid); alpha = 2 * slope
        slope, intercept = np.polyfit(lx, ly, 1)
        alpha = 2.0 * slope

        ax.scatter(np.exp(lx), np.exp(ly), s=5, alpha=0.10, color=color,
                   edgecolors="none", rasterized=True, zorder=1)
        xs = np.linspace(lx.min(), lx.max(), 100)
        # colored fit line with a black outline so it reads on top of the cloud
        ax.plot(np.exp(xs), np.exp(intercept + slope * xs),
                color="white", lw=5.0, zorder=4, solid_capstyle="round")
        ax.plot(np.exp(xs), np.exp(intercept + slope * xs),
                color=color, lw=3.0, zorder=5, solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, color=color)
        ax.set_xlabel("lipid fluorescence  —  liposome size →")
        # direct, prominent alpha annotation in the empty upper-left corner
        ax.text(0.05, 0.95, f"α ≈ {alpha:.1f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=22, fontweight="bold", color=color)
        ax.text(0.05, 0.80, f"slope ≈ {slope:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, color="#555")
        ax.text(0.97, 0.05, f"n = {len(sub):,} liposomes", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=10.5, color="#666")
        ax.grid(True, which="major", alpha=0.30)
    axes[0].set_ylabel("protein fluorescence  —  amount bound →")
    fig.suptitle("Protein binding vs liposome size:  EGFP (control) vs endophilin "
                 "(curvature sensor)")
    fig.text(0.5, 0.085,
             "α = 2 × slope of (log protein vs log lipid).    α ≈ 2 → binding tracks "
             "membrane area;    α < 2 → preference for small, high-curvature liposomes.",
             ha="center", fontsize=10.5, color=C_FOOT)
    fig.text(0.5, 0.030,
             "EGFP is steep (near area-proportional); endophilin is much shallower — the "
             "signature of curvature sensing.",
             ha="center", fontsize=11, color=C_FOOT)
    fig.text(0.99, 0.006, "native photometry, pre gain-correction (Fig. 3)",
             ha="right", fontsize=8.5, style="italic", color="#999")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(os.path.join(FIGS, "fig1_measurement.png"))
    plt.close(fig)
    print("fig1: EGFP slope/alpha, endophilin slope/alpha written")


# ==========================================================================
# Figure 2 — detection vs diameter (ours vs classical) + intensity accuracy
# ==========================================================================
def _bin_centers(bins):
    centers = []
    for b in bins:
        lo, hi = b.split("-")
        centers.append((float(lo) + float(hi)) / 2.0)
    return np.array(centers)


def figure2():
    _, rows = read_csv("bench_diameter_metrics.csv")
    rows = [r for r in rows if r["sizing"] == "emphasis"]
    order = ["40-55", "55-70", "70-90", "90-120", "120-160", "160-220", "220-300"]

    def series(method, field):
        d = {r["bin"]: float(r[field]) for r in rows if r["method"] == method}
        return np.array([d[b] for b in order])

    x = _bin_centers(order)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    def style_diam_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks([40, 60, 90, 120, 160, 220, 300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(38, 320)
        ax.set_xlabel("liposome diameter (nm)")
        ax.grid(True, axis="y", alpha=0.35)
        ax.grid(False, axis="x")

    def mark_regime(ax, ytext, va):
        ax.axvspan(40, 90, color=C_REGIME, alpha=0.45, zorder=0)
        ax.text(60, ytext, "curvature-sensing\nregime (40–90 nm)", ha="center",
                va=va, fontsize=10, fontweight="bold", color="#9A7A12")

    # (a) detection fraction (F1)
    ax1.plot(x, series("ours", "f1"), "-o", color=C_OURS, lw=2.8, ms=8,
             label="our method", zorder=5)
    ax1.plot(x, series("classical", "f1"), "--s", color=C_CLASSICAL, lw=2.8,
             ms=8, label="standard method", zorder=5)
    mark_regime(ax1, 0.05, "bottom")
    style_diam_axis(ax1)
    ax1.set_ylabel("fraction correctly detected")
    ax1.set_title("(a) Detection: we find the small liposomes")
    ax1.set_ylim(0, 0.9)

    # (b) lipid-size accuracy (lower = better)
    ax2.plot(x, series("ours", "lipid_logerr"), "-o", color=C_OURS, lw=2.8,
             ms=8, label="our method", zorder=5)
    ax2.plot(x, series("classical", "lipid_logerr"), "--s", color=C_CLASSICAL,
             lw=2.8, ms=8, label="standard method", zorder=5)
    mark_regime(ax2, 1.02, "top")
    style_diam_axis(ax2)
    ax2.set_ylabel("size error   (lower is better)")
    ax2.set_title("(b) Sizing: ...and measure their size accurately")

    fig.suptitle("Our detector recovers the small, high-curvature liposomes the "
                 "standard method misses")
    # single shared legend under the title
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=2, frameon=False, fontsize=12.5)
    fig.text(0.5, 0.02,
             "In the 40–90 nm range the standard automated method detects only a small "
             "minority and mis-sizes them badly; our detector finds most of them and "
             "sizes them accurately,\nso the curvature-sensing regime is measured "
             "representatively rather than inferred from larger liposomes.",
             ha="center", fontsize=10.5, color=C_FOOT)
    fig.tight_layout(rect=(0, 0.075, 1, 0.90))
    fig.savefig(os.path.join(FIGS, "fig2_small_liposomes.png"))
    plt.close(fig)
    print("fig2: detection + sizing vs diameter written")


# ==========================================================================
# Figure 3 — EGFP control trend + the acquisition-voltage explanation
# ==========================================================================
def figure3():
    _, rows = read_csv("real_alpha.csv")
    egfp = [r for r in rows if r["sample"].endswith("EGFP")]
    conc = np.array([int(r["sample"].split("nM")[0]) for r in egfp])
    o = np.argsort(conc)
    conc = conc[o]
    alpha = to_float([egfp[i]["alpha_corrected"] for i in o])
    lo = to_float([egfp[i]["cor_lo"] for i in o])
    hi = to_float([egfp[i]["cor_hi"] for i in o])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) EGFP recovered alpha vs concentration
    xpos = np.arange(len(conc))
    # reference line at the expected truth
    axa.axhline(2.0, ls="--", color=C_TRUTH, lw=2, zorder=2)
    axa.text(xpos[0] - 0.15, 2.0, "expected (α = 2)", va="bottom", ha="left",
             color=C_TRUTH, fontsize=12, fontweight="bold")
    axa.errorbar(xpos, alpha, yerr=[alpha - lo, hi - alpha], fmt="-o",
                 color=C_EGFP, lw=2.8, ms=10, capsize=5, capthick=1.8,
                 zorder=5, label="EGFP (measured)")
    # shade the gap between measured and expected to make it obvious
    axa.fill_between(xpos, alpha, 2.0, color=C_EGFP, alpha=0.10, zorder=1)
    axa.annotate("", xy=(xpos[-1] + 0.05, 2.0), xytext=(xpos[-1] + 0.05, alpha[-1]),
                 arrowprops=dict(arrowstyle="<->", color="#999", lw=1.4))
    axa.text(xpos[-1] + 0.12, (alpha[-1] + 2.0) / 2, "gap from\nexpected",
             va="center", ha="left", fontsize=9.5, color="#777")
    axa.set_xticks(xpos)
    axa.set_xticklabels([f"{c} nM" for c in conc])
    axa.set_xlim(-0.4, len(conc) - 0.2)
    axa.set_xlabel("EGFP concentration")
    axa.set_ylabel("recovered α")
    axa.set_title("(a) The control rises with concentration\ninstead of sitting flat at α = 2")
    axa.set_ylim(1.0, 2.18)
    axa.grid(True, axis="y", alpha=0.35)
    axa.grid(False, axis="x")

    # (b) the explanation: lipid-channel PMT voltage table
    axb.axis("off")
    axb.set_title("(b) Why: the lipid detector voltage\nwas lowered per sample")
    table_rows = [
        ("20 nM", "750 V"),
        ("50 nM", "640 V"),
        ("100 nM", "630 V"),
        ("300 nM", "580 V"),
    ]
    tbl = axb.table(
        cellText=table_rows,
        colLabels=["EGFP sample", "561 nm lipid\nPMT voltage"],
        cellLoc="center", colLoc="center",
        bbox=[0.10, 0.46, 0.80, 0.44],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12.5)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#E8EEF3")
            cell.set_text_props(weight="bold", color="#1A3A52")
            cell.set_height(cell.get_height() * 1.25)
        else:
            cell.set_facecolor("#FBFCFD" if r % 2 else "#F2F5F8")
    axb.text(0.5, 0.34,
             "The 488 nm protein detector was held constant (295 V), but the\n"
             "operator lowered the 561 nm lipid detector at higher concentrations\n"
             "to avoid saturation. Detector gain is nonlinear in voltage, so the\n"
             "size axis is scaled differently in each sample — an acquisition\n"
             "setting, not biology.\n\n"
             "Fix in progress: gain-normalize the lipid channel across samples\n"
             "before pooling.",
             ha="center", va="top", transform=axb.transAxes, fontsize=10.5,
             color="#333", linespacing=1.4)

    fig.suptitle("EGFP control should read α = 2 at all concentrations — the trend "
                 "reveals an acquisition artifact")
    fig.text(0.5, 0.015,
             "EGFP has no curvature preference, so it must read α = 2 at every "
             "concentration. It doesn't yet, and the rising trend tracks the lipid-detector "
             "voltage:\nthis is a correctable calibration issue, not a flaw in the method "
             "or the biology — the control did its job and surfaced it.",
             ha="center", fontsize=10.5, color=C_FOOT)
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig3_egfp_artifact.png"))
    plt.close(fig)
    print("fig3: EGFP trend + voltage table written")


# ==========================================================================
# Figure 4 — validation on synthetic known-answer data
# ==========================================================================
def figure4():
    _, rows = read_csv("calibration_curve.csv")
    true = to_float([r["true"] for r in rows])
    rec = to_float([r["recovered"] for r in rows])
    lo = to_float([r["lo95"] for r in rows])
    hi = to_float([r["hi95"] for r in rows])
    o = np.argsort(true)
    true, rec, lo, hi = true[o], rec[o], lo[o], hi[o]

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lim = [0.4, 2.3]
    ax.plot(lim, lim, ls="--", color=C_TRUTH, lw=2.0, zorder=2,
            label="perfect recovery (y = x)")
    ax.errorbar(true, rec, yerr=[rec - lo, hi - rec], fmt="o", color=C_OURS,
                ms=9, lw=2.0, capsize=4, capthick=1.6, zorder=5,
                label="our method (recovered)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("true α   (set in simulation)")
    ax.set_ylabel("recovered α   (measured)")
    ax.set_title("Validated on simulated data with known α")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.30)
    fig.text(0.5, 0.02,
             "On simulated liposomes where the true α is known, the method recovers it\n"
             "across the full range — this is why we trust the measurements on real data.",
             ha="center", fontsize=10.5, color=C_FOOT)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(os.path.join(FIGS, "fig4_synthetic_validation.png"))
    plt.close(fig)
    print("fig4: synthetic recovery curve written")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nAll figures written to:", FIGS)
