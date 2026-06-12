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

# ---- tasteful, colorblind-friendly styling -------------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})

# Okabe-Ito colorblind-safe palette
C_OURS = "#0072B2"        # blue
C_CLASSICAL = "#D55E00"   # vermillion
C_EGFP = "#009E73"        # green
C_ENDO = "#CC79A7"        # purple
C_TRUTH = "#555555"       # grey reference lines


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
        ("300nM_EGFP", "EGFP (negative control)", C_EGFP),
        ("300nM_endophilin", "Endophilin (curvature sensor)", C_ENDO),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True)
    for ax, (sample, title, color) in zip(axes, panels):
        sub = [r for r in rows if r["sample"] == sample]
        lx = to_float([r["log_lipid"] for r in sub])
        ly = to_float([r["log_protein"] for r in sub])
        # OLS fit of log(protein) vs log(lipid); alpha = 2 * slope
        slope, intercept = np.polyfit(lx, ly, 1)
        alpha = 2.0 * slope

        ax.scatter(np.exp(lx), np.exp(ly), s=4, alpha=0.12, color=color,
                   edgecolors="none", rasterized=True)
        xs = np.linspace(lx.min(), lx.max(), 100)
        ax.plot(np.exp(xs), np.exp(intercept + slope * xs),
                color="black", lw=2.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title}\nslope ≈ {slope:.2f}   (α ≈ {alpha:.2f})")
        ax.set_xlabel("lipid fluorescence  (liposome size →)")
        ax.text(0.04, 0.94, f"n = {len(sub):,} liposomes",
                transform=ax.transAxes, va="top", fontsize=10, color="#333")
    axes[0].set_ylabel("protein fluorescence  (amount bound →)")
    fig.suptitle("How much protein binds vs. liposome size, per detected liposome",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005,
             "α = 2×(slope of log protein vs log lipid).  α≈2 = binding tracks "
             "membrane area (no preference); α<2 = preference for small, high-curvature "
             "liposomes.\nEGFP is steeper (near area-proportional); endophilin is much "
             "shallower — the signature of curvature sensing.  Native photometry, "
             "pre gain-correction (see Fig. 3).",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(FIGS, "fig1_measurement.png"))
    plt.close(fig)
    print(f"fig1: EGFP slope/alpha, endophilin slope/alpha written")


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

    def style_diam_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks([40, 60, 90, 120, 160, 220, 300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(38, 320)
        ax.set_xlabel("liposome diameter (nm)")

    # (a) detection fraction (F1)
    ax1.plot(x, series("ours", "f1"), "-o", color=C_OURS, lw=2.4, ms=7,
             label="our method")
    ax1.plot(x, series("classical", "f1"), "-s", color=C_CLASSICAL, lw=2.4,
             ms=7, label="standard method")
    ax1.axvspan(40, 90, color="#FFE9B0", alpha=0.5, zorder=0)
    ax1.text(63, 0.04, "small,\nhigh-curvature", ha="center", fontsize=9,
             color="#8a6d00")
    style_diam_axis(ax1)
    ax1.set_ylabel("fraction correctly detected")
    ax1.set_title("(a) We detect the small liposomes")
    ax1.set_ylim(0, 0.9)
    ax1.legend(loc="lower right")

    # (b) lipid-size accuracy (lower = better)
    ax2.plot(x, series("ours", "lipid_logerr"), "-o", color=C_OURS, lw=2.4,
             ms=7, label="our method")
    ax2.plot(x, series("classical", "lipid_logerr"), "-s", color=C_CLASSICAL,
             lw=2.4, ms=7, label="standard method")
    ax2.axvspan(40, 90, color="#FFE9B0", alpha=0.5, zorder=0)
    style_diam_axis(ax2)
    ax2.set_ylabel("size error  (lower is better)")
    ax2.set_title("(b) ...and size them accurately")
    ax2.legend(loc="upper right")

    fig.suptitle("Small liposomes: where curvature sensing happens, and where the "
                 "standard method fails",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.02,
             "At 40–90 nm the standard automated method detects a small minority "
             "and mis-sizes them badly; our detector finds most of them and sizes them "
             "accurately,\nso the curvature-sensing regime is measured representatively.",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
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

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) EGFP recovered alpha vs concentration
    xpos = np.arange(len(conc))
    axa.errorbar(xpos, alpha, yerr=[alpha - lo, hi - alpha], fmt="-o",
                 color=C_EGFP, lw=2.4, ms=8, capsize=4)
    axa.axhline(2.0, ls="--", color=C_TRUTH, lw=2)
    axa.text(xpos[-1], 2.0, "  α = 2  (known truth for EGFP)",
             va="bottom", ha="right", color=C_TRUTH, fontsize=10)
    axa.set_xticks(xpos)
    axa.set_xticklabels([f"{c} nM" for c in conc])
    axa.set_xlabel("EGFP concentration")
    axa.set_ylabel("recovered α")
    axa.set_title("(a) The control isn't flat at α = 2 yet")
    axa.set_ylim(1.0, 2.15)

    # (b) the explanation: lipid-channel PMT voltage table
    axb.axis("off")
    axb.set_title("(b) Why: the lipid detector voltage changed")
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
        bbox=[0.08, 0.40, 0.84, 0.50],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#eef3f7")
            cell.set_text_props(weight="bold")
    axb.text(0.5, 0.26,
             "The 488 nm protein detector was held constant (295 V),\n"
             "but the operator lowered the 561 nm lipid detector at\n"
             "higher concentrations to avoid saturation. PMT gain is\n"
             "nonlinear in voltage, so the size axis is scaled\n"
             "differently per sample — an acquisition setting,\n"
             "not biology. Fix in progress: gain-normalize the lipid\n"
             "channel across samples before pooling.",
             ha="center", va="top", transform=axb.transAxes, fontsize=9.5,
             color="#333")

    fig.suptitle("The EGFP control caught a subtle acquisition artifact",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.02,
             "EGFP has no curvature preference, so it must read α = 2 at every "
             "concentration. It doesn't yet — α rises with concentration, and the trend "
             "tracks the\nlipid-detector voltage. This is a correctable calibration issue, "
             "not a flaw in the method or the biology.",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
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

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    lim = [0.4, 2.3]
    ax.plot(lim, lim, ls="--", color=C_TRUTH, lw=1.8, label="perfect recovery")
    ax.errorbar(true, rec, yerr=[rec - lo, hi - rec], fmt="o", color=C_OURS,
                ms=7, lw=1.8, capsize=3, label="our method")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("true α  (set in simulation)")
    ax.set_ylabel("recovered α  (measured)")
    ax.set_title("Validation on known-answer simulated data")
    ax.legend(loc="upper left")
    fig.text(0.5, 0.015,
             "On simulated liposomes where the true α is known, the method\n"
             "recovers it across the full range — this is why we trust the\n"
             "measurements on real data.",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(FIGS, "fig4_synthetic_validation.png"))
    plt.close(fig)
    print("fig4: synthetic recovery curve written")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nAll figures written to:", FIGS)
