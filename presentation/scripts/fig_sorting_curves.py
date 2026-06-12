"""Slide 2 (right): one clean endophilin sorting curve with alpha annotated.
Slide 6: the same measurement, CMEAnalysis vs our U-Net detector.

Data: archive puncta files (source_image, A_lipid, A_protein).
Math: sorting.compute_curve (ported from archive make_figure2.py).
"""
import numpy as np

from style import apply_style, save, C
import matplotlib.pyplot as plt

from sorting import load_puncta, compute_curve
from archive_paths import PUNCTA, DLS_MEAN

apply_style()


def _density_hexbin(ax, R, density, color):
    """Faint density scatter on log-log axes."""
    ax.scatter(R, density, s=6, alpha=0.05, color=color, rasterized=True,
               edgecolor="none")


def _fit_line(ax, c, color, label):
    Rl = np.array([c["R_min"], c["R_max"]])
    yl = 10 ** (c["slope"] * np.log10(Rl) + c["intercept"])
    ax.plot(Rl, yl, color=color, lw=4, label=label, zorder=5)


# ==================================================== SLIDE 2: single curve
def single_curve():
    A_l, A_p = load_puncta(PUNCTA[("v3", "25nM_endophilin")])
    c = compute_curve(A_l, A_p, dls_mean=DLS_MEAN["25nM_endophilin"])
    print(f"  endophilin (U-Net): slope={c['slope']:.2f} alpha={c['alpha']:.2f} "
          f"r2={c['r2']:.2f} n={c['n']}")

    fig, ax = plt.subplots(figsize=(11, 8))
    _density_hexbin(ax, c["R"], c["density"], C.skyblue)
    _fit_line(ax, c, C.vermillion,
              f"slope = {c['slope']:.2f}\n" rf"$\alpha$ = {c['alpha']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Tight limits clipped to ~the 1st-99th percentile of the plotted points so
    # the cloud + fit line fill the frame instead of hugging the bottom-left.
    Rlo, Rhi = np.percentile(c["R"], [1, 99])
    dlo, dhi = np.percentile(c["density"], [1, 99])
    ax.set_xlim(Rlo * 0.92, Rhi * 1.12)
    ax.set_ylim(dlo * 0.85, dhi * 1.7)
    ax.set_xlabel("Liposome radius  (nm)")
    ax.set_ylabel(r"Protein density   $A_{\rm prot}/4\pi R^2$")
    ax.set_title("Endophilin sorting curve (real data)", fontsize=26)
    # slope/alpha legend top-right; with tight limits the fit line now sits in
    # the mid/lower band, so the alpha-meaning note goes in the empty
    # bottom-left corner — clear of both the fit line and the legend.
    ax.legend(loc="upper right", fontsize=24,
              handlelength=1.4, borderpad=0.6)
    ax.text(0.03, 0.04,
            r"$\alpha<2$: binds curved" "\n" r"(small) liposomes more",
            transform=ax.transAxes, fontsize=18, color=C.black,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e0", ec=C.accent,
                      lw=1.5))
    ax.grid(True, which="both", alpha=0.25)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.91, bottom=0.12)
    return fig


save(single_curve(), "sorting_curve_endophilin.png")


# ==================================================== SLIDE 6: CME vs U-Net
def compare_curves():
    specs = [("cme", "CMEAnalysis\n(standard tool)", C.cme),
             ("v3", "Our U-Net detector", C.unet)]
    fits = {}
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), sharey=True)
    for ax, (tag, title, color) in zip(axes, specs):
        A_l, A_p = load_puncta(PUNCTA[(tag, "25nM_endophilin")])
        c = compute_curve(A_l, A_p, dls_mean=DLS_MEAN["25nM_endophilin"])
        fits[tag] = c
        _density_hexbin(ax, c["R"], c["density"], color)
        _fit_line(ax, c, C.vermillion, None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(20, 100)
        ax.set_xlabel("Liposome radius  (nm)")
        ax.set_title(title, fontsize=23, color=color)
        ax.grid(True, which="both", alpha=0.25)
        ax.text(0.04, 0.05,
                f"slope = {c['slope']:.2f}\n"
                rf"$\alpha$ = {c['alpha']:.2f}" "\n"
                f"$r^2$ = {c['r2']:.2f}\n"
                f"$n$ = {c['n']:,}",
                transform=ax.transAxes, fontsize=20, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color,
                          lw=1.8))
    axes[0].set_ylabel(r"Protein density")
    # headline improvement factors
    r2gain = fits["v3"]["r2"] / fits["cme"]["r2"]
    ngain = fits["v3"]["n"] / fits["cme"]["n"]
    print(f"  r2 gain x{r2gain:.1f}, spot gain x{ngain:.1f}")
    # No suptitle: the slide title + an explicit r2 headline carry this on the
    # deck, so a baked-in title would just duplicate them.
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.13,
                        wspace=0.08)
    return fig, r2gain, ngain


fig6, r2gain, ngain = compare_curves()
save(fig6, "sorting_curve_compare.png")
print(f"done: slides 2 & 6  (r2 x{r2gain:.1f}, spots x{ngain:.1f})")
