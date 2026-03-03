import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_COL  = "#e6edf3"
GRID_COL  = "#30363d"
ACCENT    = "#58a6ff"

C_GS      = "#f78166"   # Gauss-Seidel  – warm red
C_NW      = "#79c0ff"   # Newton        – cool blue
C_HY      = "#56d364"   # Hybrid        – green

# ── Data generation ──────────────────────────────────────────────────────────
N = 300
t = np.linspace(0, 1, N)          # normalised "time"

def residual_gs(t):
    """Fast early drop, slow tail – sublinear near solution."""
    fast  = np.exp(-8 * t)
    slow  = 0.02 * np.exp(-0.8 * (t - 0.3))
    curve = np.where(t < 0.35, fast, slow * (fast[-1] / slow[0]))
    # stitch so the curve is continuous
    join  = fast[t < 0.35][-1]
    tail  = slow[t >= 0.35]
    tail  = tail / tail[0] * join
    out   = np.concatenate([fast[t < 0.35], tail])
    # clip to [0,1] and add tiny jitter for realism
    out   = np.clip(out, 1e-6, 1.0)
    out  *= (1 + 0.015 * np.random.randn(len(out)))
    return np.clip(out, 1e-6, 1.0)

def residual_newton(t):
    """Slow start, then quadratic super-convergence near the end."""
    # Plateau then sharp plunge
    plateau = 0.7 * np.exp(-0.5 * t)
    plunge  = 10 ** (-12 * (t - 0.55) ** 0.5)   # quadratic-like in log space
    blend   = np.where(t < 0.55, plateau, np.minimum(plateau, plunge))
    blend  *= (1 + 0.015 * np.random.randn(N))
    return np.clip(blend, 1e-12, 1.0)

def residual_hybrid(t, t_switch=0.38):
    """GS phase, then switch to Newton super-convergence."""
    gs_phase = residual_gs(t)
    nw_phase = residual_newton(t)

    switch_idx = np.searchsorted(t, t_switch)
    join_val   = gs_phase[switch_idx]

    # Rescale Newton tail so it starts from the GS join value
    nw_tail    = nw_phase[switch_idx:]
    nw_tail    = nw_tail / nw_tail[0] * join_val
    nw_tail   *= (1 + 0.012 * np.random.randn(len(nw_tail)))

    out = np.concatenate([gs_phase[:switch_idx], nw_tail])
    return np.clip(out, 1e-12, 1.0)

r_gs  = residual_gs(t)
r_nw  = residual_newton(t)
r_hy  = residual_hybrid(t)

# Smooth with a small window so jitter doesn't dominate
from scipy.ndimage import uniform_filter1d
smooth = lambda x: uniform_filter1d(x, size=8)
r_gs  = smooth(r_gs)
r_nw  = smooth(r_nw)
r_hy  = smooth(r_hy)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6), facecolor=DARK_BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.15, wspace=0.38)

gs_layout = gridspec.GridSpec(1, 3, figure=fig)

axes_data = [
    (gs_layout[0], r_gs,  C_GS, "Gauss-Seidel",
     "Fast initial drop, slow\ntail near solution",
     "Sublinear (linear) convergence"),
    (gs_layout[1], r_nw,  C_NW, "Newton's Method",
     "Slow start, then explosive\nquadratic convergence",
     "Quadratic (super-linear) convergence"),
    (gs_layout[2], r_hy,  C_HY, "Continuation Hybrid",
     "GS phase → switch to Newton\nfor rapid final convergence",
     "Best of both worlds"),
]

for spec, r, color, title, subtitle, conv_label in axes_data:
    ax = fig.add_subplot(spec)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.6, alpha=0.5)

    # Shaded area under curve
    ax.fill_between(t, r, 1e-12, alpha=0.12, color=color)

    # Main convergence curve
    ax.semilogy(t, r, color=color, linewidth=2.2, zorder=3)

    # Annotate endpoints
    ax.scatter([t[0]], [r[0]], color=color, s=55, zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter([t[-1]], [r[-1]], color=color, s=55, zorder=5,
               marker="*", edgecolors="white", linewidths=0.8)

    # Switch annotation for hybrid
    if title == "Continuation Hybrid":
        sw = np.searchsorted(t, 0.38)
        ax.axvline(t[sw], color="#f0e68c", linewidth=1.2, linestyle=":", alpha=0.8)
        ax.text(t[sw] + 0.02, r[sw] * 3, "Switch\nto Newton",
                color="#f0e68c", fontsize=7.5, va="bottom")

    # Convergence rate label
    ax.text(0.97, 0.96, conv_label,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=color, alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(1e-11, 2)
    ax.set_xlabel("Iteration / Time  →", color=TEXT_COL, fontsize=9)
    ax.set_ylabel("Residual  ‖r‖", color=TEXT_COL, fontsize=9)

    # Titles
    ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, 1.045, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="#8b949e", style="italic")

# ── Super-title ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         "Convergence Behaviour: Gauss-Seidel  ·  Newton  ·  Continuation Hybrid",
         ha="center", va="center", fontsize=15, color=TEXT_COL, fontweight="bold")
fig.text(0.5, 0.915,
         "Residual vs. iteration on a log scale  —  lower is better",
         ha="center", va="center", fontsize=9.5, color="#8b949e")

# out_path = "/mnt/user-data/outputs/convergence_comparison.png"
# plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
# print(f"Saved → {out_path}")
plt.show()