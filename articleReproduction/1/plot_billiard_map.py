"""Poincare section (collision map) of the circular gravitational billiard.

Coordinates follow da Costa, Dettmann & Leonel, "Circular, elliptic and oval
billiards in a gravitational field" (arXiv:1308.0362), section II:

    theta  angular position of the particle on the boundary,     [0, 2pi)
    alpha  angle between the trajectory and the TANGENT line
           at the collision point,                               [0, pi]

with alpha = arctan[(V . N) / (V . T)]  (their Eq. 12), N the inward normal and
T the counter-clockwise unit tangent. Their phase-space figures put theta on the
horizontal axis and alpha on the vertical one.

Conserved energy, with the height measured from the BOTTOM of the billiard
(their section III):  E = |V|^2 / 2 + g * (y + R).

NOTE: the paper draws one phase portrait per FIXED energy. This data sweeps the
launch vx, so every particle sits on its own energy shell -- points are coloured
by E to make that explicit. See the notes printed at the end of this script.

Outputs one map per particle, named <out_prefix>_p00.png, <out_prefix>_p01.png,
..., each coloured light->dark by collision order.

    python3 plot_billiard_map.py [collision_table.txt] [out_prefix]
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


SRC = sys.argv[1] if len(sys.argv) > 1 else "collisions_gpu.txt"   # collision table to read
OUT = sys.argv[2] if len(sys.argv) > 2 else "billiard_map"        # prefix of the figures written

# scene / physics: must match SimpleGravCircle.cpp and physics.h
WALL_R = 1.0
G = 1.0                       # force_at_position() returns (0, -1)

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("dynhpc_blue", RAMP)

# ── data → map coordinates ────────────────────────────────────────────────────
d = np.genfromtxt(SRC, delimiter=",", names=True)
pid, x, y, vx, vy = d["particle_ID"], d["x"], d["y"], d["vx"], d["vy"]

theta = np.mod(np.arctan2(y, x), 2 * np.pi)

n_hat = np.stack([-x, -y]) / WALL_R          # inward normal
t_hat = np.stack([-y, x]) / WALL_R           # counter-clockwise tangent
v_n = vx * n_hat[0] + vy * n_hat[1]
v_t = vx * t_hat[0] + vy * t_hat[1]
alpha = np.arctan2(v_n, v_t)                 # in (0, pi) since v_n > 0 outgoing

energy = 0.5 * (vx**2 + vy**2) + G * (y + WALL_R)


# ── figure ────────────────────────────────────────────────────────────────────
def draw_map(th, al, colors, norm, cbar_label, cbar_ticks, title, subtitle, out):
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # the two vertical-bounce fixed points of the field-free circle sit at alpha=pi/2
    ax.axhline(np.pi / 2, color=GRID, lw=1.0, zorder=1)
    for tv in (np.pi / 2, 3 * np.pi / 2):
        ax.axvline(tv, color=GRID, lw=1.0, zorder=1)

    ax.scatter(th, al, s=1.0, c=colors, edgecolors=SURFACE,
               linewidths=0.4, zorder=3)

    # the period-1 vertical orbit at the bottom of the billiard (particle 0)
    ax.plot(3 * np.pi / 2, np.pi / 2, marker="o", ms=9, mfc="none",
            mec=INK_2, mew=1.3, zorder=4)
    ax.annotate("vertical bounce\n(bottom of wall)", (3 * np.pi / 2, np.pi / 2),
                textcoords="offset points", xytext=(12, 10), fontsize=8.5,
                color=INK_2, linespacing=1.35)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_yticklabels(["0", "π/4", "π/2", "3π/4", "π"])

    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6, ls="-", alpha=0.55)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.set_xlabel("θ   (angular position on the wall)", color=INK_2, fontsize=10)
    ax.set_ylabel("α   (angle to the tangent)", color=INK_2, fontsize=10)

    ax.set_title(title, color=INK, fontsize=14, pad=30, loc="left")
    ax.text(0.0, 1.008, subtitle,
            transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                      fraction=0.035, pad=0.02, ticks=cbar_ticks)
    cb.set_label(cbar_label, color=INK_2, fontsize=10)
    cb.ax.tick_params(colors=MUTED, labelsize=9, length=0)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


# ── combined map: colour identifies the particle ──────────────────────────────
# Index is an ordered quantity here (particle i launches with a linearly swept
# vx), so a light->dark ramp is the right encoding.
n_p = int(pid.max()) + 1
norm = Normalize(vmin=0, vmax=n_p - 1)


# ── one map per particle: colour is the collision order (light -> dark) ───────
for p in range(n_p):
    sel = pid == p
    n_c = int(sel.sum())
    if n_c == 0:
        continue
    order = np.arange(n_c)
    p_norm = Normalize(vmin=0, vmax=n_c - 1)
    # each particle conserves its own energy; quote it in the header
    draw_map(theta[sel], alpha[sel], CMAP(p_norm(order)), p_norm,
             "collision order", [0, n_c // 2, n_c - 1],
             f"Collision map — particle {p}",
             f"{n_c} collisions · E = {energy[sel][0]:.4f} · "
             "coordinates after da Costa, Dettmann & Leonel (arXiv:1308.0362)",
             f"{OUT}_p{p:02d}.png")

# ── what this data can and cannot show ────────────────────────────────────────
per_orbit = np.bincount(pid.astype(int))
print(f"  orbits              : {len(per_orbit)}")
print(f"  collisions per orbit: {per_orbit.min()}-{per_orbit.max()}")
print(f"  energy shells       : {energy.min():.4f} .. {energy.max():.4f} "
      f"({len(np.unique(np.round(energy, 9)))} distinct)")
print("  reference uses      : ONE fixed E per portrait, 10x10 grid of "
      "(theta0, alpha0), 2000 collisions each")
