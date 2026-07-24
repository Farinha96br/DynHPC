"""Plot each initial condition from cup_test.csv, colored by outcome:
ended in the left cup, the right cup, or neither.

A particle counts as "in a cup" if its final position lies within r_test of the
cup center — r_test is slightly larger than the cup radius (0.5) because
settled particles rest ON the surface, not strictly inside it.

usage: python plot_cup_outcomes.py [cup_test.csv]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# ── data ───────────────────────────────────────────────────────────────────────
path = sys.argv[1] if len(sys.argv) > 1 else 'cup_test.csv'
d = np.genfromtxt(path, delimiter=',', skip_header=1)
x0, y0, xf, yf = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

L_CENTER, R_CENTER, CUP_R = (-1.5, 2.5), (1.5, 2.5), 0.5
r_test = CUP_R + 0.05

in_left  = ((xf - L_CENTER[0])**2 + (yf - L_CENTER[1])**2 < r_test**2) & (yf < L_CENTER[1])
in_right = ((xf - R_CENTER[0])**2 + (yf - R_CENTER[1])**2 < r_test**2) & (yf < R_CENTER[1])
in_none  = ~in_left & ~in_right

# ── palette (validated categorical slots; gray = recessive non-event) ──────────
C_LEFT  = '#2a78d6'   # slot 1 blue
C_RIGHT = '#1baf7a'   # slot 2 aqua
C_NONE  = '#c3c2b7'   # baseline gray, recessive
INK     = '#0b0b0b'
MUTED   = '#898781'
GRID    = '#e1e0d9'

fig, ax = plt.subplots(figsize=(9, 8), facecolor='#fcfcfb')
ax.set_facecolor('#fcfcfb')

# ── scene geometry (the objects actually in cup_test.cpp's table) ──────────────
# arena: elastic circle at (0,0), r=4; plus the two cups (lower halves solid)
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(4 * np.cos(th), 4 * np.sin(th), color=MUTED, lw=1.2, zorder=3)
for (cx, cy), col, name in ((L_CENTER, C_LEFT, 'left cup'),
                            (R_CENTER, C_RIGHT, 'right cup')):
    arc = np.linspace(np.pi, 2 * np.pi, 200)              # lower semicircle
    ax.plot(cx + CUP_R * np.cos(arc), cy + CUP_R * np.sin(arc),
            color=col, lw=2.5, zorder=6)
    ax.text(cx, cy + 0.12, name, ha='center', va='bottom',
            fontsize=9, color=col, zorder=7)

# ── initial conditions, colored by outcome ─────────────────────────────────────
ax.scatter(x0[in_none],  y0[in_none],  s=0.5,  c=C_NONE,  alpha=0.35,
           lw=0, zorder=2, rasterized=True)
ax.scatter(x0[in_left],  y0[in_left],  s=0.5,  c=C_LEFT,  lw=0, zorder=4)
ax.scatter(x0[in_right], y0[in_right], s=0.5,  c=C_RIGHT, lw=0, zorder=4)

# legend carries the counts (identity never by color alone)
n = len(x0)
handles = [
    plt.Line2D([], [], marker='o', ls='', ms=7,  color=C_LEFT,
               label=f'ended in left cup — {in_left.sum():,} ({in_left.mean()*100:.2f}%)'),
    plt.Line2D([], [], marker='o', ls='', ms=7,  color=C_RIGHT,
               label=f'ended in right cup — {in_right.sum():,} ({in_right.mean()*100:.2f}%)'),
    plt.Line2D([], [], marker='o', ls='', ms=5,  color=C_NONE,
               label=f'no cup — {in_none.sum():,} ({in_none.mean()*100:.2f}%)'),
]
leg = ax.legend(handles=handles, loc='upper left', fontsize=9,
                framealpha=0.9, edgecolor=GRID)
for t in leg.get_texts():
    t.set_color(INK)

ax.set_xlim(-4.4, 4.4)
ax.set_ylim(-4.4, 4.4)
ax.set_aspect('equal')
ax.set_xlabel('x0', color=MUTED)
ax.set_ylabel('y0', color=MUTED)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ax.spines.values():
    s.set_color(GRID)
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.set_title(f'Initial conditions colored by final outcome ({n:,} particles)',
             color=INK, fontsize=11)

plt.tight_layout()
plt.savefig('cup_outcomes.png', dpi=150, bbox_inches='tight')
print(f'saved cup_outcomes.png | left: {in_left.sum()}  right: {in_right.sum()}  none: {in_none.sum()}')
plt.show()
