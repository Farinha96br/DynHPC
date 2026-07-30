"""Plot the cup-test runtime benchmark written by bench_cup.sh.

Two panels rather than one with two y-scales: kernel time and throughput are
different measures, so they get their own axes.

usage: python3 plot_bench.py [bench_cup.csv]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "bench_cup.csv"
OUT = "bench_cup.png"

# categorical slots 1 and 2, assigned in fixed order (GPU first, then CPU)
GPU_C, CPU_C = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

d = np.genfromtxt(SRC, delimiter=",", names=True, dtype=None, encoding="utf-8")
dev = np.array([str(x) for x in d["device"]])
gpu, cpu = d[dev == "gpu"], d[dev == "cpu"]
gpu, cpu = gpu[np.argsort(gpu["n_p"])], cpu[np.argsort(cpu["n_p"])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)
fig.patch.set_facecolor(SURFACE)

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("particles (N_P)", color=INK_2, fontsize=10)

# ── left: kernel time ─────────────────────────────────────────────────────────
for data, color, label in ((gpu, GPU_C, "GPU"), (cpu, CPU_C, "CPU")):
    ax1.plot(data["n_p"], data["kernel_s"], lw=2.0, color=color, marker="o",
             ms=8, mec=SURFACE, mew=1.5, label=label, zorder=3)
    # direct label at the last point (2 series, so label both)
    ax1.annotate(label, (data["n_p"][-1], data["kernel_s"][-1]),
                 textcoords="offset points", xytext=(10, -3),
                 color=INK_2, fontsize=10, fontweight="bold")

# Where the curves cross, the winning device flips. Interpolate on the SPEEDUP
# (cpu/gpu), which increases with N_P -- np.interp requires an increasing xp and
# returns garbage without warning if handed a decreasing one.
speedup = cpu["kernel_s"] / gpu["kernel_s"]
assert np.all(np.diff(speedup) > 0), "speedup must be monotonic to interpolate the crossover"
cross = np.exp(np.interp(0.0, np.log(speedup), np.log(gpu["n_p"])))
ax1.axvline(cross, color=MUTED, lw=1.0, ls="--", zorder=1)
ax1.annotate(f"crossover ≈ {cross:,.0f} particles\n← CPU faster   GPU faster →",
             (cross, gpu["kernel_s"].max()), textcoords="offset points",
             xytext=(-118, 22), color=INK_2, fontsize=8.5, linespacing=1.5)

ax1.set_ylabel("kernel time (s)", color=INK_2, fontsize=10)
ax1.set_title("Kernel time — lower is better", color=INK, fontsize=12, loc="left", pad=12)
ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

# ── right: throughput ─────────────────────────────────────────────────────────
for data, color, label in ((gpu, GPU_C, "GPU"), (cpu, CPU_C, "CPU")):
    ax2.plot(data["n_p"], data["particle_steps_per_s"], lw=2.0, color=color,
             marker="o", ms=8, mec=SURFACE, mew=1.5, label=label, zorder=3)
    ax2.annotate(label, (data["n_p"][-1], data["particle_steps_per_s"][-1]),
                 textcoords="offset points", xytext=(10, -3),
                 color=INK_2, fontsize=10, fontweight="bold")

ax2.set_ylabel("particle-steps / second", color=INK_2, fontsize=10)
ax2.set_title("Throughput — higher is better", color=INK, fontsize=12, loc="left", pad=12)
ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

fig.suptitle("cup_test: GPU offload vs host multicore (same source, nvc++, FP64)",
             color=INK, fontsize=13.5, x=0.008, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, facecolor=SURFACE)
print(f"wrote {OUT}")

# table view: identity never rests on color alone
print(f"\n{'N_P':>9s} {'GPU kernel s':>13s} {'CPU kernel s':>13s} {'GPU speedup':>12s}")
for i in range(len(gpu)):
    sp = cpu["kernel_s"][i] / gpu["kernel_s"][i]
    print(f"{gpu['n_p'][i]:>9d} {gpu['kernel_s'][i]:>13.4f} {cpu['kernel_s'][i]:>13.4f} "
          f"{sp:>11.2f}x")
