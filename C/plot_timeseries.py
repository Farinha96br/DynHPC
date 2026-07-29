"""Plot the sampled trajectories (timeseries_gpu.txt) together with the
collision table (collisions_gpu.txt) written by gpuFunctions.cpp.

Left: trajectories in space, with the scene (two touching circles, the
pass-through disk, the bounding box) and collision points on top.
Right: y(t) per particle, with its collision times marked.

usage: python3 plot_timeseries.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ts = np.genfromtxt("timeseries_gpu.txt", delimiter=",", names=True)
co = np.genfromtxt("collisions_gpu.txt", delimiter=",", names=True)

n_p = int(ts["particle_ID"].max()) + 1
cmap = plt.get_cmap("tab10")


fig, (ax1) = plt.subplots(1, 1, figsize=(14, 6))

# ── left: trajectories in space ───────────────────────────────────────────────
for i in range(n_p):
    s = ts["particle_ID"] == i
    ax1.plot(ts["x"][s], ts["y"][s], lw=0.6, color=cmap(i % 10), alpha=0.8)
ax1.scatter(co["x"], co["y"], s=2, c="black", zorder=3, label="collisions")

# scene: two touching circles, pass-through disk (dashed), bounding box
th = np.linspace(0, 2 * np.pi, 200)
for cx in (-1.0, 1.0):
    ax1.plot(cx + np.cos(th), np.sin(th), color="gray", lw=1.5)
ax1.plot(0.3 * np.cos(th), 0.3 * np.sin(th), color="gray", lw=1.0, ls="--")
ax1.plot([-3, 3, 3, -3, -3], [-3, -3, 3, 3, -3], color="gray", lw=1.5)

ax1.set_xlim(-3.2, 3.2)
ax1.set_ylim(-3.2, 3.2)
ax1.set_aspect("equal")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("trajectories (sampled every N steps) + collision points")
ax1.legend(loc="upper right", fontsize=8)

fig.tight_layout()
#fig.savefig("timeseries_gpu.png", dpi=150)
plt.show()
print("saved timeseries_gpu.png")
