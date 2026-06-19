#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("trajectories.csv", delimiter=",", skiprows=1)
t   = data[:, 0]
xs  = data[:, 1::2]   # columns x0, x1, x2, ...
ys  = data[:, 2::2]   # columns y0, y1, y2, ...
N   = xs.shape[1]

# Ellipse parameters matching integrator2.cpp
ellipses = [
    dict(cx=0.0,  cy=0.0,   a=2.0, b=1.0, sca=0,          eca=2*np.pi),
    dict(cx=0.01, cy=-0.01, a=0.1, b=0.2, sca=np.pi/2,    eca=3*np.pi/2),
]

lines = [
    dict(m=0.0, b=-0.5),  # horizontal line at y = -0.5
]

colors = plt.cm.tab10(np.linspace(0, 1, N))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- trajectories in position space ---
ax = axes[0]
for e in ellipses:
    theta = np.linspace(e["sca"], e["eca"], 2048)
    ex = e["cx"] + e["a"] * np.cos(theta)
    ey = e["cy"] + e["b"] * np.sin(theta)
    ax.plot(ex, ey, "k-", linewidth=0.1)

x_range = np.array([-2.5, 2.5])
for l in lines:
    ax.plot(x_range, l["m"] * x_range + l["b"], "k--", linewidth=1.5)

for i in range(N):
    ax.plot(xs[:, i], ys[:, i], color=colors[i], linewidth=0.1,
            label=f"particle {i}", alpha=0.9)
    ax.plot(xs[0, i], ys[0, i], "o", color=colors[i], markersize=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"Trajectories — {N} particles")
#ax.legend()
ax.grid(True, alpha=0.3)

# --- y(t) ---
ax = axes[1]
for i in range(N):
    ax.plot(t, ys[:, i], color=colors[i], linewidth=0.8, label=f"particle {i}")
ax.set_xlabel("time (s)")
ax.set_ylabel("y")
ax.set_title("y(t)")
#ax.legend()
ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("trajectories.png", dpi=300)
print("saved trajectories.png")
