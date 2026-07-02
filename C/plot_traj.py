import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_cpu = pd.read_csv('trajectories_cpu.txt')
df_gpu = pd.read_csv('trajectories_gpu.txt')

fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

def draw_scene(ax, df, title):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(4 * np.cos(theta), 4 * np.sin(theta), 'k-', lw=2, zorder=5)

    xv = np.linspace(-4, 4, 400)
    ax.axhline(-2, color='saddlebrown', lw=2, zorder=5)
    ax.plot(xv, np.sin(xv), color='purple', lw=2, zorder=5)

    cx, cy, cr = -1.5, 2.5, 0.5
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    ax.plot(cx + cr * np.cos(t), cy + cr * np.sin(t), color='darkred', lw=1.5, zorder=6)
    ax.text(cx - cr - 0.05, cy, 'LeftOnly', ha='right', va='center', fontsize=7, color='darkred')

    cx, cy, cr = 1.5, 2.5, 0.5
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.plot(cx + cr * np.cos(t), cy + cr * np.sin(t), color='darkgreen', lw=1.5, zorder=6)
    ax.text(cx + cr + 0.05, cy, 'RightOnly', ha='left', va='center', fontsize=7, color='darkgreen')

    pids = df['particle'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(pids)))
    for pid, color in zip(pids, colors):
        sub = df[df['particle'] == pid]
        ax.plot(sub['x'].to_numpy(), sub['y'].to_numpy(),
                color=color, lw=0.6, alpha=0.6, zorder=2)

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-2.5, 4.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)

draw_scene(axes[0], df_cpu, 'CPU (OpenMP)')
draw_scene(axes[1], df_gpu, 'GPU (OpenMP offload)')

fig.suptitle('Particle trajectories  (gravity ↓, Yoshida 4th-order, elastic collisions)',
             fontsize=12)
plt.tight_layout()
plt.savefig('trajectories_compare.png', dpi=150, bbox_inches='tight')
print('Saved trajectories_compare.png')
plt.show()
