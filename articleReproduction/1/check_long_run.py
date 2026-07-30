"""Precision and similarity checks for the long-time gravitational billiard.

Two independent questions, answered with different tools:

  PRECISION  (per platform)  -- after 1e9 timesteps, does the integrator still
             conserve what it should? Physical invariants only.

  SIMILARITY (CPU vs GPU)    -- trajectory-by-trajectory comparison is meaningless
             for a chaotic system: FMA rounding alone drives chaotic orbits apart
             to O(1) within a few dozen collisions. So the no-FMA build is used as
             a bitwise control, and the FMA build is compared statistically.

Map coordinates (theta, alpha) follow plot_billiard_map.py / da Costa, Dettmann &
Leonel (arXiv:1308.0362).

    python3 check_long_run.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# must match the sources: force_at_position() = (0, -0.5), wall = circle r = 1
G, WALL_R = 0.5, 1.0
N_REFLECTIONS = 100000        # the cap in gravCPU.cpp / gravGPU.cpp

CPU, GPU, GPU_NOFMA = "collisions_cpu.txt", "collisions_gpu.txt", "collisions_gpu_nofma.txt"


def load(path):
    """Collision table -> {particle_id: recarray of its rows}, in time order."""
    try:                      # pandas is ~10x faster on these 130 MB tables
        import pandas as pd
        d = pd.read_csv(path).to_records(index=False)
    except ImportError:
        d = np.genfromtxt(path, delimiter=",", names=True)
    pid = d["particle_ID"].astype(int)
    return {p: d[pid == p] for p in range(pid.max() + 1)}


def coords(s):
    """(theta, alpha, energy, speed) for one particle's collision rows."""
    x, y, vx, vy = s["x"], s["y"], s["vx"], s["vy"]
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)
    n = np.stack([-x, -y]) / WALL_R          # inward normal
    t = np.stack([-y, x]) / WALL_R           # counter-clockwise tangent
    alpha = np.arctan2(vx * n[0] + vy * n[1], vx * t[0] + vy * t[1])
    energy = 0.5 * (vx**2 + vy**2) + G * (y + WALL_R)
    return theta, alpha, energy, np.hypot(vx, vy)


def rule(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def ks_safe(a, b):
    """Two-sample KS, but nan for a near-constant sample.

    A perfectly regular orbit has essentially constant collision intervals. Two
    such samples agreeing to ~1e-14 can still produce a large ECDF gap, because
    KS measures rank order and ignores magnitude. Reporting that as a difference
    would be an artifact, so degenerate samples are excluded instead.
    """
    pooled = np.concatenate([a, b])
    spread = pooled.max() - pooled.min()
    if spread / max(abs(pooled.mean()), 1e-30) < 1e-9:
        return np.nan, np.nan
    k = ks_2samp(a, b)
    return k.statistic, k.pvalue


# ── load ──────────────────────────────────────────────────────────────────────
cpu, gpu = load(CPU), load(GPU)
parts = sorted(cpu.keys())

# ── B1: bitwise control ───────────────────────────────────────────────────────
# a byte comparison, so the control table is never parsed
rule("B1  bitwise control: GPU(no FMA) vs CPU  -- must be byte-identical")
if not os.path.exists(GPU_NOFMA):
    print("  SKIPPED (no collisions_gpu_nofma.txt)")
    b1 = None
else:
    import filecmp
    same = filecmp.cmp(CPU, GPU_NOFMA, shallow=False)
    b1 = same
    print(f"  {'PASS -- byte-identical' if same else 'FAIL -- tables differ'}")
    print("  => any CPU/GPU difference below is FMA rounding, not a logic difference"
          if same else "  => investigate before trusting anything below")

# ── A: precision, per platform ────────────────────────────────────────────────
rule("A  precision checks (per platform, after 1e9 timesteps)")
print(f"{'':4s}{'particle':>9s} {'n_coll':>7s} {'E0':>10s} {'max|E-E0|':>11s} "
      f"{'secular slope':>14s} {'|r-1| max':>10s} {'bad bounce':>11s}")
energy_series = {}
for label, data in (("CPU", cpu), ("GPU", gpu)):
    print(f"  {label}")
    for p in parts:
        s = data[p]
        theta, alpha, E, spd = coords(s)
        t = s["t"]
        # A1 energy: excursion + secular trend (symplectic => bounded, slope ~ 0)
        dE = E - E[0]
        slope = np.polyfit(t, dE, 1)[0] if len(t) > 2 else 0.0
        # A2 geometric residual
        rres = np.abs(np.hypot(s["x"], s["y"]) - WALL_R).max()
        # A4 reflection law: the recorded velocity is the OUTGOING one, so it must
        #    point back into the domain (v . inward-normal >= 0, i.e. alpha in
        #    [0, pi]). A negative value means the bounce left the particle heading
        #    into the wall -- a real failure of resolveCollision, independent of
        #    the energy check. (Exactly 0 is legitimate: the resting-contact clamp.)
        vn = (s["vx"] * (-s["x"]) + s["vy"] * (-s["y"])) / WALL_R
        bad = int((vn < -1e-12).sum())
        print(f"    {p:>9d} {len(s):>7d} {E[0]:>10.6f} {np.abs(dE).max():>11.3e} "
              f"{slope:>14.3e} {rres:>10.3e} {bad:>11d}")
        energy_series[(label, p)] = (t, dE)

# A3 bookkeeping
rule("A3  bookkeeping sanity")
for label, data in (("CPU", cpu), ("GPU", gpu)):
    trunc = [p for p in parts if len(data[p]) >= N_REFLECTIONS]
    nonmono = [p for p in parts if np.any(np.diff(data[p]["t"]) < 0)]
    tmax = max(data[p]["t"][-1] for p in parts)
    print(f"  {label}: truncated particles={trunc or 'none'}  "
          f"non-monotone times={nonmono or 'none'}  latest t={tmax:.2f}")

# ── B4: chaos-independent invariant ───────────────────────────────────────────
rule("B4  per-particle mean energy: conserved exactly, so CPU and GPU must agree\n"
     "    even though their trajectories are completely different")
print(f"{'particle':>9s} {'E_cpu':>14s} {'E_gpu':>14s} {'|diff|':>11s} {'n_cpu':>8s} {'n_gpu':>8s} {'n diff':>8s}")
worst = 0.0
for p in parts:
    ec = coords(cpu[p])[2].mean()
    eg = coords(gpu[p])[2].mean()
    dif = abs(ec - eg)
    worst = max(worst, dif)
    nc, ng = len(cpu[p]), len(gpu[p])
    print(f"{p:>9d} {ec:>14.10f} {eg:>14.10f} {dif:>11.2e} {nc:>8d} {ng:>8d} "
          f"{100*abs(nc-ng)/max(nc,1):>7.2f}%")
print(f"\n  worst mean-energy disagreement: {worst:.3e}")

# ── B2: divergence horizon ────────────────────────────────────────────────────
rule("B2  divergence horizon: first collision index where CPU and GPU differ by >eps")
THRESH = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]
print(f"{'particle':>9s} " + " ".join(f"{f'>{e:.0e}':>10s}" for e in THRESH) + f" {'lyap/coll':>11s}")
diverg = {}
for p in parts:
    a, b = cpu[p], gpu[p]
    n = min(len(a), len(b))
    d = np.hypot(a["x"][:n] - b["x"][:n], a["y"][:n] - b["y"][:n])
    diverg[p] = d
    row = []
    for e in THRESH:
        idx = np.argmax(d > e)
        row.append(f"{idx:>10d}" if d[idx] > e else f"{'never':>10s}")
    # Effective Lyapunov exponent per collision. Fit ONLY the initial exponential
    # rise: once the separation saturates at O(1) it wanders and dips back below
    # the band, and including those late points flattens the slope into nonsense.
    sat = np.argmax(d > 1e-3)
    if d[sat] > 1e-3 and sat > 5:
        idx = np.arange(sat)
        band = d[:sat] > 1e-17
        lam = np.polyfit(idx[band], np.log(d[:sat][band]), 1)[0] if band.sum() > 5 else np.nan
    else:
        lam = np.nan          # never diverged: regular orbit
    print(f"{p:>9d} " + " ".join(row) + f" {lam:>11.4f}")

# ── B3: statistical equivalence ───────────────────────────────────────────────
rule("B3  statistical equivalence beyond the horizon\n"
     "    Judged on the KS STATISTIC D (largest gap between the two ECDFs, 0..1),\n"
     "    not the p-value: with tens of thousands of collisions the KS test is so\n"
     "    powerful that it rejects on differences far too small to matter. D is an\n"
     "    effect size, so D < 0.05 means the two samples follow the same law for\n"
     "    any practical purpose. p-values are printed alongside for reference.")
D_OK = 0.05
print(f"\n{'particle':>9s} {'D(theta)':>9s} {'D(alpha)':>9s} {'D(dt)':>9s} "
      f"{'TV(map)':>9s} {'min p':>9s}  verdict")
BINS = 60
fails = []
for p in parts:
    tc, ac, _, _ = coords(cpu[p])
    tg, ag, _, _ = coords(gpu[p])
    dtc, dtg = np.diff(cpu[p]["t"]), np.diff(gpu[p]["t"])
    (d_th, p_th), (d_al, p_al), (d_dt, p_dt) = ks_safe(tc, tg), ks_safe(ac, ag), ks_safe(dtc, dtg)
    Hc, _, _ = np.histogram2d(tc, ac, bins=BINS, range=[[0, 2*np.pi], [0, np.pi]])
    Hg, _, _ = np.histogram2d(tg, ag, bins=BINS, range=[[0, 2*np.pi], [0, np.pi]])
    tv = 0.5 * np.abs(Hc/Hc.sum() - Hg/Hg.sum()).sum()
    Ds = [d for d in (d_th, d_al, d_dt) if not np.isnan(d)]
    Dmax = max(Ds) if Ds else 0.0
    ok = Dmax < D_OK
    if not ok:
        fails.append(p)
    fmt = lambda v: f"{v:>9.4f}" if not np.isnan(v) else f"{'degen':>9s}"
    pmin = np.nanmin([p_th, p_al, p_dt])
    print(f"{p:>9d} {fmt(d_th)} {fmt(d_al)} {fmt(d_dt)} {tv:>9.4f} "
          f"{pmin if not np.isnan(pmin) else float('nan'):>9.3f}  "
          f"{'same law' if ok else 'DIFFER'}")
print(f"\n  particles with KS statistic D > {D_OK}: {fails or 'none'}")

# ── figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, label in zip(axes, ("CPU", "GPU")):
    for p in parts:
        t, dE = energy_series[(label, p)]
        ax.plot(t[::37], dE[::37], lw=0.5)
    ax.set_xlabel("t"); ax.set_ylabel("E(t) - E(0)")
    ax.set_title(f"A1  energy drift, {label} (flat band = symplectic, no secular growth)")
fig.tight_layout(); fig.savefig("check_energy.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
for p in parts:
    d = np.maximum(diverg[p], 1e-18)
    ax.semilogy(d[:min(len(d), 2000)], lw=0.7, label=f"p{p}" if p % 5 == 0 else None)
ax.set_xlabel("collision index"); ax.set_ylabel("|CPU - GPU| position")
ax.set_title("B2  divergence growth (straight line on log axis = exponential = chaos)")
ax.legend(fontsize=7, ncols=2)
fig.tight_layout(); fig.savefig("check_divergence.png", dpi=140); plt.close(fig)

reg = parts[0]
cha = max(parts, key=lambda p: np.nan_to_num(diverg[p][-1]))
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for row, p in enumerate((reg, cha)):
    for col, (label, data) in enumerate((("CPU", cpu), ("GPU", gpu))):
        th, al, _, _ = coords(data[p])
        axes[row][col].scatter(th, al, s=0.2, alpha=0.35, c="#2a78d6")
        axes[row][col].set_xlim(0, 2*np.pi); axes[row][col].set_ylim(0, np.pi)
        axes[row][col].set_xlabel("theta"); axes[row][col].set_ylabel("alpha")
        kind = "regular" if row == 0 else "chaotic"
        axes[row][col].set_title(f"B3  particle {p} ({kind}) -- {label}")
fig.tight_layout(); fig.savefig("check_maps.png", dpi=140); plt.close(fig)

rule("summary")
print(f"  B1 bitwise control     : {'PASS' if b1 else ('FAIL' if b1 is not None else 'skipped')}")
print(f"  B4 mean-energy agreement: worst {worst:.3e}")
print(f"  B3 distributions differ : {fails or 'none'}")
print("  figures: check_energy.png  check_divergence.png  check_maps.png")
