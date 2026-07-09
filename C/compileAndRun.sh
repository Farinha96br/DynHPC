#!/usr/bin/env bash
set -euo pipefail

# Builds and runs the CPU and GPU versions of the integrator sample.
# Optional argument: base name of the sample (default: integrator_sample);
# expects <base>_cpu.cpp and <base>_gpu.cpp next to this script.
# Outputs are saved as trajectories_cpu.txt / trajectories_gpu.txt
# (the two files plot_traj.py reads).
#
# The GPU build needs nvc++ (NVHPC) — run inside the container:
#   apptainer exec --nv ../containers/dynhpc.sif bash compileAndRun.sh

BASE=${1:-integrator_sample}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Compiling CPU build ==="
g++ -O2 -fopenmp -o "$DIR/CPUbuild" "$DIR/${BASE}_cpu.cpp" -lm
echo "CPU build OK"

echo ""
echo "=== Running CPU ==="
(cd "$DIR" && ./CPUbuild && mv trajectories.txt trajectories_cpu.txt)

echo ""
echo "=== Compiling GPU build ==="
nvc++ -O2 -mp=gpu -gpu=ccnative -Minfo=mp -o "$DIR/GPUbuild" "$DIR/${BASE}_gpu.cpp" -lm
echo "GPU build OK"

echo ""
echo "=== Running GPU ==="
(cd "$DIR" && OMP_TARGET_OFFLOAD=MANDATORY ./GPUbuild && mv trajectories.txt trajectories_gpu.txt)
