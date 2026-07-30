set -e

# ── GPU ───────────────────────────────────────────────────────────────────────
echo "Compiling on the GPU"
# nvc++: NVIDIA HPC SDK C++ compiler
# -O3: enable high-level optimization
# -mp=gpu: enable OpenMP offloading for GPU execution
# -gpu=cc89: target the NVIDIA Ada Lovelace architecture used by RTX 4060 series GPUs
# -I../../C: add ../../C to the include search path
# -o GPUvalidate.out: write the executable to GPUvalidate.out
# gravGPU.cpp: source file to compile
# -lm: link the math library
nvc++ -O3 -mp=gpu -gpu=cc89 -I../../C -o GPUbuild.out gravGPU.cpp -lm

echo "Running on the GPU"
time ./GPUbuild.out            # writes collisions_gpu.txt

echo "Plotting"
python3 plot_billiard_map.py collisions_gpu.txt billiard_map_gpu
mkdir -p GPUFigs
mv *.png GPUFigs
rm -f *.txt

# ── CPU ───────────────────────────────────────────────────────────────────────
echo "Compiling on the CPU"
g++ -O3 -fopenmp -I../../C -o CPUbuild.out gravCPU.cpp -lm

echo "Running on the CPU"
time ./CPUbuild.out            # writes collisions_cpu.txt

echo "Plotting"
python3 plot_billiard_map.py collisions_cpu.txt billiard_map_cpu
mkdir -p CPUFigs
mv *.png CPUFigs
rm -f *.txt

echo "\a"