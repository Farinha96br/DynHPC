# Precision and similarity checks for the long-time billiard run.
#
# Builds three binaries from the same two sources and runs each into its own
# collision table:
#   collisions_cpu.txt        g++, no FMA (baseline)
#   collisions_gpu.txt        nvc++, FMA on  (the production GPU build)
#   collisions_gpu_nofma.txt  nvc++, FMA off (must be byte-identical to the CPU)
# then runs check_long_run.py over them.
#
# At dt=1e-4, t_end=1e5 this is 1e9 timesteps: each GPU run takes several
# minutes, the CPU run under a minute.
set -e

echo "=== building ==="
g++   -O2 -fopenmp                    -I../../C -o CPUcheck.out       gravCPU.cpp -lm
nvc++ -O2 -mp=gpu -gpu=ccnative       -I../../C -o GPUcheck.out       gravGPU.cpp -lm
nvc++ -O2 -mp=gpu -gpu=ccnative,nofma -I../../C -o GPUcheck_nofma.out gravGPU.cpp -lm

echo "=== running CPU ==="
time ./CPUcheck.out                                  # -> collisions_cpu.txt

# no-FMA first, then FMA: both write collisions_gpu.txt, so the first is renamed
echo "=== running GPU (no FMA) ==="
time ./GPUcheck_nofma.out
mv collisions_gpu.txt collisions_gpu_nofma.txt

echo "=== running GPU (FMA) ==="
time ./GPUcheck.out                                  # -> collisions_gpu.txt

echo "=== analysis ==="
python3 check_long_run.py
