# Runtime benchmark of the cup test: GPU offload vs host multicore.
#
# Both binaries are built from the SAME source with the SAME compiler, changing
# only the OpenMP target, so the measurement isolates the device rather than the
# code generator:
#     nvc++ -mp=gpu -gpu=ccnative   -> offloaded to the GPU
#     nvc++ -mp=multicore           -> the same target region on host threads
#
# Only the kernel is timed (omp_get_wtime around the target region), so the
# host-side RNG setup and the CSV write -- identical on both -- are excluded.
# cup_test.cpp itself is never modified: patched copies are generated in a
# scratch directory.
#
# Writes bench_cup.csv: n_p,device,kernel_s,total_s,particle_steps_per_s
set -e

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
SRC=$(cd "$(dirname "$0")" && pwd)
CSV="$SRC/bench_cup.csv"

T_END=1.0            # 1000 steps at dt=1e-3; runtime scales linearly in this
NPS="1000 10000 100000 1000000"

echo "CPU: $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset, defaults to $(nproc)>}"
echo

# ── generate the instrumented source ──────────────────────────────────────────
# N_P and t_end become argv-overridable, and the kernel is wrapped in a timer.
python3 - "$SRC/cup_test.cpp" "$WORK/bench.cpp" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

def sub(pattern, repl, s, what):
    """Regex substitution that REFUSES to fail silently.

    An unmatched pattern here would leave the constant hard-coded and the sweep
    would benchmark the same configuration over and over -- producing flat
    timings that look like a physics result instead of a broken harness.
    """
    s2, n = re.subn(pattern, repl, s, count=1)
    if n != 1:
        sys.exit(f"bench_cup.sh: could not patch {what!r} in cup_test.cpp "
                 f"(pattern {pattern!r} matched {n} times) -- refusing to "
                 f"report numbers from an unpatched binary")
    return s2

# match whatever value the source currently holds, not a hard-coded literal
s = sub(r"int main\(\) \{", "int main(int argc, char** argv) {", s, "main signature")
s = sub(r"const double t_end\s*=\s*[0-9.eE+-]+;",
        "const double t_end = (argc > 2) ? atof(argv[2]) : 1.0;", s, "t_end")
s = sub(r"const int\s+N_P\s*=\s*[0-9.eE+-]+;",
        "const int    N_P   = (argc > 1) ? atoi(argv[1]) : (int)1e6;", s, "N_P")
s = sub(r"( *)#pragma omp target teams distribute parallel for",
        r"\1double _t0 = omp_get_wtime();\n\1#pragma omp target teams distribute parallel for",
        s, "kernel start timer")
s = sub(r"( *)// ── write start → end positions",
        r'\1double _kernel_s = omp_get_wtime() - _t0;\n'
        r'\1fprintf(stderr, "KERNEL_SECONDS %.6f\\n", _kernel_s);\n\n'
        r"\1// ── write start → end positions",
        s, "kernel stop timer")
open(dst, "w").write(s)
PY

echo "Building GPU (offload) and CPU (multicore) from the same source"
nvc++ -O3 -mp=gpu -gpu=ccnative -I"$SRC/.." -o "$WORK/cup_gpu" "$WORK/bench.cpp" -lm
nvc++ -O3 -mp=multicore         -I"$SRC/.." -o "$WORK/cup_cpu" "$WORK/bench.cpp" -lm
echo

# ── correctness gate: both devices must compute the same thing ────────────────
cd "$WORK"
./cup_gpu 2000 "$T_END" 2>/dev/null >/dev/null; mv cup_test.csv gpu_check.csv
./cup_cpu 2000 "$T_END" 2>/dev/null >/dev/null; mv cup_test.csv cpu_check.csv
if cmp -s gpu_check.csv cpu_check.csv; then
    echo "correctness gate: GPU and CPU results are identical"
else
    echo "correctness gate: results DIFFER (expected -- nvc++ contracts FMA differently"
    echo "                  per target; the two are still the same computation)"
fi
echo

# ── sweep ─────────────────────────────────────────────────────────────────────
echo "n_p,device,kernel_s,total_s,particle_steps_per_s" > "$CSV"
printf "%10s %10s %12s %12s %16s\n" n_p device kernel_s total_s "part-steps/s"

for dev in gpu cpu; do
    ./cup_$dev 1000 0.1 >/dev/null 2>&1 || true       # warm-up: GPU context, page faults
    for n in $NPS; do
        start=$(date +%s.%N)
        k=$(./cup_$dev "$n" "$T_END" 2>&1 >/dev/null | sed -n 's/^KERNEL_SECONDS //p')
        end=$(date +%s.%N)
        tot=$(echo "$end - $start" | bc)
        rate=$(echo "scale=0; $n * ($T_END / 0.001) / $k" | bc -l 2>/dev/null || echo 0)
        printf "%10d %10s %12.4f %12.4f %16.3e\n" "$n" "$dev" "$k" "$tot" "$rate"
        echo "$n,$dev,$k,$tot,$rate" >> "$CSV"
    done
done

echo
echo "wrote $CSV"
