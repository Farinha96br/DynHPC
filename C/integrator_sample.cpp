
#define MAX_PARAMS 10   // max parameters per shape / mask
#define MAX_MASKS   8   // max no-collision masks per CollisionableObject


// load basic libs
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

// load parallel relevant libs
#pragma omp declare target
#include "types.h"
#include "thisTable.h"
#include "physics.h"
#pragma omp end declare target

#include "cpuUtils.h"

/*
compile (CPU, multi-threaded):
    g++ -O2 -fopenmp -o integrator integrator.cpp -lm
compile (GPU — NVIDIA via NVHPC nvc++):
    nvc++ -O2 -mp=gpu -DGPU_OFFLOAD -o integrator integrator.cpp
compile (GPU — NVIDIA via GCC/Clang):
    g++ -O2 -fopenmp -fopenmp-targets=nvptx64 -DGPU_OFFLOAD -o integrator integrator.cpp -lm
run:
    ./integrator

GPU note: OpenMP offloading does not support function pointer types on device.
Shape dispatch uses integer IDs + switch. To add a new shape: implement its
F/dx/dy functions (in the declare target block below), assign it an integer ID,
and add one case to each of eval_F / eval_dx / eval_dy.
*/









// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    const double dt      = 1e-4;
    const double t_end   = 100.0;              // 10 000 steps — keeps final-state buffer small
    const int    n_steps = (int)(t_end / dt);
    const int    N_P     = 200;

    // ── random initial conditions (reproducible seed) ──────────────────────────
    srand(42);
    state* ps = (state*)malloc(N_P * sizeof(state));
    int placed = 0;
    while (placed < N_P) {
        double x  = (rand()/(double)RAND_MAX) * 6.0 - 3.0;   // [-3, 3]
        double y  = (rand()/(double)RAND_MAX) * 3.0 + 0.3;   // [0.3, 3.3]
        if (x*x + y*y > 3.5*3.5) continue;                   // inside r=3.5
        if (y < sin(x) + 0.2)    continue;                   // above sine wave
        double vx = (rand()/(double)RAND_MAX) * 4.0 - 2.0;
        double vy = (rand()/(double)RAND_MAX) * 2.0 - 1.0;
        ps[placed++] = state(x, y, vx, vy);
    }

    // ── collision objects ──────────────────────────────────────────────────────
    CollisionableObject cobjs[5];
    cobjs[0].obj = {"Circle",    3, {0.0,  0.0, 4.0}, 0};  cobjs[0].restitution = 1.0;
    cobjs[1].obj = {"Line",      2, {0.0,  2.0},      1};  cobjs[1].restitution = 1.0;
    cobjs[2].obj = {"Sine",      3, {1.0,  1.0, 0.0}, 2};  cobjs[2].restitution = 1.0;
    cobjs[3].obj = {"LeftOnly",  3, {-1.5, 2.5, 0.5}, 0};  cobjs[3].restitution = 1.0;
    cobjs[3].masks[0] = {0, {-1.5}};  cobjs[3].n_masks = 1;
    cobjs[4].obj = {"RightOnly", 3, { 1.5, 2.5, 0.5}, 0};  cobjs[4].restitution = 1.0;
    cobjs[4].masks[0] = {1, { 1.5}};  cobjs[4].n_masks = 1;

    // ── trajectory buffer: N_TRAJ waypoints evenly spaced through the run ────────
    const int N_TRAJ        = 500;
    const int N_TRAJ_STRIDE = n_steps / N_TRAJ;
    double*   traj = new double[(long long)N_P * N_TRAJ * 2]();

    // ── simulation ─────────────────────────────────────────────────────────────
// if using the GPU
#ifdef GPU_OFFLOAD
    #pragma omp target data map(to: cobjs[0:5], ps[0:N_P]) \
                            map(from: traj[0:N_P * N_TRAJ * 2])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < N_P; ++i) {
            state p = ps[i];
            for (int t = 0; t < N_TRAJ; ++t) {
                for (int s = 0; s < N_TRAJ_STRIDE; ++s) {
                    state s_new = singleStep(p, dt);
                    collisionEvent ev = detectFirstCollision(p, s_new, cobjs, 5, dt);
                    if (ev.obj_idx >= 0) {
                        state s_bounced = resolveCollision(ev, cobjs[ev.obj_idx], p.mass);
                        s_new = singleStep(s_bounced, dt - ev.dt_hit);
                    }
                    p = s_new;
                }
                long long tidx = ((long long)i * N_TRAJ + t) * 2;
                traj[tidx + 0] = p.position.x;
                traj[tidx + 1] = p.position.y;
            }
        }
    }

// if using the CPU
#else
    #pragma omp parallel for
    for (int i = 0; i < N_P; ++i) {
        state p = ps[i]; // initialize all particles from the list generated first
        // time loop, with N_TRAJ evenly spaced times
        for (int t = 0; t < N_TRAJ; ++t) {
            
            for (int s = 0; s < N_TRAJ_STRIDE; ++s) {
                state s_new = singleStep(p, dt);
                collisionEvent ev = detectFirstCollision(p, s_new, cobjs, 5, dt);
                if (ev.obj_idx >= 0) {
                    state s_bounced = resolveCollision(ev, cobjs[ev.obj_idx], p.mass);
                    s_new = singleStep(s_bounced, dt - ev.dt_hit);
                }
                p = s_new;
            }
            long long tidx = ((long long)i * N_TRAJ + t) * 2;
            traj[tidx + 0] = p.position.x;
            traj[tidx + 1] = p.position.y;
        }
    }
#endif

    // ── write trajectories ────────────────────────────────────────────────────
    FILE* fout = fopen("trajectories.txt", "w");
    fprintf(fout, "particle,x,y\n");
    for (int i = 0; i < N_P; ++i) {
        for (int t = 0; t < N_TRAJ; ++t) {
            long long tidx = ((long long)i * N_TRAJ + t) * 2;
            fprintf(fout, "%d,%.6f,%.6f\n", i, traj[tidx+0], traj[tidx+1]);
        }
    }
    fclose(fout);
    printf("Wrote %d particles x %d waypoints to trajectories.txt\n", N_P, N_TRAJ);

    free(ps);
    delete[] traj;
    return 0;
}
