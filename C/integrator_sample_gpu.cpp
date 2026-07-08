
// integrator_sample_gpu.cpp — GPU version: OpenMP target offload,
// one GPU thread per particle. CPU version: integrator_sample_cpu.cpp.

#define MAX_PARAMS 10   // max parameters per shape / mask
#define MAX_MASKS   8   // max no-collision masks per CollisionableObject


// load basic libs (host-only)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

// everything included inside this block is compiled for BOTH host and GPU;
// only GPU-safe code may live here (no I/O, no allocation, no function pointers)
#pragma omp declare target
#include "types.h"      // vector2D, state, equationSet, mask, collisionEvent, CollisionableObject
#include "thisTable.h"  // implicit shape/mask functions + integer-ID dispatch (eval_F/eval_dx/eval_dy/eval_mask)
#include "physics.h"    // yoshida4Step, singleStep, detectFirstCollision, resolveCollision
#pragma omp end declare target

#include "cpuUtils.h"   // host-only debug helpers

/*
compile (NVIDIA via NVHPC nvc++):
    nvc++ -O2 -mp=gpu -gpu=ccnative -o GPUbuild integrator_sample_gpu.cpp -lm
compile (NVIDIA via GCC/Clang):
    g++ -O2 -fopenmp -fopenmp-targets=nvptx64 -o GPUbuild integrator_sample_gpu.cpp -lm
run:
    ./GPUbuild

GPU note: OpenMP offloading does not support function pointer types on device.
Shape dispatch uses integer IDs + switch (see thisTable.h).
*/



// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    // ── simulation parameters ───────────────────────────────────────────────────
    const double dt      = 1e-4;               // integration time step
    const double t_end   = 100.0;              // total simulated time
    const int    n_steps = (int)(t_end / dt);  // 1 000 000 steps per particle
    const int    N_P     = 200;                // number of particles

    // ── random initial conditions (reproducible seed) ──────────────────────────
    srand(42);
    state* ps = (state*)malloc(N_P * sizeof(state));
    int placed = 0;
    while (placed < N_P) {
        double x  = (rand()/(double)RAND_MAX) * 6.0 - 3.0;   // [-3, 3]
        double y  = (rand()/(double)RAND_MAX) * 3.0 + 0.3;   // [0.3, 3.3]
        if (x*x + y*y > 3.5*3.5) continue;                   // inside r=3.5
        if (y < sin(x) + 0.2)    continue;                   // above sine wave
        double vx = (rand()/(double)RAND_MAX) * 4.0 - 2.0;   // [-2, 2]
        double vy = (rand()/(double)RAND_MAX) * 2.0 - 1.0;   // [-1, 1]
        ps[placed++] = state(x, y, vx, vy);
    }

    // ── collision objects ──────────────────────────────────────────────────────
    // Each surface is the zero level-set of an implicit function F(x,y) = 0,
    // selected by shape_id (see eval_F in thisTable.h). equationSet initializer:
    //   { name, n_params, { params... }, shape_id }
    // Masks carve out regions where a surface does NOT collide: a hit is ignored
    // when any of the object's masks evaluates negative at the impact point.
    CollisionableObject cobjs[5];
    cobjs[0].obj = {"Circle",    3, {0.0,  0.0, 4.0}, 0};  cobjs[0].restitution = 1.0;  // outer wall: circle at (0,0), r=4
    cobjs[1].obj = {"Line",      2, {0.0,  2.0},      1};  cobjs[1].restitution = 1.0;  // flat floor: y = 0*x - 2
    cobjs[2].obj = {"Sine",      3, {1.0,  1.0, 0.0}, 2};  cobjs[2].restitution = 1.0;  // wavy floor: y = 1*sin(1*x + 0)
    cobjs[3].obj = {"LeftOnly",  3, {-1.5, 2.5, 0.5}, 0};  cobjs[3].restitution = 1.0;  // circle at (-1.5,2.5), r=0.5 ...
    cobjs[3].masks[0] = {0, {-1.5}};  cobjs[3].n_masks = 1;                             // ... collides on its left half only (masked for x > -1.5)
    cobjs[4].obj = {"RightOnly", 3, { 1.5, 2.5, 0.5}, 0};  cobjs[4].restitution = 1.0;  // circle at (1.5,2.5), r=0.5 ...
    cobjs[4].masks[0] = {1, { 1.5}};  cobjs[4].n_masks = 1;                             // ... collides on its right half only (masked for x < 1.5)

    // ── trajectory buffer: N_TRAJ waypoints evenly spaced through the run ────────
    // Positions are recorded every N_TRAJ_STRIDE steps (not every step) to keep
    // the output small. Flat layout: traj[(particle * N_TRAJ + waypoint) * 2] = {x, y}.
    const int N_TRAJ        = 500;
    const int N_TRAJ_STRIDE = n_steps / N_TRAJ;
    double*   traj = new double[(long long)N_P * N_TRAJ * 2]();

    // ── simulation ─────────────────────────────────────────────────────────────
    // Particles never interact, so each one runs its whole time loop
    // independently — one GPU thread per particle.
    //
    // Per fine step, collisions are handled explicitly here in the loop:
    //   1. integrate:  s_new  = singleStep(p, dt)          (Yoshida 4, no collisions)
    //   2. detect:     did the path p → s_new cross a surface? (earliest hit wins)
    //   3. resolve:    reflect velocity at the impact point, then integrate
    //                  the remaining dt - dt_hit from the bounced state
    // At most one bounce per dt is handled (change `if` to `while` for multi-bounce).

    // target data: upload objects + initial states once (to), download the
    // finished trajectory buffer once when the region ends (from) —
    // no host↔device traffic inside the time loop.
    #pragma omp target data map(to: cobjs[0:5], ps[0:N_P]) \
                            map(from: traj[0:N_P * N_TRAJ * 2])
    {
        // one GPU kernel: the runtime spreads the N_P particles over teams/threads
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < N_P; ++i) {
            state p = ps[i];
            for (int t = 0; t < N_TRAJ; ++t) {                     // waypoint loop
                for (int s = 0; s < N_TRAJ_STRIDE; ++s) {          // fine steps between waypoints
                    state s_new = singleStep(p, dt);                                       // 1. integrate
                    collisionEvent ev = detectFirstCollision(p, s_new, cobjs, 5, dt);      // 2. detect
                    if (ev.obj_idx >= 0) {
                        state s_bounced = resolveCollision(ev, cobjs[ev.obj_idx], p.mass); // 3. resolve
                        s_new = singleStep(s_bounced, dt - ev.dt_hit);                     //    + finish the step
                    }
                    p = s_new;
                }
                // record waypoint t of particle i
                long long tidx = ((long long)i * N_TRAJ + t) * 2;
                traj[tidx + 0] = p.position.x;
                traj[tidx + 1] = p.position.y;
            }
        }
    }

    // ── write trajectories ────────────────────────────────────────────────────
    // CSV, one row per waypoint: particle index, x, y (read by plot_traj.py)
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
