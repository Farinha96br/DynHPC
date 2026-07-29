
// cup_test.cpp — GPU (OpenMP target offload) test scene where the two small
// circles collide only on their lower halves (upward-open "cups").
// Based on integrator_sample_gpu.cpp.
//
// Particles are advanced from t = 0 to t_end; afterwards one CSV row per
// particle is written to cup_test.csv:  x0,y0,xf,yf  (start → end position).

#define MAX_PARAMS 10   // max parameters per shape / mask
#define MAX_MASKS   8   // max no-collision masks per CollisionableObject

// load basic libs (host-only)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// everything included inside this block is compiled for BOTH host and GPU;
// only GPU-safe code may live here (no I/O, no allocation, no function pointers)
#pragma omp declare target
#include "types.h"      // vector2D, state, equationSet, mask, collisionEvent, CollisionableObject
#include "sampleTable.h"  // implicit shape/mask functions + integer-ID dispatch (eval_F/eval_dx/eval_dy/eval_mask)
#include "physics.h"    // yoshida4Step, singleStep, detectFirstCollision, resolveCollision
#pragma omp end declare target

/*
compile (NVIDIA via NVHPC nvc++):
    nvc++ -O2 -mp=gpu -gpu=ccnative -o cup_test cup_test.cpp -lm
compile (NVIDIA via GCC/Clang):
    g++ -O2 -fopenmp -fopenmp-targets=nvptx64 -o cup_test cup_test.cpp -lm
run:
    ./cup_test

GPU note: OpenMP offloading does not support function pointer types on device.
Shape dispatch uses integer IDs + switch (see thisTable.h).
*/



// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    // ── simulation parameters ───────────────────────────────────────────────────
    const double dt    = 1e-3;   // integration time step
    const double t_end = 100.0;    // total simulated time
    const int    N_P   = 1e6;    // number of particles

    // ── random initial conditions (reproducible seed) ──────────────────────────
    srand(42);
    state* ps = (state*)malloc(N_P * sizeof(state));
    int placed = 0;
    while (placed < N_P) {
        double x  = (rand()/(double)RAND_MAX) * 8.0 - 4.0;   // [-4, 4]
        double y  = (rand()/(double)RAND_MAX) * 8.0 - 4.0;   // [-4, 4]
        if (x*x + y*y > 3.99*3.99) continue;                   // inside r=3.5
        double vx =  0.0;
        double vy = -3.0;
        ps[placed++] = state(x, y, vx, vy);
    }

    // keep a copy of the initial states — the kernel overwrites ps with the
    // final states, and the CSV needs both
    state* ps0 = (state*)malloc(N_P * sizeof(state));
    memcpy(ps0, ps, N_P * sizeof(state));

    // ── collision objects ──────────────────────────────────────────────────────
    // Each surface is the zero level-set of an implicit function F(x,y) = 0,
    // selected by shape_id (see eval_F in thisTable.h). equationSet initializer:
    //   { name, n_params, { params... }, shape_id }
    // Masks live on layers and carve out regions where NO object of that layer
    // collides: a hit is ignored when any of the layer's masks evaluates
    // negative at the impact point. Both cups share layer 1's mask.
    const int N_L = 2;
    Layer layers[N_L];                                     // layer 0: no masks (outer wall)
    layers[1].masks[0] = {2, {2.5}};  layers[1].n_masks = 1;  // layer 1: masked for y > 2.5 (lower halves only)

    CollisionableObject cobjs[5];
    cobjs[0].obj = {"Circle",    3, {0.0,  0.0, 4.0}, 0};  cobjs[0].restitution = 1.0;  // outer wall: circle at (0,0), r=4
    cobjs[3].obj = {"LeftCup",   3, {-1.5, 2.5, 0.5}, 0};  cobjs[3].restitution = 0.1;  // circle at (-1.5,2.5), r=0.5 ...
    cobjs[3].layer_id = 1;                                                              // ... collides on its lower half only (layer 1)
    cobjs[4].obj = {"RightCup",  3, { 1.5, 2.5, 0.5}, 0};  cobjs[4].restitution = 0.1;  // circle at (1.5,2.5), r=0.5 ...
    cobjs[4].layer_id = 1;                                                              // ... collides on its lower half only (layer 1)

    // ── simulation ─────────────────────────────────────────────────────────────
    // Particles never interact, so each one runs its whole time loop
    // independently — one GPU thread per particle.
    //
    // Per time step, collisions are handled explicitly here in the loop:
    //   1. integrate:  s_new  = singleStep(p, dt_left)      (Yoshida 4, no collisions)
    //   2. detect:     did the path p → s_new cross a surface? (earliest hit wins;
    //                  the returned impact state is on the NEAR side of the surface)
    //   3. resolve:    reflect velocity at the impact point (restitution applied;
    //                  near-zero normal speed is clamped → resting/sliding contact)
    // then continue with the remaining dt_left, up to MAX_BOUNCES_PER_STEP contacts.


    




    #pragma omp target teams distribute parallel for \
            map(to: cobjs[0:5], layers[0:N_L]) map(tofrom: ps[0:N_P])
    for (int i = 0; i < N_P; ++i) {
        state  p = ps[i];
        double t = 0.0;   // elapsed simulated time of particle i
        while (t < t_end) {                                    // time loop
            double dt_left = dt;
            for (int b = 0; b < MAX_BOUNCES_PER_STEP; ++b) {
                state s_new = singleStep(p, dt_left);                                  // 1. integrate
                collisionEvent ev = detectFirstCollision(p, s_new, cobjs, 5, layers, dt_left); // 2. detect
                if (ev.obj_idx < 0) { p = s_new; break; }      // no (more) contacts this step
                // ── collision event hook: per-collision logic goes here ──
                p = resolveCollision(ev, cobjs[ev.obj_idx], p.mass, dt_left);          // 3. resolve
                dt_left -= ev.dt_hit;
                if (dt_left <= 0.0) break;
            }
            t += dt;
        }
        ps[i] = p;   // final state back to host
    }

    // ── write start → end positions ─────────────────────────────────────────────
    // CSV, one row per particle (i.e. per starting condition): x0,y0,xf,yf
    FILE* fout = fopen("cup_test.csv", "w");
    fprintf(fout, "x0,y0,xf,yf\n");
    for (int i = 0; i < N_P; ++i)
        fprintf(fout, "%.6f,%.6f,%.6f,%.6f\n",
                ps0[i].position.x, ps0[i].position.y,
                ps[i].position.x,  ps[i].position.y);
    fclose(fout);
    printf("Wrote %d particles to cup_test.csv\n", N_P);

    free(ps0);
    free(ps);
    return 0;
}
