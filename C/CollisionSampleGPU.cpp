
// integrator_sample_gpu.cpp — GPU version: OpenMP target offload,
// one GPU thread per particle. CPU version: integrator_sample_cpu.cpp.
//
// Simulation only, no I/O: particles are advanced from t = 0 to t_end and
// their final states are left in ps[] (mapped back from the GPU).

#define MAX_PARAMS 10   // max parameters per shape


// load basic libs (host-only)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

// everything included inside this block is compiled for BOTH host and GPU;
// only GPU-safe code may live here (no I/O, no allocation, no function pointers)
#pragma omp declare target
#include "types.h"      // vector2D, state, equationSet, collisionEvent, CollisionableObject
#include "sampleTable.h"  // implicit shape functions + integer-ID dispatch (eval_F/eval_dx/eval_dy)
#include "physics.h"    // yoshida4Step, singleStep, detectFirstCollision, resolveCollision
#pragma omp end declare target

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
    const double dt    = 1e-3;   // integration time step
    const double t_end = 10000.0;    // total simulated time
    const int    N_P   = 10;    // number of particles

    // ── initial conditions ─────────────────────────────────────────────────────
    // All particles start above the touch point of the circles; vx is swept
    // linearly from 0 (particle 0, falls straight down) to 0.5 (particle N_P-1).
    state* ps = (state*)malloc(N_P * sizeof(state));
    for (int i = 0; i < N_P; ++i) {
        double vx = 1.0 * i / (N_P - 1);
        ps[i] = state(0.0, 2.0, vx, 0.0);
    }

    const int N_OBJ = 6;
    CollisionableObject cobjs[N_OBJ];
    cobjs[0].obj = {"CircleL", 3, {-1.0, 0.0, 1.0}, 0};  // touching circles at (±1,0), r=1,
    cobjs[1].obj = {"CircleR", 3, { 1.0, 0.0, 1.0}, 0};  // touch point = origin
    cobjs[2].obj = {"Floor",   2, {0.0,  3.0}, 1};       // y = -3
    cobjs[3].obj = {"Ceil",    2, {0.0, -3.0}, 1};       // y =  3
    cobjs[4].obj = {"WallL",   1, {-3.0}, 3};            // x = -3 (vertical line shape)
    cobjs[5].obj = {"WallR",   1, { 3.0}, 3};            // x =  3

    int N_REFLECTIONS = 10000;   // max collisions recorded per particle

    // 6 buffers: for times, IDs, x, y, vx, vy
    // Each buffer has size: N_P * N_REFLECTIONS * sizeof(double)
    // Particle i owns the slice [i*N_REFLECTIONS, (i+1)*N_REFLECTIONS) in every
    // buffer, so no two threads ever touch the same slot — no atomics needed.
    int BUF_SZ = N_P * N_REFLECTIONS;
    double *buffer_times = (double*)malloc(BUF_SZ * sizeof(double));
    double *buffer_IDS = (double*)malloc(BUF_SZ * sizeof(double));
    double *buffer_x = (double*)malloc(BUF_SZ * sizeof(double));
    double *buffer_y = (double*)malloc(BUF_SZ * sizeof(double));
    double *buffer_vx = (double*)malloc(BUF_SZ * sizeof(double));
    double *buffer_vy = (double*)malloc(BUF_SZ * sizeof(double));
    // how many slots of its slice each particle actually filled
    int *buffer_n = (int*)malloc(N_P * sizeof(int));

    // the buffers are written on the device and read on the host -> map(from:)
    #pragma omp target teams distribute parallel for \
        map(to: cobjs[0:N_OBJ]) \
        map(from: buffer_times[0:BUF_SZ], buffer_IDS[0:BUF_SZ], \
                    buffer_x[0:BUF_SZ], buffer_y[0:BUF_SZ], \
                    buffer_vx[0:BUF_SZ], buffer_vy[0:BUF_SZ], buffer_n[0:N_P]) \
        map(tofrom: ps[0:N_P])


    for (int i = 0; i < N_P; ++i) {
        state  p = ps[i];
        double t = 0.0;   // elapsed simulated time of particle i
        int    k = 0;     // collisions recorded so far for particle i
        while (t < t_end) {                                    // time loop
            // Bounces are not capped: the step is subdivided until its time is
            // spent. Termination rests on every continuing pass consuming a
            // strictly positive dt_hit — see the resting-contact break below.
            double dt_left = dt;
            while (dt_left > 0.0) {
                state s_new = singleStep(p, dt_left);                                  // 1. integrate
                collisionEvent ev = detectFirstCollision(p, s_new, cobjs, N_OBJ, dt_left); // 2. detect
                if (ev.obj_idx < 0) { p = s_new; break; }      // no (more) contacts this step
                p = resolveCollision(ev, cobjs[ev.obj_idx], p.mass, dt_left);          // 3. resolve
                bool resting = (ev.dt_hit <= 0.0);
                if (!resting) dt_left -= ev.dt_hit;   // the step's remaining time shrinks
                // ── collision event hook: per-collision logic goes here ──
                // p now holds the impact point and the OUTGOING velocity;
                // dt - dt_left is the time consumed within this step up to impact.
                if (k < N_REFLECTIONS) {
                    int slot = i * N_REFLECTIONS + k;
                    buffer_IDS  [slot] = i;
                    buffer_times[slot] = t + (dt - dt_left);   // absolute time of impact
                    buffer_x    [slot] = p.position.x;
                    buffer_y    [slot] = p.position.y;
                    buffer_vx   [slot] = p.velocity.x;
                    buffer_vy   [slot] = p.velocity.y;
                    ++k;
                }
                // dt_hit == 0 means the particle was already touching at step start:
                // resolving again would reproduce this exact state and never exit.
                if (resting) break;
            }
            t += dt;
        }
        ps[i] = p;   // final state back to host
        buffer_n[i] = k;
    }

    // ── write the collision table (host-side: the device cannot do I/O) ────────
    // %.17g round-trips a double exactly, so CPU and GPU tables can be diffed
    // literally rather than at printed precision.
    FILE *f = fopen("collisions_gpu.txt", "w");
    if (!f) { perror("collisions_gpu.txt"); return 1; }
    fprintf(f, "particle_ID,t,x,y,vx,vy\n");
    long long rows = 0, full = 0;
    for (int i = 0; i < N_P; ++i) {
        if (buffer_n[i] >= N_REFLECTIONS) ++full;
        for (int k = 0; k < buffer_n[i]; ++k) {
            int slot = i * N_REFLECTIONS + k;
            fprintf(f, "%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    (int)buffer_IDS[slot], buffer_times[slot],
                    buffer_x[slot], buffer_y[slot],
                    buffer_vx[slot], buffer_vy[slot]);
            ++rows;
        }
    }
    fclose(f);

    printf("collisions_gpu.txt: %lld rows\n", rows);
    if (full)
        printf("WARNING: %lld particle(s) hit the %d-slot limit — their logs are "
               "truncated; raise N_REFLECTIONS to keep the rest\n", full, N_REFLECTIONS);

    free(buffer_times);
    free(buffer_IDS);
    free(buffer_x);
    free(buffer_y);
    free(buffer_vx);
    free(buffer_vy);
    free(buffer_n);
    free(ps);
    return 0;
}
