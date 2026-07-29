
// integrator_sample_gpu.cpp — GPU version: OpenMP target offload,
// one GPU thread per particle. CPU version: integrator_sample_cpu.cpp.
//
// Simulation only, no I/O: particles are advanced from t = 0 to t_end and
// their final states are left in ps[] (mapped back from the GPU).

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
#include "sampleTable.h"  // implicit shape/mask functions + integer-ID dispatch (eval_F/eval_dx/eval_dy/eval_mask)
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

// extern "C" keeps the symbol name unmangled, so ctypes can find "MakeMap".
extern "C"
void MakeMap(double dt, double t_end, int N_P, int N_REFLECTIONS, double *host_buffer_times, double *host_buffer_IDS, double *host_buffer_x, double *host_buffer_y, double *host_buffer_vx, double *host_buffer_vy, int *host_buffer_n) {
    // The caller (numpy) owns every output buffer; we only fill them, so nothing
    // has to be freed across the boundary.
    // Each buffer has size: N_P * N_REFLECTIONS * sizeof(double)
    // Particle i owns the slice [i*N_REFLECTIONS, (i+1)*N_REFLECTIONS) in every
    // buffer, so no two threads ever touch the same slot — no atomics needed.
    int BUF_SZ = N_P * N_REFLECTIONS;

    state* ps = (state*)malloc(N_P * sizeof(state));
    for (int i = 0; i < N_P; ++i) {
        double x = 0.5 * i / (N_P - 1);
        ps[i] = state(x, 0, 0.0, 0.0);
    }

    // For now make the collisionable objects fixed.
    // Declared before the pragma: the pragma must sit immediately above its for
    // loop, and cobjs has to exist before the map clause can name it.
    const int N_OBJ = 2;
    CollisionableObject cobjs[N_OBJ];
    cobjs[0].obj = {"Circle",    3, {0.0,  0.0  , 1.0}, 0};  cobjs[0].restitution = 1.0;  // wall: circle at (0,0), r=1
    cobjs[1].obj = {"Line",      2, {0.0,  -1.1},      1};  cobjs[1].restitution = 1.0;  // line: y = 1.1, outside the wall

    #pragma omp target teams distribute parallel for \
        map(to: cobjs[0:N_OBJ], ps[0:N_P])          \
        map(from: host_buffer_times[0:BUF_SZ], host_buffer_IDS[0:BUF_SZ], \
                  host_buffer_x[0:BUF_SZ], host_buffer_y[0:BUF_SZ], \
                  host_buffer_vx[0:BUF_SZ], host_buffer_vy[0:BUF_SZ], \
                  host_buffer_n[0:N_P])
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
                    host_buffer_IDS  [slot] = i;
                    host_buffer_times[slot] = t + (dt - dt_left);   // absolute time of impact
                    host_buffer_x    [slot] = p.position.x;
                    host_buffer_y    [slot] = p.position.y;
                    host_buffer_vx   [slot] = p.velocity.x;
                    host_buffer_vy   [slot] = p.velocity.y;
                    ++k;
                }
                // dt_hit == 0 means the particle was already touching at step start:
                // resolving again would reproduce this exact state and never exit.
                if (resting) break;
            }
            t += dt;
        }
        ps[i] = p;   // final state back to host
        host_buffer_n[i] = k;
    }

    free(ps);   // the output buffers belong to the caller — do not free them
}


