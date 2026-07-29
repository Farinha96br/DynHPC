
// integrator_sample_cpu.cpp — CPU version: OpenMP multi-threading,
// one CPU thread per particle. GPU version: integrator_sample_gpu.cpp.
//
// Simulation only, no I/O: particles are advanced from t = 0 to t_end and
// their final states are left in ps[].

#define MAX_PARAMS 10   // max parameters per shape / mask
#define MAX_MASKS   8   // max no-collision masks per CollisionableObject


// load basic libs
#include <cmath>
#include <cstdlib>
#include <omp.h>

// simulation headers (shared with the GPU version)
#include "types.h"      // vector2D, state, equationSet, mask, collisionEvent, CollisionableObject
#include "sampleTable.h"  // implicit shape/mask functions + integer-ID dispatch (eval_F/eval_dx/eval_dy/eval_mask)
#include "physics.h"    // yoshida4Step, singleStep, detectFirstCollision, resolveCollision

/*
compile:
    g++ -O2 -fopenmp -o CPUbuild integrator_sample_cpu.cpp -lm
run:
    ./CPUbuild
*/


int main() {
    // ── simulation parameters ───────────────────────────────────────────────────
    const double dt    = 1e-4;   // integration time step
    const double t_end = 1.0;    // total simulated time
    const int    N_P   = 200;    // number of particles

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
    // Masks live on layers and carve out regions where NO object of that layer
    // collides: a hit is ignored when any of the layer's masks evaluates
    // negative at the impact point. Objects pick their layer via layer_id.
    Layer layers[3];                                       // layer 0: no masks (walls collide everywhere)
    layers[1].masks[0] = {0, {-1.5}};  layers[1].n_masks = 1;  // layer 1: masked for x > -1.5
    layers[2].masks[0] = {1, { 1.5}};  layers[2].n_masks = 1;  // layer 2: masked for x <  1.5

    CollisionableObject cobjs[5];
    cobjs[0].obj = {"Circle",    3, {0.0,  0.0, 4.0}, 0};  cobjs[0].restitution = 1.0;  // outer wall: circle at (0,0), r=4
    cobjs[1].obj = {"Line",      2, {0.0,  2.0},      1};  cobjs[1].restitution = 1.0;  // flat floor: y = 0*x - 2
    cobjs[2].obj = {"Sine",      3, {1.0,  1.0, 0.0}, 2};  cobjs[2].restitution = 1.0;  // wavy floor: y = 1*sin(1*x + 0)
    cobjs[3].obj = {"LeftOnly",  3, {-1.5, 2.5, 0.5}, 0};  cobjs[3].restitution = 1.0;  // circle at (-1.5,2.5), r=0.5 ...
    cobjs[3].layer_id = 1;                                                              // ... collides on its left half only (layer 1)
    cobjs[4].obj = {"RightOnly", 3, { 1.5, 2.5, 0.5}, 0};  cobjs[4].restitution = 1.0;  // circle at (1.5,2.5), r=0.5 ...
    cobjs[4].layer_id = 2;                                                              // ... collides on its right half only (layer 2)

    // ── simulation ─────────────────────────────────────────────────────────────
    // Particles never interact, so each one runs its whole time loop
    // independently — one CPU thread per particle.
    //
    // Per time step, collisions are handled explicitly here in the loop:
    //   1. integrate:  s_new  = singleStep(p, dt_left)      (Yoshida 4, no collisions)
    //   2. detect:     did the path p → s_new cross a surface? (earliest hit wins;
    //                  the returned impact state is on the NEAR side of the surface)
    //   3. resolve:    reflect velocity at the impact point (restitution applied;
    //                  near-zero normal speed is clamped → resting/sliding contact)
    // then continue with the remaining dt_left, up to MAX_BOUNCES_PER_STEP contacts.
    #pragma omp parallel for
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
        ps[i] = p;   // final state
    }

    free(ps);
    return 0;
}
