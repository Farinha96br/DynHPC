
#define MAX_PARAMS 10   // max parameters per shape / mask
#define MAX_MASKS   8   // max no-collision masks per CollisionableObject

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

// ── all device-visible code ───────────────────────────────────────────────────

#pragma omp declare target

// ── value types ───────────────────────────────────────────────────────────────

struct vector2D {
    double x, y;
    vector2D(double x, double y) : x(x), y(y) {}
};

struct state {
    vector2D position;
    vector2D velocity;
    double   mass = 1.0;
    state(double x, double y, double vx, double vy, double m = 1.0)
        : position(x, y), velocity(vx, vy), mass(m) {}
};

// ── structs ───────────────────────────────────────────────────────────────────

struct equationSet {
    char   name[64]      = "dummyName";
    double p[MAX_PARAMS] = {};
    int    n             = 0;
    int    shape_id      = -1;   // registered shape index (see eval_F/eval_dx/eval_dy)
};

struct mask {
    int    mask_id       = -1;   // registered mask index (see eval_mask); -1 = inactive
    double p[MAX_PARAMS] = {};
};

class CollisionableObject {
public:
    equationSet obj;
    double      restitution = 1.0;
    mask        masks[MAX_MASKS];
    int         n_masks = 0;
};

// ── shape functions ───────────────────────────────────────────────────────────

// shape IDs: 0 = circle, 1 = line, 2 = sine
double circle_F (const double* p, double x, double y){ double dx=x-p[0], dy=y-p[1]; return dx*dx+dy*dy-p[2]*p[2]; }
double circle_dx(const double* p, double x, double y){ return 2.0*(x-p[0]); }
double circle_dy(const double* p, double x, double y){ return 2.0*(y-p[1]); }

double line_F (const double* p, double x, double y){ return y - p[0]*x + p[1]; }
double line_dx(const double* p, double x, double y){ return -p[0]; }
double line_dy(const double* p, double x, double y){ return 1.0; }

double sine_F (const double* p, double x, double y){ return y - p[0]*sin(p[1]*x + p[2]); }
double sine_dx(const double* p, double x, double y){ return -p[0]*p[1]*cos(p[1]*x + p[2]); }
double sine_dy(const double* p, double x, double y){ return 1.0; }

// mask IDs: 0 = mask_right_F (suppresses right side), 1 = mask_left_F (suppresses left side)
double mask_right_F(const double* p, double x, double y){ return p[0] - x; }
double mask_left_F (const double* p, double x, double y){ return x - p[0]; }

// ── integer-dispatch wrappers (GPU-compatible, no function pointers) ──────────

double eval_F(int id, const double* p, double x, double y) {
    switch (id) {
        case 0: return circle_F(p, x, y);
        case 1: return line_F  (p, x, y);
        case 2: return sine_F  (p, x, y);
        default: return 0.0;
    }
}
double eval_dx(int id, const double* p, double x, double y) {
    switch (id) {
        case 0: return circle_dx(p, x, y);
        case 1: return line_dx  (p, x, y);
        case 2: return sine_dx  (p, x, y);
        default: return 0.0;
    }
}
double eval_dy(int id, const double* p, double x, double y) {
    switch (id) {
        case 0: return circle_dy(p, x, y);
        case 1: return line_dy  (p, x, y);
        case 2: return sine_dy  (p, x, y);
        default: return 0.0;
    }
}
double eval_mask(int id, const double* p, double x, double y) {
    switch (id) {
        case 0: return mask_right_F(p, x, y);
        case 1: return mask_left_F (p, x, y);
        default: return 1.0;   // positive → no suppression
    }
}

// ── physics helpers ───────────────────────────────────────────────────────────

vector2D force_at_position(double x, double y) {
    return vector2D(0.0, -1.0);
}

vector2D reflect(const vector2D& v, const vector2D& n) {
    double dot = v.x*n.x + v.y*n.y;
    return vector2D(v.x - 2.0*dot*n.x, v.y - 2.0*dot*n.y);
}

vector2D calculateNormal(const equationSet& obj, double x, double y) {
    double dFdx = eval_dx(obj.shape_id, obj.p, x, y);
    double dFdy = eval_dy(obj.shape_id, obj.p, x, y);
    double len  = sqrt(dFdx*dFdx + dFdy*dFdy);
    double sig  = eval_F(obj.shape_id, obj.p, x, y) < 0.0 ? -1.0 : 1.0;
    return vector2D(sig*dFdx/len, sig*dFdy/len);
}

state yoshida4Step(const state& p, double dt) {
    const double cbrt2 = pow(2.0, 1.0/3.0);
    const double w1    = 1.0 / (2.0 - cbrt2);
    const double w0    = -cbrt2 * w1;
    const double c1 = w1/2.0, c2 = (w0+w1)/2.0, d1 = w1, d2 = w0;

    double x1 = p.position.x + c1*dt*p.velocity.x;
    double y1 = p.position.y + c1*dt*p.velocity.y;
    vector2D f1 = force_at_position(x1, y1);
    double vx1 = p.velocity.x + d1*dt*f1.x, vy1 = p.velocity.y + d1*dt*f1.y;

    double x2 = x1 + c2*dt*vx1, y2 = y1 + c2*dt*vy1;
    vector2D f2 = force_at_position(x2, y2);
    double vx2 = vx1 + d2*dt*f2.x, vy2 = vy1 + d2*dt*f2.y;

    double x3 = x2 + c2*dt*vx2, y3 = y2 + c2*dt*vy2;
    vector2D f3 = force_at_position(x3, y3);
    double vx3 = vx2 + d1*dt*f3.x, vy3 = vy2 + d1*dt*f3.y;

    double x4 = x3 + c1*dt*vx3, y4 = y3 + c1*dt*vy3;
    return state(x4, y4, vx3, vy3, p.mass);
}

state singleStep(const state& p, const CollisionableObject* objs, int n_objs, double dt) {
    state s_new = yoshida4Step(p, dt);

    int    hit_idx = -1;
    double dt_hit  = dt;
    state  s_hit   = s_new;

    for (int i = 0; i < n_objs; ++i) {
        double F_old = eval_F(objs[i].obj.shape_id, objs[i].obj.p, p.position.x,     p.position.y);
        double F_new = eval_F(objs[i].obj.shape_id, objs[i].obj.p, s_new.position.x, s_new.position.y);
        if (F_old * F_new >= 0.0) continue;

        double dt_small = 0.0, dt_large = dt;
        state  s_surf = s_new;
        for (int iter = 0; iter < 32; ++iter) {
            double dt_mid = 0.5*(dt_small + dt_large);
            state  s_mid  = yoshida4Step(p, dt_mid);
            double F_mid  = eval_F(objs[i].obj.shape_id, objs[i].obj.p, s_mid.position.x, s_mid.position.y);
            if (F_old * F_mid < 0.0) { dt_large = dt_mid; s_surf = s_mid; }
            else                      { dt_small = dt_mid; }
        }

        bool masked = false;
        for (int m = 0; m < objs[i].n_masks && !masked; ++m)
            if (eval_mask(objs[i].masks[m].mask_id, objs[i].masks[m].p, s_surf.position.x, s_surf.position.y) < 0.0)
                masked = true;
        if (masked) continue;

        if (dt_large < dt_hit) { hit_idx = i; dt_hit = dt_large; s_hit = s_surf; }
    }

    if (hit_idx < 0) return s_new;

    vector2D normal = calculateNormal(objs[hit_idx].obj, s_hit.position.x, s_hit.position.y);
    vector2D v_ref  = reflect(s_hit.velocity, normal);
    state s_bounced(s_hit.position.x, s_hit.position.y,
                    objs[hit_idx].restitution * v_ref.x,
                    objs[hit_idx].restitution * v_ref.y,
                    p.mass);

    return yoshida4Step(s_bounced, dt - dt_hit);
}

#pragma omp end declare target

// ── host-only helpers ─────────────────────────────────────────────────────────

void printTest(const equationSet& obj, double x, double y) {
    bool inside = eval_F(obj.shape_id, obj.p, x, y) < 0.0;
    vector2D normal = calculateNormal(obj, x, y);
    printf("obj: %s x,y: %g,%g, nx, ny: %g, %g, inside: %s\n",
           obj.name, x, y, normal.x, normal.y, inside ? "true" : "false");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    const double dt      = 1e-5;
    const double t_end   = 10.1;              // 10 000 steps — keeps final-state buffer small
    const int    n_steps = (int)(t_end / dt);
    const int    N_P     = 10000;

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
    cobjs[0].obj = {"Circle",    {0.0,  0.0, 4.0}, 3, 0};  cobjs[0].restitution = 1.0;
    cobjs[1].obj = {"Line",      {0.0,  2.0},      2, 1};  cobjs[1].restitution = 1.0;
    cobjs[2].obj = {"Sine",      {1.0,  1.0, 0.0}, 3, 2};  cobjs[2].restitution = 1.0;
    cobjs[3].obj = {"LeftOnly",  {-1.5, 2.5, 0.5}, 3, 0};  cobjs[3].restitution = 1.0;
    cobjs[3].masks[0] = {0, {-1.5}};  cobjs[3].n_masks = 1;
    cobjs[4].obj = {"RightOnly", { 1.5, 2.5, 0.5}, 3, 0};  cobjs[4].restitution = 1.0;
    cobjs[4].masks[0] = {1, { 1.5}};  cobjs[4].n_masks = 1;

    // ── trajectory buffer: N_TRAJ waypoints evenly spaced through the run ────────
    const int N_TRAJ        = 200;
    const int N_TRAJ_STRIDE = n_steps / N_TRAJ;
    double*   traj = new double[(long long)N_P * N_TRAJ * 2]();

    // ── simulation ─────────────────────────────────────────────────────────────
// if using the GPU
#ifdef GPU_OFFLOAD
    #pragma omp target data map(to: cobjs[0:5], ps[0:N_P]) \
                            map(from: traj[0:N_P * N_TRAJ * 2])
    {
        #pragma omp target teams distribute parallel for \
                num_teams((N_P + 255) / 256) thread_limit(256)
        for (int i = 0; i < N_P; ++i) {
            state p = ps[i];
            for (int t = 0; t < N_TRAJ; ++t) {
                for (int s = 0; s < N_TRAJ_STRIDE; ++s)
                    p = singleStep(p, cobjs, 5, dt);
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
        state p = ps[i];
        for (int t = 0; t < N_TRAJ; ++t) {
            for (int s = 0; s < N_TRAJ_STRIDE; ++s)
                p = singleStep(p, cobjs, 5, dt);
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
