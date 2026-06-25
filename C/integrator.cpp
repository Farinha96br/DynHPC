
#define MAX_PARAMS 10 // maximun number of parameters for the equations defining the collisionable objects
#define MAX_MASKS  8  // maximum number of no-collision masks per CollisionableObject


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>

/*
compile with:
    g++ -O2 -fopenmp -o integrator integrator.cpp -lm
ran with:
    ./integrator

a
*/ 



struct vector2D {
    /*
    Basic struct for vector in 2D space.
    */
    double x, y;
    vector2D(double x, double y) : x(x), y(y) {}
};

struct vector3D {
    double x, y, z;
    vector3D(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct state {
    /*
    Basic struct for state with position and velocity and unit mass
    */
    vector2D position;
    vector2D velocity;
    double mass = 1.0; // Assuming unit mass for simplicity
    state(double x, double y, double vx, double vy, double m = 1.0) : position(x, y), velocity(vx, vy), mass(m) {}
};






vector2D reflect(const vector2D& velocity, const vector2D& normal) {
    // Reflect velocity across the plane defined by the normal
    double dot = velocity.x * normal.x + velocity.y * normal.y;
    return vector2D(velocity.x - 2 * dot * normal.x, velocity.y - 2 * dot * normal.y);
}

state eulerStep(const state& p, const vector2D& force, double dt) {
    // Update velocity based on force
    vector2D new_velocity(p.velocity.x + force.x * dt, p.velocity.y + force.y * dt);
    // Update position based on new velocity
    vector2D new_position(p.position.x + new_velocity.x * dt, p.position.y + new_velocity.y * dt);
    return state(new_position.x, new_position.y, new_velocity.x, new_velocity.y, p.mass);
}

state rk4Step(const state& p, const vector2D& force, double dt) {
    // For simplicity, we will assume the force is constant over the time step.
    vector2D k1_v = force;
    vector2D k1_p = p.velocity;

    vector2D k2_v = force; // Assuming constant force
    vector2D k2_p = vector2D(p.velocity.x + 0.5 * k1_v.x * dt, p.velocity.y + 0.5 * k1_v.y * dt);

    vector2D k3_v = force; // Assuming constant force
    vector2D k3_p = vector2D(p.velocity.x + 0.5 * k2_v.x * dt, p.velocity.y + 0.5 * k2_v.y * dt);

    vector2D k4_v = force; // Assuming constant force
    vector2D k4_p = vector2D(p.velocity.x + k3_v.x * dt, p.velocity.y + k3_v.y * dt);

    vector2D new_velocity(
        p.velocity.x + (dt / 6.0) * (k1_v.x + 2 * k2_v.x + 2 * k3_v.x + k4_v.x),
        p.velocity.y + (dt / 6.0) * (k1_v.y + 2 * k2_v.y + 2 * k3_v.y + k4_v.y)
    );

    vector2D new_position(
        p.position.x + (dt / 6.0) * (k1_p.x + 2 * k2_p.x + 2 * k3_p.x + k4_p.x),
        p.position.y + (dt / 6.0) * (k1_p.y + 2 * k2_p.y + 2 * k3_p.y + k4_p.y)
    );

    return state(new_position.x, new_position.y, new_velocity.x, new_velocity.y, p.mass);
}

vector2D force_at_position(double x, double y, double t = 0.0) {
    // swirl force: rotational force around the origin

    double fx = 0.0; // Avoid division by zero
    double fy = -1.0;

    return vector2D(fx, fy);
}

state yoshida4Step(const state& p, double t, double dt) {
    const double cbrt2 = pow(2.0, 1.0/3.0);
    const double w1    = 1.0 / (2.0 - cbrt2);
    const double w0    = -cbrt2 * w1;

    const double c1 = w1 / 2.0;
    const double c2 = (w0 + w1) / 2.0;
    const double d1 = w1;
    const double d2 = w0;

    // Sub-step 1 — position drift, then velocity kick; kick time = t + c1*dt
    double x1  = p.position.x + c1 * dt * p.velocity.x;
    double y1  = p.position.y + c1 * dt * p.velocity.y;
    vector2D f1 = force_at_position(x1, y1, t + c1 * dt);
    double vx1 = p.velocity.x + d1 * dt * f1.x;
    double vy1 = p.velocity.y + d1 * dt * f1.y;

    // Sub-step 2 — position drift, then velocity kick; kick time = t + (c1+c2)*dt
    double x2  = x1 + c2 * dt * vx1;
    double y2  = y1 + c2 * dt * vy1;
    vector2D f2 = force_at_position(x2, y2, t + (c1 + c2) * dt);
    double vx2 = vx1 + d2 * dt * f2.x;
    double vy2 = vy1 + d2 * dt * f2.y;

    // Sub-step 3 — position drift, then velocity kick; kick time = t + (c1+2*c2)*dt
    double x3  = x2 + c2 * dt * vx2;
    double y3  = y2 + c2 * dt * vy2;
    vector2D f3 = force_at_position(x3, y3, t + (c1 + 2*c2) * dt);
    double vx3 = vx2 + d1 * dt * f3.x;
    double vy3 = vy2 + d1 * dt * f3.y;

    // Final position drift (c4 = c1)
    double x4  = x3 + c1 * dt * vx3;
    double y4  = y3 + c1 * dt * vy3;

    return state(x4, y4, vx3, vy3, p.mass);
}




struct equationSet{
    // struct to hold the equations defining a collisionable object
    char name[64] = "dummyName";          // name of the object
    double p[MAX_PARAMS] = {};   // any length up to MAX, flat, no heap
    int    n = 0;                // how many are actually used
    double (*implicit_form)(const double*, double, double) = nullptr;
    double (*dx)(const double*, double, double) = nullptr;
    double (*dy)(const double*, double, double) = nullptr;
};


vector2D calculateNormal(const equationSet& obj, double x, double y) {
    /* Calculate the normal vector at point (x, y) on the surface defined by the implicit function
    Compunting the normal on top of the line results in a normal OUTWARDS
    */
    double dFdx = obj.dx(obj.p, x, y);
    double dFdy = obj.dy(obj.p, x, y);
    double length = std::sqrt(dFdx * dFdx + dFdy * dFdy);
    double signal = obj.implicit_form(obj.p, x, y) < 0 ? -1 : 1; // Determine the sign based on whether the point is inside or outside
    dFdx *= signal;
    dFdy *= signal;
    return vector2D(dFdx / length, dFdy / length); // Normalize the normal vector
}

struct mask {
    // A no-collision region: collision is suppressed where F(p, x, y) < 0.
    double (*F)(const double*, double, double) = nullptr;
    double p[MAX_PARAMS] = {};
};

class CollisionableObject {
public:
    equationSet obj;
    double restitution = 1.0;
    mask   masks[MAX_MASKS];  // no-collision regions (holes in the surface)
    int    n_masks = 0;
};



state singleStep(state& p, const CollisionableObject* objs, int n_objs, double t, double dt, state (*integrator)(const state&, double, double)) {
    /*
    Advances p by dt using the given integrator, resolving at most one collision
    (the earliest one across all objects). Any remaining time after the bounce is
    advanced freely; subsequent collisions are caught in the next call.
    */
    state s_new = integrator(p, t, dt);

    // Find the earliest collision among all objects
    int    hit_idx = -1;
    double dt_hit  = dt;
    state  s_hit   = s_new;

    for (int i = 0; i < n_objs; ++i) {
        // loops through all objects
        double F_old = objs[i].obj.implicit_form(objs[i].obj.p, p.position.x,     p.position.y);
        double F_new = objs[i].obj.implicit_form(objs[i].obj.p, s_new.position.x, s_new.position.y);

        // if the sign changes the particle has crossed a surface
        if (F_old * F_new >= 0.0) continue;

        // process of smallification of the timestep
        double dt_small = 0.0, dt_large = dt;
        state  s_surf = s_new;
        for (int iter = 0; iter < 32; ++iter) {
            // keep halving the timestep
            double dt_mid = 0.5 * (dt_small + dt_large);
            state  s_mid  = integrator(p, t, dt_mid);
            double F_mid  = objs[i].obj.implicit_form(objs[i].obj.p, s_mid.position.x, s_mid.position.y);
            if (F_old * F_mid < 0.0) { dt_large = dt_mid; s_surf = s_mid; }
            else                      { dt_small = dt_mid; }
        }

        // Suppress collision if the impact point is inside any no-collision mask
        bool masked = false;
        for (int m = 0; m < objs[i].n_masks && !masked; ++m)
            if (objs[i].masks[m].F(objs[i].masks[m].p, s_surf.position.x, s_surf.position.y) < 0.0)
                masked = true;
        if (masked) continue;

        if (dt_large < dt_hit) { hit_idx = i; dt_hit = dt_large; s_hit = s_surf; }
    }

    if (hit_idx < 0) return s_new;

    // Reflect velocity off the earliest-hit surface, apply restitution
    vector2D normal = calculateNormal(objs[hit_idx].obj, s_hit.position.x, s_hit.position.y);
    vector2D v_ref  = reflect(s_hit.velocity, normal);
    state s_bounced(s_hit.position.x, s_hit.position.y,
                    objs[hit_idx].restitution * v_ref.x,
                    objs[hit_idx].restitution * v_ref.y,
                    p.mass);

    return integrator(s_bounced, t + dt_hit, dt - dt_hit);
}

















void printTest(const equationSet& obj, double x, double y) {
    // Print the value of the implicit function at point (x, y)
    
    bool inside = obj.implicit_form(obj.p, x, y) < 0;
    vector2D normal = calculateNormal(obj, x, y);
    printf("obj: %s x,y: %g,%g, nx, ny: %g, %g, inside: %s\n", obj.name, x, y, normal.x, normal.y, inside ? "true" : "false");   
}


// define the equations of the circle
double circle_F (const double* p, double x, double y){ double delta_x=x-p[0], delta_y=y-p[1]; return delta_x*delta_x + delta_y*delta_y - p[2]*p[2]; }
double circle_dx(const double* p, double x, double y){ return 2*(x-p[0]); }
double circle_dy(const double* p, double x, double y){ return 2*(y-p[1]); }

// define the equations of a line (for example, y - m*x - b = 0)
double line_F (const double* p, double x, double y){ return y - p[0]*x + p[1]; } // p[0] = slope (m), p[1] = y-intercept (b)
double line_dx(const double* p, double x, double y){ return -p[0]; } // derivative with respect to x
double line_dy(const double* p, double x, double y){ return 1; } // derivative with respect to y

// define a sine wave (for example, y - A*sin(k*x + phi) = 0)
double sine_F (const double* p, double x, double y){ return y - p[0]*sin(p[1]*x + p[2]); } // p[0] = amplitude (A), p[1] = wave number (k), p[2] = phase (phi)
double sine_dx(const double* p, double x, double y){ return -p[0]*p[1]*cos(p[1]*x + p[2]); }
double sine_dy(const double* p, double x, double y){ return 1; }

// Half-plane masks for suppressing one side of a vertical line x = p[0]
// F < 0 on the side that is suppressed
double mask_right_F(const double* p, double x, double y){ return p[0] - x; } // F < 0 when x > p[0] → suppresses right side
double mask_left_F (const double* p, double x, double y){ return x - p[0]; } // F < 0 when x < p[0] → suppresses left side




int main(){
    const double dt      = 1e-5;
    const double t_end   = 10.0;
    const int    n_steps = (int)(t_end / dt);

    // 3 particles inside the bounding circle, above the floor and sine wave
    state ps[3] = {
        state( 0.0,  2.0,  1.5,  1.0),
        state(-1.5,  1.0, -1.0,  1.5),
        state( 1.5,  2.0,  100.0,  2.5)
    };

    // Collision objects — all restitution = 1 for energy conservation test
    CollisionableObject cobjs[5];
    cobjs[0].obj         = {"Circle",    {0.0,  0.0, 4.0}, 3, circle_F, circle_dx, circle_dy};
    cobjs[0].restitution = 1.0;
    cobjs[1].obj         = {"Line",      {0.0,  2.0},      2, line_F,   line_dx,   line_dy};   // floor at y = -2
    cobjs[1].restitution = 1.0;
    cobjs[2].obj         = {"Sine",      {1.0,  1.0, 0.0}, 3, sine_F,   sine_dx,   sine_dy};   // y = sin(x)
    cobjs[2].restitution = 1.0;
    // Small circle — only LEFT side collisionable; right side (x > cx) is masked
    cobjs[3].obj            = {"LeftOnly",  {-1.5, 2.5, 0.5}, 3, circle_F, circle_dx, circle_dy};
    cobjs[3].restitution    = 1.0;
    cobjs[3].masks[0].F     = mask_right_F;  // F < 0 when x > -1.5 → suppresses right side
    cobjs[3].masks[0].p[0]  = -1.5;
    cobjs[3].n_masks        = 1;
    // Small circle — only RIGHT side collisionable; left side (x < cx) is masked
    cobjs[4].obj            = {"RightOnly", { 1.5, 2.5, 0.5}, 3, circle_F, circle_dx, circle_dy};
    cobjs[4].restitution    = 1.0;
    cobjs[4].masks[0].F     = mask_left_F;   // F < 0 when x < 1.5 → suppresses left side
    cobjs[4].masks[0].p[0]  = 1.5;
    cobjs[4].n_masks        = 1;

    FILE* fout = fopen("trajectories.txt", "w");
    fprintf(fout, "t,particle,x,y,vx,vy,energy\n");

    // Outer loop over particles, inner loop over timesteps
    for (int i = 0; i < 3; ++i) {
        printf("Simulating particle %d\n", i);
        for (int step = 0; step <= n_steps; ++step) {
            double t = step * dt;
            double vx = ps[i].velocity.x, vy = ps[i].velocity.y;
            double y  = ps[i].position.y;
            // Total energy = kinetic + potential. For the current force field
            // force = (0, -1) = -grad V => dV/dy = 1 => V = y (up to constant)
            double V  = ps[i].position.y;
            double E  = 0.5 * (vx*vx + vy*vy) + V;
            fprintf(fout, "%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.10f\n",
                    t, i, ps[i].position.x, y, vx, vy, E);

            if (step < n_steps)
                ps[i] = singleStep(ps[i], cobjs, 5, t, dt, yoshida4Step);
        }
    }

    fclose(fout);
    printf("Done. Wrote trajectories.txt\n");
    return 0;
}