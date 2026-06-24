
#define MAX_PARAMS 10 // maximun number of parameters for the equations defining the collisionable objects
#define MAX_


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <random>
/*
compile with:
    g++ -O2 -fopenmp -o integrator integrator2.cpp -lm
ran with:
    ./integrator


*/ 



struct vector2D {
    /*
    Basic struct for vector in 2D space.
    */
    double x, y;
    vector2D(double x, double y) : x(x), y(y) {}
};

struct particle {
    /*
    Basic struct for particle with position and velocity and unit mass
    */
    vector2D position;
    vector2D velocity;
    double mass = 1.0; // Assuming unit mass for simplicity
    particle(double x, double y, double vx, double vy, double m = 1.0) : position(x, y), velocity(vx, vy), mass(m) {}
};






vector2D reflect(const vector2D& velocity, const vector2D& normal) {
    // Reflect velocity across the plane defined by the normal
    double dot = velocity.x * normal.x + velocity.y * normal.y;
    return vector2D(velocity.x - 2 * dot * normal.x, velocity.y - 2 * dot * normal.y);
}

particle eulerStep(const particle& p, const vector2D& force, double dt) {
    // Update velocity based on force
    vector2D new_velocity(p.velocity.x + force.x * dt, p.velocity.y + force.y * dt);
    // Update position based on new velocity
    vector2D new_position(p.position.x + new_velocity.x * dt, p.position.y + new_velocity.y * dt);
    return particle(new_position.x, new_position.y, new_velocity.x, new_velocity.y, p.mass);
}

particle rk4Step(const particle& p, const vector2D& force, double dt) {
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

    return particle(new_position.x, new_position.y, new_velocity.x, new_velocity.y, p.mass);
}

vector2D force_at_position(double x, double y) {

    return vector2D(0, -1);
}

particle yoshida4Step(const particle& p, double dt) {
    const double cbrt2 = pow(2.0, 1.0/3.0);
    const double w1    = 1.0 / (2.0 - cbrt2);
    const double w0    = -cbrt2 * w1;

    const double c1 = w1 / 2.0;
    const double c2 = (w0 + w1) / 2.0;
    const double d1 = w1;
    const double d2 = w0;

    // Sub-step 1 — position drift, then velocity kick at new position
    double x1  = p.position.x + c1 * dt * p.velocity.x;
    double y1  = p.position.y + c1 * dt * p.velocity.y;
    vector2D f1 = force_at_position(x1, y1);
    double vx1 = p.velocity.x + d1 * dt * f1.x;
    double vy1 = p.velocity.y + d1 * dt * f1.y;

    // Sub-step 2 — position drift, then velocity kick at new position
    double x2  = x1 + c2 * dt * vx1;
    double y2  = y1 + c2 * dt * vy1;
    vector2D f2 = force_at_position(x2, y2);
    double vx2 = vx1 + d2 * dt * f2.x;
    double vy2 = vy1 + d2 * dt * f2.y;

    // Sub-step 3 — position drift, then velocity kick at new position (c3=c2, d3=d1)
    double x3  = x2 + c2 * dt * vx2;
    double y3  = y2 + c2 * dt * vy2;
    vector2D f3 = force_at_position(x3, y3);
    double vx3 = vx2 + d1 * dt * f3.x;
    double vy3 = vy2 + d1 * dt * f3.y;

    // Final position drift (c4 = c1)
    double x4  = x3 + c1 * dt * vx3;
    double y4  = y3 + c1 * dt * vy3;

    return particle(x4, y4, vx3, vy3, p.mass);
}




struct collisionableObject {
    double p[MAX_PARAMS] = {};   // any length up to MAX, flat, no heap
    int    n = 0;                // how many are actually used
    double (*implicit_form)(const double*, double, double) = nullptr;
    double (*dx)(const double*, double, double) = nullptr;
    double (*dy)(const double*, double, double) = nullptr;
};


vector2D calculateNormal(const collisionableObject& obj, double x, double y) {
    // Calculate the normal vector at point (x, y) on the surface defined by the implicit function
    double dFdx = obj.dx(obj.p, x, y);
    double dFdy = obj.dy(obj.p, x, y);
    double length = std::sqrt(dFdx * dFdx + dFdy * dFdy);
    double signal = obj.implicit_form(obj.p, x, y) < 0 ? -1 : 1; // Determine the sign based on whether the point is inside or outside
    dFdx *= signal;
    dFdy *= signal;
    return vector2D(dFdx / length, dFdy / length); // Normalize the normal vector
}






void printTest(const collisionableObject& obj, double x, double y) {
    // Print the value of the implicit function at point (x, y)
    vector2D normal = calculateNormal(obj, x, y);
    double F = obj.implicit_form(obj.p, x, y);
    if (F < 0) {
        printf("Point (%g, %g) is inside (F = %g)\n", x, y, F);
    } else if (F > 0) {
        printf("Point (%g, %g) is outside  (F = %g)\n", x, y, F);
    } else {
        printf("Point (%g, %g) is on the surface (F = %g)\n", x, y, F);
    }
    
    printf("Normal at (%g, %g): (%g, %g)\n", x, y, normal.x, normal.y);
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




int main(){
    collisionableObject circle;
    // circle:  p = {cx, cy, r}


    circle.n = 3;       // number of parameters for the circle
    circle.p[0] = 0;   // center x
    circle.p[1] = 0;   // center y
    circle.p[2] = 1;   // radius
    circle.implicit_form = circle_F;
    circle.dx = circle_dx;
    circle.dy = circle_dy;

    double x = 0.0;
    
    for (double y = -1.5; y <= 1.5; y += 0.5) {
    printf("===========\n");
    
    printf("Testing circle at (%g, %g)\n", x, y);
    printTest(circle, x, y);

    collisionableObject line;
    line.n = 2;       // number of parameters for the line
    line.p[0] = 0.5;   // slope (m)
    line.p[1] = 0.0;   // y-intercept (b)
    line.implicit_form = line_F;
    line.dx = line_dx;
    line.dy = line_dy;

    printf("\nTesting line at (%g, %g)\n", x, y);
    printTest(line, x, y);

    collisionableObject sine;
    sine.n = 3;       // number of parameters for the sine wave
    sine.p[0] = 1.0;   // amplitude (A)
    sine.p[1] = 1.0;   // wave number (k)
    sine.p[2] = 0.0;   // phase (phi)
    sine.implicit_form = sine_F;
    sine.dx = sine_dx;
    sine.dy = sine_dy;

    printf("\nTesting sine wave at (%g, %g)\n", x, y);
    printTest(sine, x, y);
    }

    return 0;
}