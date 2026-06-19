
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>



/*
compile with:
    g++ -O2 -fopenmp -o integrator integrator2.cpp -lm
ran with:
    ./integrator


*/ 



struct point {
    /*
    Basic struct for XY point pair.
    */
    double x, y;
    point(double x, double y) : x(x), y(y) {}
};

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
    point position;
    vector2D velocity;
    double mass = 1.0; // Assuming unit mass for simplicity
    particle(double x, double y, double vx, double vy, double m = 1.0) : position(x, y), velocity(vx, vy), mass(m) {}
};



vector2D reflect(const vector2D& velocity, const vector2D& normal) {
    // Reflect velocity across the plane defined by the normal
    double dot = velocity.x * normal.x + velocity.y * normal.y;
    return vector2D(velocity.x - 2 * dot * normal.x, velocity.y - 2 * dot * normal.y);
}













class elipse {
public:
    double Haxix, Vaxix; // semi-major and semi-minor axes
    double centerX, centerY; // center of the elipse
    double startCollisionAngle, endCollisionAngle; // angle range in radians in which the elipse is collidable (0 to 2*pi)
    elipse(double h, double v, double cx, double cy, double sca, double eca) : Haxix(h), Vaxix(v), centerX(cx), centerY(cy), startCollisionAngle(sca), endCollisionAngle(eca) {}
    // Outward unit normal at surface point (x, y): gradient of F = (x-cx)²/a² + (y-cy)²/b² - 1, normalized.
    vector2D normal(double x, double y) const {
        double dx = (x - centerX) / (Haxix * Haxix);
        double dy = (y - centerY) / (Vaxix * Vaxix);
        double length = sqrt(dx * dx + dy * dy);
        return vector2D(dx / length, dy / length);
    }

    bool is_inside(double x, double y) const {
        // Check if point (x, y) is inside the elipse: F < 0
        double check = pow((x - centerX) / Haxix, 2) + std::pow((y - centerY) / Vaxix, 2) - 1;
        return check < 0;
    }

    bool is_on_surface(double x, double y) const {
        // Check if point (x, y) is on the surface of the elipse: F = 0
        double check = pow((x - centerX) / Haxix, 2) + std::pow((y - centerY) / Vaxix, 2) - 1;
        return fabs(check) < 1e-6; // Tolerance for floating-point comparison
    }

    bool is_on_collidable_section(double x, double y) const {
        double angle = atan2(y - centerY, x - centerX);
        if (angle < 0) angle += 2 * M_PI;
        return (startCollisionAngle <= endCollisionAngle)
            ? (angle >= startCollisionAngle && angle <= endCollisionAngle)
            : (angle >= startCollisionAngle || angle <= endCollisionAngle);
    }

    // Returns true if a collision occurred and modifies p in place.
    // Encapsulates the crossing check so the time loop stays clean.
    bool handle_collision(particle& p, double old_x, double old_y) const {
        if (is_inside(old_x, old_y) != is_inside(p.position.x, p.position.y)
                && is_on_collidable_section(old_x, old_y)) {
            p.velocity   = reflect(p.velocity, normal(old_x, old_y));
            p.position.x = old_x;
            p.position.y = old_y;
            return true;
        }
        return false;
    }

};

class line {
public:
    // points that define the line
    double m, b; // slope and intercept
    line(double m, double b) : m(m), b(b) {}
    // Get the normal vector of the line
    vector2D normal() const {
        // For a line y = mx + b, the normal vector can be represented as (-m, 1) or (m, -1) depending on the direction. We will use (-m, 1) for the normal vector.
        // The outoward normal is "UP"
        double length = sqrt(m * m + 1);
        return vector2D(-m / length, 1 / length);
    }

    bool is_above(double x, double y) const {
        // Check if point (x, y) is above the line: y > mx + b
        return y > m * x + b;
    }

    bool is_on_line(double x, double y) const {
        // Check if point (x, y) is on the line: y = mx + b
        return fabs(y - (m * x + b)) < 1e-6; // Tolerance for floating-point comparison
    }

    // Returns true if a collision occurred and modifies p in place.
    bool handle_collision(particle& p, double old_x, double old_y) const {
        if (is_above(old_x, old_y) != is_above(p.position.x, p.position.y)) {
            p.velocity   = reflect(p.velocity, normal());
            p.position.x = old_x;
            p.position.y = old_y;
            return true;
        }
        return false;
    }
};








particle eulerStep(const particle& p, const vector2D& force, double dt) {
    // Update velocity based on force
    vector2D new_velocity(p.velocity.x + force.x * dt, p.velocity.y + force.y * dt);
    // Update position based on new velocity
    point new_position(p.position.x + new_velocity.x * dt, p.position.y + new_velocity.y * dt);
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

    point new_position(
        p.position.x + (dt / 6.0) * (k1_p.x + 2 * k2_p.x + 2 * k3_p.x + k4_p.x),
        p.position.y + (dt / 6.0) * (k1_p.y + 2 * k2_p.y + 2 * k3_p.y + k4_p.y)
    );

    return particle(new_position.x, new_position.y, new_velocity.x, new_velocity.y, p.mass);
}



vector2D force_at_position(double x, double y) {

    return vector2D(0, -1);
}





// Yoshida 4th-order symplectic integrator (Yoshida 1990).
// Coefficients: w1 = 1/(2 - 2^(1/3)), w0 = -2^(1/3)*w1
//   c positions: c1=c4=w1/2,  c2=c3=(w0+w1)/2
//   d velocities: d1=d3=w1,   d2=w0
// force argument removed: each velocity kick now queries force_at_position
// at the current intermediate position, making this correct for any
// position-dependent force field, not just constant gravity.
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

int main() {
    // create the elipse:

    elipse elipses[2] = {
        elipse(2.0, 1.0, 0.0,  0.0,  0,      2*M_PI),
        elipse(0.1, 0.2, 0.01, -0.01, M_PI/2, 3*M_PI/2)
    };

    line lines[1] = {
        line(0.0, -0.5) // A horizontal line at y = -0.5
    };


    // initializa an array of 3 particles:
    particle particles[3] = {
        particle(1.0,  0.5,  1,  0.2),
        particle(1.5,  0.5,  1,  0.5),
        particle(-0.3, -0.8,  1,  0.0)
    };

    double t0 = 0.0;
    double dt = 1e-5;
    double tf = 100.0;
    double t = t0;
    int step = 0;
    int strobe_interval = 0.01/dt;

    

    int n_particles = sizeof(particles) / sizeof(particles[0]);
    int n_elipses   = sizeof(elipses)   / sizeof(elipses[0]);
    int n_lines     = sizeof(lines)     / sizeof(lines[0]);

    FILE* f = fopen("trajectories.csv", "w");
    fprintf(f, "t");
    for (int i = 0; i < n_particles; i++)
        fprintf(f, ",x%d,y%d", i, i);
    fprintf(f, "\n");

    // temporal loop
    while (t < tf) {
        for (int i = 0; i < n_particles; i++) {
            double old_x = particles[i].position.x;
            double old_y = particles[i].position.y;

            particles[i] = yoshida4Step(particles[i], dt);

            // loop for elipse collisions
            for (int j = 0; j < n_elipses; j++)
                if (elipses[j].handle_collision(particles[i], old_x, old_y)) break;


            // loop for line collisions
            for (int j = 0; j < n_lines; j++)
                if (lines[j].handle_collision(particles[i], old_x, old_y)) break;
        }

        if (step % strobe_interval == 0) {
            fprintf(f, "%.6f", t);
            for (int i = 0; i < n_particles; i++)
                fprintf(f, ",%.6f,%.6f",
                        particles[i].position.x, particles[i].position.y);
            fprintf(f, "\n");
        }

        t += dt;
        step++;
    }

    fclose(f);
    return 0;
}