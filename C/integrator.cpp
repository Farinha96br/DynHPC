
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

};












particle eulerStep(const particle& p, const vector2D& force, double dt) {
    // Update velocity based on force
    vector2D new_velocity(p.velocity.x + force.x * dt, p.velocity.y + force.y * dt);
    // Update position based on new velocity
    point new_position(p.position.x + new_velocity.x * dt, p.position.y + new_velocity.y * dt);
    return particle(new_position.x, new_position.y, new_velocity.x, new_velocity.y, p.mass);
}

int main() {
    // create the elipse:

    elipse elipses[2] = {
        elipse(2.0, 1.0, 0.0, 0.0, 0, 2*M_PI), // A large elipse with a collidable section of 3 radians
        elipse(0.1, 0.2, 0.01, -0.01, M_PI/2, 3*M_PI/2) // A small elipse with a collidable section of π radians
    };
    // create the gravity force:
    vector2D gravity(0.0, -1.0);

    // initializa an array of 3 particles:
    particle particles[2] = {
        particle(1.0,  0.5,  0.2,  0.2),
        particle(1.5,  0.5,  0.5,  0.5)
    };

    double t0 = 0.0;
    double dt = 0.00001;
    double tf = 100.0;
    double t = t0;
    int step = 0;
    int strobe_interval = 1000;
    int n_particles = sizeof(particles) / sizeof(particles[0]);
    int n_elipses   = sizeof(elipses)   / sizeof(elipses[0]);

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

            // Snapshot inside/outside state for every ellipse before the step
            bool was_inside[n_elipses];
            for (int j = 0; j < n_elipses; j++)
                was_inside[j] = elipses[j].is_inside(old_x, old_y);

            particles[i] = eulerStep(particles[i], gravity, dt);

            // Check each ellipse for a boundary crossing
            for (int j = 0; j < n_elipses; j++) {
                bool now_inside = elipses[j].is_inside(particles[i].position.x,
                                                        particles[i].position.y);
                if (was_inside[j] != now_inside && elipses[j].is_on_collidable_section(old_x, old_y)) {
                    // Reflect off the outward normal at the pre-step position
                    vector2D n = elipses[j].normal(old_x, old_y);
                    particles[i].velocity = reflect(particles[i].velocity, n);
                    particles[i].position.x = old_x;
                    particles[i].position.y = old_y;
                    break; // one collision per step per particle
                }
            }
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