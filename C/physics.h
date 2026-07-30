// file related to physics. Such as time integration methods.

// upper bound on collisions handled within one time step (fixed cap → GPU-safe);
// when exhausted the particle is rest-clamped on the legal side, so dropping the
// residual step time is safe
#define MAX_BOUNCES_PER_STEP 1

// The force field is supplied by the main script, not by this file: declare it
// here, and every .cpp that includes physics.h must define it (inside its
// "#pragma omp declare target" region so the GPU gets a copy). This keeps the
// integrator generic without device function pointers, which OpenMP offload
// does not support.
vector2D force_at_position(double x, double y);


vector2D reflect(const vector2D& v, const vector2D& n) {
    double dot = v.x*n.x + v.y*n.y;
    return vector2D(v.x - 2.0*dot*n.x, v.y - 2.0*dot*n.y);
}

vector2D calculateNormal(const equationSet& obj, double x, double y) {
    double dFdx = eval_dx(obj.shape_id, obj.p, x, y);
    double dFdy = eval_dy(obj.shape_id, obj.p, x, y);
    double len  = sqrt(dFdx*dFdx + dFdy*dFdy);
    // |grad F| = 0 at a shape's critical point (e.g. a degenerate zero-radius
    // circle). Returning a null normal makes the bounce a no-op instead of
    // letting a NaN propagate silently through the whole trajectory.
    if (!(len > 0.0)) return vector2D(0.0, 0.0);
    double sig  = eval_F(obj.shape_id, obj.p, x, y) < 0.0 ? -1.0 : 1.0;
    return vector2D(sig*dFdx/len, sig*dFdy/len);
}


state yoshida4Step(const state& p, double dt) {
    // Yoshida 4th-order coefficients, hard-coded (no pow(), no runtime algebra).
    // Derived from cbrt2 = 2^(1/3), w1 = 1/(2 - cbrt2), w0 = -cbrt2*w1 as
    //   c1 = w1/2,  c2 = (w0+w1)/2,  d1 = w1,  d2 = w0
    // and written at 17 significant digits, so each literal is the same double
    // the expressions produced.
    const double c1 =  0.67560359597982889;
    const double c2 = -0.17560359597982889;
    const double d1 =  1.3512071919596578;
    const double d2 = -1.7024143839193155;

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

// advance one time step, ignoring collisions (encapsulates integrator choice)
state singleStep(const state& p, double dt) {
    return yoshida4Step(p, dt);
}

// find the earliest surface crossing between p and s_new = singleStep(p, dt);
// obj_idx = -1 when nothing (unmasked) was hit.
// The returned s_hit is the NEAR-side state: the last integrated state that has
// NOT yet crossed the surface. Bouncing from there keeps the particle on the
// legal side for any restitution — it can never be marooned inside/behind a wall.
collisionEvent detectFirstCollision(const state& p, const state& s_new,
                                    const CollisionableObject* objs, int n_objs, double dt) {
    collisionEvent ev(-1, dt, s_new);

    for (int i = 0; i < n_objs; ++i) {
        double F_old = eval_F(objs[i].obj.shape_id, objs[i].obj.p, p.position.x,     p.position.y);
        double F_new = eval_F(objs[i].obj.shape_id, objs[i].obj.p, s_new.position.x, s_new.position.y);
        if (F_old * F_new >= 0.0) continue;

        double dt_small = 0.0, dt_large = dt;
        state  s_before = p;      // near side (not yet crossed)
        state  s_surf   = s_new;  // far side (crossed) — used only for the mask test
        if (fabs(F_old) <= 1e-8 * fabs(F_new - F_old)) {
            // contact at step start: the particle already touches the surface
            // (resting/sliding). Impact = the start state itself; skipping the
            // bisection keeps resting particles ~free instead of 32× a step.
            s_surf = p;
        } else {
            // bisect until the bracket is one ULP wide: the impact accuracy is set
            // by this tolerance, not by the arithmetic, and it scales with dt
            for (int iter = 0; iter < 60; ++iter) {
                double dt_mid = 0.5*(dt_small + dt_large);
                if (dt_mid <= dt_small || dt_mid >= dt_large) break;   // converged
                state  s_mid  = singleStep(p, dt_mid);
                double F_mid  = eval_F(objs[i].obj.shape_id, objs[i].obj.p, s_mid.position.x, s_mid.position.y);
                // F_mid == 0 counts as crossed, so s_before always keeps a strictly
                // legal-sign F — an exact-zero F would blind the next step's detection
                if (F_old * F_mid < 0.0 || F_mid == 0.0) { dt_large = dt_mid; s_surf   = s_mid; }
                else                                      { dt_small = dt_mid; s_before = s_mid; }
            }
        }

        bool masked = false;
        for (int m = 0; m < objs[i].n_masks && !masked; ++m)
            if (eval_mask(objs[i].masks[m].mask_id, objs[i].masks[m].p, s_surf.position.x, s_surf.position.y) < 0.0)
                masked = true;
        if (masked) continue;

        if (dt_small < ev.dt_hit) { ev.obj_idx = i; ev.dt_hit = dt_small; ev.s_hit = s_before; }
    }

    return ev;
}

// reflect the velocity at the surface (with restitution); returns the bounced
// state at the impact point — integrating the remaining time is the caller's job.
// Resting contact: if the outgoing normal speed is too small to outrun the local
// force for even ~2 steps of size dt, it is zeroed (tangential part kept) — the
// particle rests/slides on the surface instead of micro-bouncing forever (Zeno).
state resolveCollision(const collisionEvent& ev, const CollisionableObject& obj,
                       double mass, double dt) {
    vector2D normal = calculateNormal(obj.obj, ev.s_hit.position.x, ev.s_hit.position.y);

    // Restitution acts on the NORMAL component only; a frictionless wall leaves
    // the tangential component untouched. Scaling the whole reflected vector
    // would damp the tangential motion too, i.e. silently add friction.
    //   v_out = v_t - e*v_n*n  =  v - (1+e)*(v.n)*n
    // At e = 1 this is exactly reflect(), so elastic scenes are unchanged.
    double vn_in = ev.s_hit.velocity.x*normal.x + ev.s_hit.velocity.y*normal.y;  // < 0, incoming
    double vx = ev.s_hit.velocity.x - (1.0 + obj.restitution) * vn_in * normal.x;
    double vy = ev.s_hit.velocity.y - (1.0 + obj.restitution) * vn_in * normal.y;

    // Only the force component pushing INTO the surface can pull the particle
    // back onto it, so that -- not |f| -- sets the escape speed. On a surface
    // where the force is tangential (a vertical wall under gravity) nothing
    // pulls the particle back and the clamp must never fire.
    vector2D f     = force_at_position(ev.s_hit.position.x, ev.s_hit.position.y);
    double   f_in  = -(f.x*normal.x + f.y*normal.y);
    double   v_esc = 2.0 * (f_in > 0.0 ? f_in : 0.0) * dt;
    double   vn    = vx*normal.x + vy*normal.y;   // outgoing normal speed (normal points to the legal side)
    if (vn < v_esc) { vx -= vn*normal.x; vy -= vn*normal.y; }

    return state(ev.s_hit.position.x, ev.s_hit.position.y, vx, vy, mass);
}
