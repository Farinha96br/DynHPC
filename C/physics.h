// file related to physics. Such as time integration methods.

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
