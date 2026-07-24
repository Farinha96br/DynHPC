// this file defines the table, for the collision world.

// shape IDs: 0 = circle, 1 = line, 2 = sine
double circle_F (const double* p, double x, double y){ return (x-p[0])*(x-p[0])+(y-p[1])*(y-p[1])-p[2]*p[2]; }
double circle_dx(const double* p, double x, double y){ return 2.0*(x-p[0]); }
double circle_dy(const double* p, double x, double y){ return 2.0*(y-p[1]); }

double line_F (const double* p, double x, double y){ return y - p[0]*x + p[1]; }
double line_dx(const double* p, double x, double y){ return -p[0]; }
double line_dy(const double* p, double x, double y){ return 1.0; }

double sine_F (const double* p, double x, double y){ return y - p[0]*sin(p[1]*x + p[2]); }
double sine_dx(const double* p, double x, double y){ return -p[0]*p[1]*cos(p[1]*x + p[2]); }
double sine_dy(const double* p, double x, double y){ return 1.0; }


// mask IDs: 0 = mask_right_F (suppresses right side),  1 = mask_left_F  (suppresses left side)
//           2 = mask_upper_F (suppresses upper side),  3 = mask_lower_F (suppresses lower side)
double mask_right_F(const double* p, double x, double y){ return p[0] - x; }
double mask_left_F (const double* p, double x, double y){ return x - p[0]; }
double mask_upper_F(const double* p, double x, double y){ return p[0] - y; }
double mask_lower_F(const double* p, double x, double y){ return y - p[0]; }

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
        case 2: return mask_upper_F(p, x, y);
        case 3: return mask_lower_F(p, x, y);
        default: return 1.0;   // positive → no suppression
    }
}
