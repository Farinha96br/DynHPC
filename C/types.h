

struct vector2D {
    /*
    simple 2d vector to return two values
    */
    double x, y;
    vector2D(double x, double y) : x(x), y(y) {}
};

struct state {
    /*
    define the state of the particle: position, velocity, mass.
    */
    vector2D position;
    vector2D velocity;
    double   mass = 1.0;
    state(double x, double y, double vx, double vy, double m = 1.0)
        : position(x, y), velocity(vx, vy), mass(m) {}
};

struct equationSet {
    /*
    Define a set of equations, where:
    name: name of the shape
    p: parameters of the shape (up to MAX_PARAMS)
    n: number of parameters used
    shape_id: registered shape index (see eval_F/eval_dx/eval_dy), -1 = inactive/undefined.  

    */
    char   name[64]      = "dummyName";
    int    n             = 0;
    double p[MAX_PARAMS] = {};
    int    shape_id      = -1;
};

struct collisionEvent {
    /*
    Result of collision detection over one time step, where:
    obj_idx: index into the objects array; -1 = no collision
    dt_hit: time-of-impact within the step
    s_hit: integrated state at the surface
    */
    int    obj_idx;
    double dt_hit;
    state  s_hit;
    collisionEvent(int idx, double dth, const state& s)
        : obj_idx(idx), dt_hit(dth), s_hit(s) {}
};

class CollisionableObject {
public:
    equationSet obj;
    double      restitution = 1.0;
};