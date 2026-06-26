

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

struct mask {
    /*
    Define a mask, where:
    mask_id: registered mask index (see eval_mask), -1 = inactive/undefined.
    p: parameters of the mask (up to MAX_PARAMS)
    */
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