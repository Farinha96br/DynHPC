void printTest(const equationSet& obj, double x, double y) {
    bool inside = eval_F(obj.shape_id, obj.p, x, y) < 0.0;
    vector2D normal = calculateNormal(obj, x, y);
    printf("obj: %s x,y: %g,%g, nx, ny: %g, %g, inside: %s\n",
           obj.name, x, y, normal.x, normal.y, inside ? "true" : "false");
}


