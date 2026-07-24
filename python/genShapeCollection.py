#!/usr/bin/env python3
"""

"""

import sys
import re
from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import PRECEDENCE


def buildCfunctionLine(Cexpression, functionName):
    return "double " + functionName + " (const double* p, double x, double y){\n return " + Cexpression + ";}"
    
    
def buildCfunctions(sympyImplicitEq,coordinates,parameters,functionName):
    # target:
    # double circle_F (const double* p, double x, double y){ double dx=x-p[0], dy=y-p[1]; return dx*dx+dy*dy-p[2]*p[2]; }
    # double circle_dx(const double* p, double x, double y){ return 2.0*(x-p[0]); }
    # double circle_dy(const double* p, double x, double y){ return 2.0*(y-p[1]); }
    output = ""
    if not isinstance(functionName, str):
        raise TypeError("functionName must be a string")
    if not functionName:
        raise ValueError("functionName must not be empty")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", functionName):
        raise ValueError("functionName must be a valid C identifier for a variable")

    # compute the number of parameters
    numParameters = len(parameters)
    # make an indexed base for the parameters
    p = sp.IndexedBase("p", shape=(numParameters,))


    for i in range(numParameters):
        # specify the index order of the parameters in the sympyImplicitEq
        output += f"mapping {p[i]} to {parameters[i]}\n"
        sympyImplicitEq = sympyImplicitEq.subs(parameters[i], p[i])

    # first make dx, dy
    dx = sp.diff(sympyImplicitEq, coordinates[0])
    dy = sp.diff(sympyImplicitEq, coordinates[1])

    # then make the C code for F, dx, dy
    F_code = sp.ccode(sympyImplicitEq)
    dx_code = sp.ccode(dx)
    dy_code = sp.ccode(dy)
    output += buildCfunctionLine(F_code, f"{functionName}_F")
    output += "\n"
    output += buildCfunctionLine(dx_code, f"{functionName}_dx")
    output += "\n"
    output += buildCfunctionLine(dy_code, f"{functionName}_dy")
    return output


# test with an off center circle: (x-p[0])^2 + (y-p[1])^2 - p[2]^2 = 0


x, y = sp.symbols("x y", real=True)
x0, y0, r = sp.symbols("x0 y0 r", real=True)
circle_eq = (x - x0)**2 + (y - y0)**2 - r**2

# build the C function for the circle
print(buildCfunctions(circle_eq, [x,y], [x0, y0, r], "circle"))

# build the C functions for the line

