#!/usr/bin/env python3
"""Call MakeMap from C/libmakemap.so and get the collisions back as numpy arrays.

Build the library first (from C/):
    apptainer exec --nv ../containers/dynhpc.sif nvc++ -O2 -mp=gpu -gpu=ccnative \
        -fPIC -shared -Wl,-rpath,'$ORIGIN/../containers/nvhpc-redist' \
        -o libmakemap.so MapGPU.cpp -lm

Then just:  python3 run_map.py
"""

import ctypes
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parent.parent / "C" / "libmakemap.so"

lib = ctypes.CDLL(str(LIB))

# ndpointer lets us pass numpy arrays straight in, and checks dtype/contiguity
d = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
i = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

lib.MakeMap.restype = None
lib.MakeMap.argtypes = [
    ctypes.c_double,   # dt
    ctypes.c_double,   # t_end
    ctypes.c_int,      # N_P
    ctypes.c_int,      # N_REFLECTIONS
    d, d, d, d, d, d,  # times, IDs, x, y, vx, vy
    i,                 # n (collisions actually recorded per particle)
]


def make_map(dt, t_end, n_particles, n_reflections):
    """Run the simulation; return one row per recorded collision."""
    size = n_particles * n_reflections
    times = np.zeros(size)
    ids = np.zeros(size)
    x = np.zeros(size)
    y = np.zeros(size)
    vx = np.zeros(size)
    vy = np.zeros(size)
    n = np.zeros(n_particles, dtype=np.int32)

    lib.MakeMap(dt, t_end, n_particles, n_reflections,
                times, ids, x, y, vx, vy, n)

    # n[i] is how many of particle i's n_reflections slots are real;
    # drop the untouched tail of every slice.
    shape = (n_particles, n_reflections)
    keep = np.arange(n_reflections)[None, :] < n[:, None]
    return dict(t=times.reshape(shape)[keep],
                particle=ids.reshape(shape)[keep].astype(int),
                x=x.reshape(shape)[keep], y=y.reshape(shape)[keep],
                vx=vx.reshape(shape)[keep], vy=vy.reshape(shape)[keep],
                n=n)


if __name__ == "__main__":
    res = make_map(dt=1e-3, t_end=100.0, n_particles=1000, n_reflections=1000)
    print(res["particle"])


    