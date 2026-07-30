# Compile, run and plot the cup test: 1e6 particles dropped into a box-less
# arena containing two upward-open "cups" (circles that collide only on their
# lower halves). Each particle is colored by where it ended up.
#
# Writes: cup_test.csv (one row per particle: x0,y0,xf,yf)
#         cup_outcomes.png (initial conditions colored by final outcome)
#
# The scene needs the shared headers one directory up, hence -I..
# With N_P = 1e6 the run takes a few minutes on the GPU.
set -e

echo "Compiling"
nvc++ -O3 -mp=gpu -gpu=ccnative -I.. -o cup_test.out cup_test.cpp -lm

echo "Running"
time ./cup_test.out            # writes cup_test.csv

echo "Plotting"
python3 plot_cup_outcomes.py   # writes cup_outcomes.png
