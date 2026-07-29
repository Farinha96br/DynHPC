# compile the source code
echo "Compiling SimpleGravCircle.cpp"
nvc++ -O2 -mp=gpu -gpu=ccnative -I../../C -o GPUbuild.out SimpleGravCircle.cpp -lm
# erase old file if it exists

rm -f billiard_map.dat
# run the program
echo "Running"
./GPUbuild.out 
# plot it

echo "Plotting"
python3 plot_billiard_map.py 
