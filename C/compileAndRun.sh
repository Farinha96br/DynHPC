#!/usr/bin/env bash
set -euo pipefail


SCRIPT=$1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/$SCRIPT"

echo "=== Compiling CPU build ==="
g++ -O2 -fopenmp -foffload=disable -o "$DIR/CPUbuild" "$SRC" -lm
echo "CPU build OK"

echo ""
echo "=== Running CPU ==="
"$DIR/CPUbuild"

echo ""
echo "=== Compiling GPU build ==="
nvc++ -O2 -mp=gpu -gpu=ccnative -Minfo=mp -DGPU_OFFLOAD -o "$DIR/GPUbuild" "$SRC" -lm
echo "GPU build OK"

echo ""
echo "=== Running GPU ==="
"$DIR/GPUbuild"
