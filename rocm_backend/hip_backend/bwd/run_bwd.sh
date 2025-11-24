#!/bin/bash
# run_bwd.sh

echo "Compiling Causal Conv1D Backward kernel..."
hipcc -O3 -std=c++17 \
    causal_conv1d_bwd_hip.cpp \
    -o conv1d_bwd_test

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# echo -e "\n=== Running Performance Tests ==="
# ./conv1d_bwd_test

echo -e "\n=== Running Verification Tests ==="
./conv1d_bwd_test --verify