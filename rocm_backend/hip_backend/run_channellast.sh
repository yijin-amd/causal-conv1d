#!/bin/bash

# Build and run script for channel-last kernel
set -e

echo "========================================="
echo "Building Channel-Last Causal Conv1D"
echo "========================================="

cd "$(dirname "$0")"

# Check hipcc
if ! command -v hipcc &> /dev/null; then
    echo "✗ Error: hipcc not found"
    exit 1
fi

# Detect GPU architecture (optional)
GPU_ARCH=""
if command -v rocminfo &> /dev/null; then
    DETECTED=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}' || echo "")
    if [ -n "$DETECTED" ]; then
        GPU_ARCH="--offload-arch=$DETECTED"
        echo "✓ Detected GPU: $DETECTED"
    fi
fi

# Compilation flags
HIPCC_FLAGS="-O2 -std=c++17 $GPU_ARCH"

echo "Compiling conv1d_hip_channellast.cpp..."
echo "Flags: $HIPCC_FLAGS"
echo ""

# Compile
hipcc $HIPCC_FLAGS \
    conv1d_hip_channellast.cpp \
    -o conv1d_hip_channellast 2>&1 | tee compile_channellast.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful!"
    echo ""
    echo "========================================="
    echo "Running Test"
    echo "========================================="
    echo ""
    
    # Run with timeout (3 minutes)
    if command -v timeout &> /dev/null; then
        timeout 180 ./conv1d_hip_channellast
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 124 ]; then
            echo ""
            echo "✗ Test timed out after 3 minutes"
            exit 124
        fi
    else
        ./conv1d_hip_channellast
        EXIT_CODE=$?
    fi
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ All tests passed!"
    else
        echo "✗ Tests failed (exit code: $EXIT_CODE)"
    fi
    
    exit $EXIT_CODE
else
    echo ""
    echo "✗ Compilation failed"
    echo "See compile_channellast.log for details"
    exit 1
fi

