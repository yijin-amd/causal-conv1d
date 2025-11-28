#!/bin/bash

# Build and run script for causal_conv1d_kernel_update.hip
# Target: AMD MI308 GPU (gfx942)

set -o pipefail

# HIP compiler
HIPCC=${HIPCC:-hipcc}

# Target architecture for MI308
# gfx942: AMD Instinct MI300 series (CDNA3)
ARCH="gfx942"

# Compiler flags
FLAGS=(
    # Architecture
    --offload-arch=${ARCH}
    
    # Optimization
    -O3
    -ffast-math
    
    # C++ standard
    -std=c++17
    
    # Warnings
    -Wall
    -Wextra
    
    # Defines
    -D__HIP_PLATFORM_AMD__
    
    # MI308 specific optimizations
    -munsafe-fp-atomics        # Fast atomic operations
    -mwavefrontsize64          # Explicit wavefront size (CDNA3 default)
    
    # Aggressive optimizations
    -fno-strict-aliasing
    -fomit-frame-pointer
)

echo "=========================================="
echo "Compilation Options:"
echo "  1) Compile object file (.o)"
echo "  2) Compile test program with kernel"
echo "=========================================="
echo ""

# Check if --with-test flag is provided
if [ "$1" == "--with-test" ]; then
    COMPILE_MODE="test"
elif [ "$1" == "--object" ]; then
    COMPILE_MODE="object"
else
    # Default: compile test program
    COMPILE_MODE="test"
fi

if [ "$COMPILE_MODE" == "object" ]; then
    echo "=========================================="
    echo "Compiling kernel to object file..."
    echo "Target: AMD MI308 (gfx942)"
    echo "=========================================="
    
    SOURCE="causal_conv1d_kernel_update.hip"
    OUTPUT="causal_conv1d_kernel_update.o"
    
    echo "Running: ${HIPCC} ${FLAGS[@]} -c ${SOURCE} -o ${OUTPUT}"
    ${HIPCC} "${FLAGS[@]}" -c ${SOURCE} -o ${OUTPUT}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Object file compiled successfully!"
        echo "   Output: ${OUTPUT}"
        ls -lh ${OUTPUT}
    else
        echo ""
        echo "❌ Compilation failed!"
        exit 1
    fi

else
    echo "=========================================="
    echo "Compiling test program with kernel..."
    echo "Target: AMD MI308 (gfx942)"
    echo "=========================================="
    
    TEST_SOURCE="test_causal_conv1d_update.cpp"
    OUTPUT="test_causal_conv1d_update"
    
    if [ ! -f ${TEST_SOURCE} ]; then
        echo "❌ Error: ${TEST_SOURCE} not found!"
        echo "   Please ensure test_causal_conv1d_update.cpp exists in the current directory."
        exit 1
    fi
    
    echo "Running: ${HIPCC} ${FLAGS[@]} ${TEST_SOURCE} -o ${OUTPUT}"
    ${HIPCC} "${FLAGS[@]}" ${TEST_SOURCE} -o ${OUTPUT}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Test program compiled successfully!"
        echo "   Output: ${OUTPUT}"
        echo ""
        echo "Binary info:"
        ls -lh ${OUTPUT}
        echo ""
        echo "To run the test:"
        echo "  ./${OUTPUT}"
        echo ""
        echo "To check GPU:"
        echo "  rocm-smi"
    else
        echo ""
        echo "❌ Compilation failed!"
        exit 1
    fi
fi

echo "=========================================="
echo "Compilation complete"
echo "=========================================="

# If test program was compiled, offer to run it
if [ "$COMPILE_MODE" == "test" ] && [ -f "${OUTPUT}" ]; then
    echo ""
    echo "=========================================="
    echo "Running Test Program"
    echo "=========================================="
    echo ""
    
    if command -v timeout &> /dev/null; then
        timeout 300 ./${OUTPUT}
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 124 ]; then
            echo ""
            echo "✗ Test timed out after 5 minutes"
            exit 124
        fi
    else
        ./${OUTPUT}
        EXIT_CODE=$?
    fi
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Tests completed successfully!"
    else
        echo "❌ Tests failed with exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi
fi
