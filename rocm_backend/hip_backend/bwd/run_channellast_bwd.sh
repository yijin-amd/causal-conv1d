#!/bin/bash

# Test script for channel-last backward kernel
set -e

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Channel-Last Causal Conv1D Backward Kernel Test              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
TEST_MODE=0

if [ $# -gt 0 ]; then
    case "$1" in
        --accuracy|-a)
            TEST_MODE=1
            ;;
        --performance|-p)
            TEST_MODE=2
            ;;
        --all)
            TEST_MODE=0
            ;;
        --help|-h)
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  --all          Run both accuracy and performance tests (default)"
            echo "  --accuracy, -a Run accuracy tests only"
            echo "  --performance, -p Run performance tests only"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
fi

# Check prerequisites
echo "[1/3] Checking prerequisites..."

if ! command -v hipcc &> /dev/null; then
    echo "✗ Error: hipcc not found"
    exit 1
fi
echo "✓ hipcc found"

# Detect GPU
GPU_ARCH=""
if command -v rocminfo &> /dev/null; then
    DETECTED=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}' || echo "")
    if [ -n "$DETECTED" ]; then
        GPU_ARCH="--offload-arch=$DETECTED"
        echo "✓ Detected GPU: $DETECTED"
    fi
fi

echo ""

# Compile
echo "[2/3] Compiling..."

HIPCC_FLAGS="-O2 -std=c++17 $GPU_ARCH"
echo "Flags: $HIPCC_FLAGS"
echo ""

hipcc $HIPCC_FLAGS causal_conv1d_channellast_bwd_hip.cpp -o test_channellast_bwd_kernel 2>&1 | tee compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "✗ Compilation failed"
    echo "See compile.log for details"
    exit 1
fi

echo ""
echo "✓ Compilation successful!"
echo ""

# Run
echo "[3/3] Running test..."
echo ""

./test_channellast_bwd_kernel $TEST_MODE

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                     ✅ TESTS PASSED! ✅                        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
else
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                     ⚠️  TESTS FAILED! ⚠️                      ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
fi

echo ""
exit $EXIT_CODE

