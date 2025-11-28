#!/bin/bash

# Enhanced build and run script for channel-last kernel with test mode selection
# Note: Not using 'set -e' to allow better error handling
set -o pipefail

# ==================== Configuration ====================

TEST_MODE=0  # Default: run all tests

# Parse command line arguments
if [ $# -gt 0 ]; then
    TEST_MODE=$1
    if [ "$TEST_MODE" != "0" ] && [ "$TEST_MODE" != "1" ] && [ "$TEST_MODE" != "2" ] && [ "$TEST_MODE" != "3" ]; then
        echo "✗ Invalid test mode: $TEST_MODE"
        echo ""
        echo "Usage: $0 [mode]"
        echo "  mode 0 or omit: Run all tests (default)"
        echo "  mode 1: Run basic functionality tests only"
        echo "  mode 2: Run seq_idx tests only"
        echo "  mode 3: Run final_states tests only"
        exit 1
    fi
fi

# ==================== Functions ====================

print_header() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Channel-Last Causal Conv1D Test Suite                        ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

print_section() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

print_test_mode() {
    case $TEST_MODE in
        0)
            echo "Test Mode: ALL (Basic + seq_idx + final_states)"
            ;;
        1)
            echo "Test Mode: Basic Functionality Only"
            ;;
        2)
            echo "Test Mode: seq_idx Tests Only"
            ;;
        3)
            echo "Test Mode: final_states Tests Only"
            ;;
    esac
}

# ==================== Start ====================

cd "$(dirname "$0")"

print_header
echo ""
print_test_mode
echo ""

# ==================== Check Prerequisites ====================

print_section "Checking Prerequisites"

# Check for hipcc
if ! command -v hipcc &> /dev/null; then
    echo "✗ Error: hipcc not found"
    echo "  Please install ROCm and ensure hipcc is in your PATH"
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit 1
fi
echo "✓ hipcc found"

# Check for test source file
if [ ! -f "test_causal_conv1d_channellast.cpp" ]; then
    echo "✗ Error: test_causal_conv1d_channellast.cpp not found"
    echo "  Please ensure you are in the correct directory"
    echo "  Current directory: $(pwd)"
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit 1
fi
echo "✓ Test source file found"

# Detect GPU architecture
GPU_ARCH=""
if command -v rocminfo &> /dev/null; then
    DETECTED=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}' || echo "")
    if [ -n "$DETECTED" ]; then
        GPU_ARCH="--offload-arch=$DETECTED"
        echo "✓ Detected GPU: $DETECTED"
    fi
else
    echo "⚠ rocminfo not found, using default architecture"
fi

# ==================== Compilation ====================

print_section "Compiling"

HIPCC_FLAGS="-O2 -std=c++17 $GPU_ARCH"
echo "Flags: $HIPCC_FLAGS"
echo ""

# Compile and capture both stdout and stderr
hipcc $HIPCC_FLAGS \
    test_causal_conv1d_channellast.cpp \
    -o test_causal_conv1d_channellast 2>&1 | tee compile_channellast.log

# Check compilation status
COMPILE_STATUS=$?
if [ $COMPILE_STATUS -ne 0 ]; then
    echo ""
    echo "✗ Compilation failed with exit code: $COMPILE_STATUS"
    echo "See compile_channellast.log for details"
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit 1
fi

# Also check if the output file was created
if [ ! -f conv1d_hip_channellast ]; then
    echo ""
    echo "✗ Compilation output file not created"
    echo "See compile_channellast.log for details"
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit 1
fi

echo ""
echo "✓ Compilation successful!"

# ==================== Run Tests ====================

print_section "Running Tests"
echo ""

# Run with timeout (5 minutes)
OUTPUT_FILE="/tmp/channellast_test_$$.log"

if command -v timeout &> /dev/null; then
    timeout 300 ./test_causal_conv1d_channellast $TEST_MODE 2>&1 | tee "$OUTPUT_FILE"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo ""
        echo "✗ Test timed out after 5 minutes"
        echo ""
        echo "Press Enter to exit..."
        read -r
        rm -f "$OUTPUT_FILE"
        exit 124
    fi
else
    ./conv1d_hip_channellast $TEST_MODE 2>&1 | tee "$OUTPUT_FILE"
    EXIT_CODE=$?
fi

# ==================== Test Results Analysis ====================

print_section "Test Results Summary"
echo ""

# Extract summary
if grep -q "Test Summary:" "$OUTPUT_FILE"; then
    grep -A 3 "Test Summary:" "$OUTPUT_FILE"
    echo ""
else
    echo "⚠ Could not extract test summary"
    echo ""
fi

# Detailed results by test mode
if [ "$TEST_MODE" = "0" ]; then
    # All tests - show all parts
    
    echo "─────────────────────────────────────────"
    echo "PART 1: Basic Functionality Tests"
    echo "─────────────────────────────────────────"
    if grep -q "PART 1:" "$OUTPUT_FILE"; then
        awk '/PART 1: Basic/,/PART 2:/' "$OUTPUT_FILE" | grep -E "Test: |Status:" | sed 's/^/  /'
    else
        echo "  (Not run)"
    fi
    echo ""
    
    echo "─────────────────────────────────────────"
    echo "PART 2: seq_idx Tests"
    echo "─────────────────────────────────────────"
    if grep -q "PART 2:" "$OUTPUT_FILE"; then
        awk '/PART 2: seq_idx/,/PART 3:/' "$OUTPUT_FILE" | grep -E "Test \(seq_idx\)|Status:" | sed 's/^/  /'
    else
        echo "  (Not run)"
    fi
    echo ""
    
    echo "─────────────────────────────────────────"
    echo "PART 3: final_states Tests"
    echo "─────────────────────────────────────────"
    if grep -q "PART 3:" "$OUTPUT_FILE"; then
        awk '/PART 3: States/,/Test Summary/' "$OUTPUT_FILE" | grep -E "Test \(final_states\)|Status:|Chunked processing with states" | sed 's/^/  /'
    else
        echo "  (Not run)"
    fi
    echo ""
    
elif [ "$TEST_MODE" = "1" ]; then
    # Basic tests only
    echo "─────────────────────────────────────────"
    echo "Basic Functionality Tests Details"
    echo "─────────────────────────────────────────"
    if grep -q "PART 1:" "$OUTPUT_FILE"; then
        grep -E "Test: |Status:|Max difference:" "$OUTPUT_FILE" | sed 's/^/  /'
    fi
    echo ""
    
elif [ "$TEST_MODE" = "2" ]; then
    # seq_idx tests only
    echo "─────────────────────────────────────────"
    echo "seq_idx Tests Details"
    echo "─────────────────────────────────────────"
    if grep -q "PART 2:" "$OUTPUT_FILE"; then
        grep -E "Test \(seq_idx\)|Status:|Max difference:" "$OUTPUT_FILE" | sed 's/^/  /'
    fi
    echo ""
    
    echo "Key Verification Points:"
    echo "  • Sequence boundaries respected (no cross-boundary convolution)"
    echo "  • Padding positions (seq_idx < 0) output zero"
    echo "  • GPU output matches CPU reference"
    echo ""
    
elif [ "$TEST_MODE" = "3" ]; then
    # final_states tests only
    echo "─────────────────────────────────────────"
    echo "final_states Tests Details"
    echo "─────────────────────────────────────────"
    if grep -q "PART 3:" "$OUTPUT_FILE"; then
        grep -E "Test \(final_states\)|Status:|Max difference:|Chunked processing" "$OUTPUT_FILE" | sed 's/^/  /'
    fi
    echo ""
    
    echo "Key Verification Points:"
    echo "  • Chunked output matches full sequence output"
    echo "  • final_states correctly save boundary information"
    echo "  • initial_states correctly restore context"
    echo ""
fi

# ==================== Precision Analysis ====================

if [ "$TEST_MODE" = "0" ] || [ "$TEST_MODE" = "2" ]; then
    if grep -q "seq_idx" "$OUTPUT_FILE"; then
        echo "─────────────────────────────────────────"
        echo "Precision: seq_idx Tests"
        echo "─────────────────────────────────────────"
        awk '/seq_idx/,/Status:/' "$OUTPUT_FILE" | grep "Max difference:" | sed 's/^/  /' | head -3
        echo ""
    fi
fi

if [ "$TEST_MODE" = "0" ] || [ "$TEST_MODE" = "3" ]; then
    if grep -q "final_states" "$OUTPUT_FILE"; then
        echo "─────────────────────────────────────────"
        echo "Precision: final_states Tests"
        echo "─────────────────────────────────────────"
        awk '/final_states/,/Status:/' "$OUTPUT_FILE" | grep "Max difference:" | sed 's/^/  /' | head -3
        echo ""
    fi
fi

# ==================== Final Status ====================

print_section "Final Status"
echo ""

# Check if all tests passed
TOTAL_TESTS=$(grep "Total:" "$OUTPUT_FILE" | tail -1 | awk '{print $2}')
PASSED_TESTS=$(grep "Passed:" "$OUTPUT_FILE" | tail -1 | awk '{print $2}')
FAILED_TESTS=$(grep "Failed:" "$OUTPUT_FILE" | tail -1 | awk '{print $2}')

if [ "$EXIT_CODE" -eq 0 ] && [ "$FAILED_TESTS" = "0" ]; then
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                     ✅ ALL TESTS PASSED! ✅                    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Total Tests: $TOTAL_TESTS"
    echo "  Passed: $PASSED_TESTS ✓"
    echo "  Failed: $FAILED_TESTS"
    echo ""
    
    case $TEST_MODE in
        1)
            echo "  ✓ Basic functionality validated"
            ;;
        2)
            echo "  ✓ seq_idx functionality validated"
            echo "  ✓ Sub-sequence handling working correctly"
            ;;
        3)
            echo "  ✓ final_states functionality validated"
            echo "  ✓ Streaming/chunked processing working correctly"
            ;;
        0)
            echo "  ✓ All functionalities validated"
            echo "    - Basic operations"
            echo "    - seq_idx (sub-sequence handling)"
            echo "    - final_states (streaming/chunked processing)"
            ;;
    esac
else
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                     ⚠️  TESTS FAILED! ⚠️                      ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Total Tests: $TOTAL_TESTS"
    echo "  Passed: $PASSED_TESTS"
    echo "  Failed: $FAILED_TESTS ✗"
    echo ""
    echo "  Please review the detailed output above."
fi

# ==================== Usage Information ====================

if [ "$TEST_MODE" = "0" ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Tip: You can run specific test categories:"
    echo "  $0 1  - Run basic functionality tests only"
    echo "  $0 2  - Run seq_idx tests only"
    echo "  $0 3  - Run final_states tests only"
    echo "────────────────────────────────────────────────────────────────"
fi

echo ""

# Optional: Pause before exit to prevent terminal from closing
if [ -n "$PAUSE_ON_EXIT" ]; then
    echo "Press Enter to exit..."
    read -r
fi

# Cleanup
rm -f "$OUTPUT_FILE"

exit $EXIT_CODE
