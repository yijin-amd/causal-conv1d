#!/bin/bash

# Enhanced build and run script for channel-last kernel with test mode selection
set -e

# ==================== Configuration ====================

TEST_MODE=0  # Default: run all tests
SAVE_OUTPUT=0  # Default: don't save to file
OUTPUT_FILE="channellast_test_results.txt"

# Parse command line arguments
if [ $# -gt 0 ]; then
    TEST_MODE=$1
    if [ "$TEST_MODE" != "0" ] && [ "$TEST_MODE" != "1" ] && [ "$TEST_MODE" != "2" ] && [ "$TEST_MODE" != "3" ]; then
        echo "✗ Invalid test mode: $TEST_MODE"
        echo ""
        echo "Usage: $0 [mode] [--save-output]"
        echo "  mode 0 or omit: Run all tests (default)"
        echo "  mode 1: Run basic functionality tests only"
        echo "  mode 2: Run seq_idx tests only"
        echo "  mode 3: Run final_states tests only"
        echo ""
        echo "Options:"
        echo "  --save-output: Save test results to $OUTPUT_FILE"
        exit 1
    fi
fi

# Check for --save-output flag
if [ $# -gt 1 ] && [ "$2" == "--save-output" ]; then
    SAVE_OUTPUT=1
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

if ! command -v hipcc &> /dev/null; then
    echo "✗ Error: hipcc not found"
    exit 1
fi
echo "✓ hipcc found"

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

hipcc $HIPCC_FLAGS \
    test_rocm_channellast_fwd_conv1d_hip.cpp \
    -o test_rocm_channellast_fwd_conv1d_hip 2>&1 | tee compile_channellast.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "✗ Compilation failed"
    echo "See compile_channellast.log for details"
    exit 1
fi

echo ""
echo "✓ Compilation successful!"

# ==================== Run Tests ====================

print_section "Running Tests"
echo ""

# Run with timeout (5 minutes)
TEMP_OUTPUT_FILE="/tmp/channellast_test_$$.log"

if command -v timeout &> /dev/null; then
    timeout 300 ./test_rocm_channellast_fwd_conv1d_hip $TEST_MODE 2>&1 | tee "$TEMP_OUTPUT_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo ""
        echo "✗ Test timed out after 5 minutes"
        rm -f "$TEMP_OUTPUT_FILE"
        exit 124
    fi
else
    ./test_rocm_channellast_fwd_conv1d_hip $TEST_MODE 2>&1 | tee "$TEMP_OUTPUT_FILE"
    EXIT_CODE=$?
fi

# ==================== Test Results Analysis ====================

print_section "Test Results Summary"
echo ""

# Extract summary
if grep -q "Test Summary:" "$TEMP_OUTPUT_FILE"; then
    grep -A 3 "Test Summary:" "$TEMP_OUTPUT_FILE"
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
    if grep -q "PART 1:" "$TEMP_OUTPUT_FILE"; then
        awk '/PART 1: Basic/,/PART 2:/' "$TEMP_OUTPUT_FILE" | grep -E "Test: |Status:" | sed 's/^/  /'
    else
        echo "  (Not run)"
    fi
    echo ""
    
    echo "─────────────────────────────────────────"
    echo "PART 2: seq_idx Tests"
    echo "─────────────────────────────────────────"
    if grep -q "PART 2:" "$TEMP_OUTPUT_FILE"; then
        awk '/PART 2: seq_idx/,/PART 3:/' "$TEMP_OUTPUT_FILE" | grep -E "Test \(seq_idx\)|Status:" | sed 's/^/  /'
    else
        echo "  (Not run)"
    fi
    echo ""
    
    echo "─────────────────────────────────────────"
    echo "PART 3: final_states Tests"
    echo "─────────────────────────────────────────"
    if grep -q "PART 3:" "$TEMP_OUTPUT_FILE"; then
        awk '/PART 3: States/,/Test Summary/' "$TEMP_OUTPUT_FILE" | grep -E "Test \(final_states\)|Status:|Chunked processing with states" | sed 's/^/  /'
    else
        echo "  (Not run)"
    fi
    echo ""
    
elif [ "$TEST_MODE" = "1" ]; then
    # Basic tests only
    echo "─────────────────────────────────────────"
    echo "Basic Functionality Tests Details"
    echo "─────────────────────────────────────────"
    if grep -q "PART 1:" "$TEMP_OUTPUT_FILE"; then
        grep -E "Test: |Status:|Max difference:" "$TEMP_OUTPUT_FILE" | sed 's/^/  /'
    fi
    echo ""
    
elif [ "$TEST_MODE" = "2" ]; then
    # seq_idx tests only
    echo "─────────────────────────────────────────"
    echo "seq_idx Tests Details"
    echo "─────────────────────────────────────────"
    if grep -q "PART 2:" "$TEMP_OUTPUT_FILE"; then
        grep -E "Test \(seq_idx\)|Status:|Max difference:" "$TEMP_OUTPUT_FILE" | sed 's/^/  /'
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
    if grep -q "PART 3:" "$TEMP_OUTPUT_FILE"; then
        grep -E "Test \(final_states\)|Status:|Max difference:|Chunked processing" "$TEMP_OUTPUT_FILE" | sed 's/^/  /'
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
    if grep -q "seq_idx" "$TEMP_OUTPUT_FILE"; then
        echo "─────────────────────────────────────────"
        echo "Precision: seq_idx Tests"
        echo "─────────────────────────────────────────"
        awk '/seq_idx/,/Status:/' "$TEMP_OUTPUT_FILE" | grep "Max difference:" | sed 's/^/  /' | head -3
        echo ""
    fi
fi

if [ "$TEST_MODE" = "0" ] || [ "$TEST_MODE" = "3" ]; then
    if grep -q "final_states" "$TEMP_OUTPUT_FILE"; then
        echo "─────────────────────────────────────────"
        echo "Precision: final_states Tests"
        echo "─────────────────────────────────────────"
        awk '/final_states/,/Status:/' "$TEMP_OUTPUT_FILE" | grep "Max difference:" | sed 's/^/  /' | head -3
        echo ""
    fi
fi

# ==================== Final Status ====================

print_section "Final Status"
echo ""

# Check if all tests passed
TOTAL_TESTS=$(grep "Total:" "$TEMP_OUTPUT_FILE" | tail -1 | awk '{print $2}')
PASSED_TESTS=$(grep "Passed:" "$TEMP_OUTPUT_FILE" | tail -1 | awk '{print $2}')
FAILED_TESTS=$(grep "Failed:" "$TEMP_OUTPUT_FILE" | tail -1 | awk '{print $2}')

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
    echo ""
    echo "Save test results to file:"
    echo "  $0 1 --save-output  - Run tests and save to $OUTPUT_FILE"
    echo "────────────────────────────────────────────────────────────────"
fi

echo ""

# ==================== Save Output if Requested ====================

if [ "$SAVE_OUTPUT" = "1" ]; then
    echo "────────────────────────────────────────────────────────────────"
    echo "Saving test results to: $OUTPUT_FILE"
    echo "────────────────────────────────────────────────────────────────"
    
    # Copy the full output to the specified file
    cp "$TEMP_OUTPUT_FILE" "$OUTPUT_FILE"
    
    echo "✓ Test results saved successfully!"
    echo ""
    echo "You can view the results with:"
    echo "  cat $OUTPUT_FILE"
    echo "  less $OUTPUT_FILE"
    echo ""
    
    # Also extract performance data to a separate summary file
    PERF_SUMMARY_FILE="channellast_perf_summary.txt"
    
    echo "Extracting performance summary to: $PERF_SUMMARY_FILE"
    
    {
        echo "================================================================================================"
        echo "Channel-Last Kernel Performance Summary"
        echo "Extracted from: $OUTPUT_FILE"
        echo "Test Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================================================"
        echo ""
        echo "Performance Data (Average time and Bandwidth):"
        echo "================================================================================================"
        echo ""
        
        # Extract all test names and their performance data
        grep -E "Test: |Average time:|Bandwidth:" "$TEMP_OUTPUT_FILE" | \
        awk '
        /Test: / { test=$0; getline; getline }
        /Average time:/ { time=$3; getline }
        /Bandwidth:/ { bw=$2; if (test != "") print test " | Time: " time " | BW: " bw; test="" }
        '
        
        echo ""
        echo "================================================================================================"
        echo "Test Summary:"
        echo "================================================================================================"
        grep -A 3 "Test Summary:" "$TEMP_OUTPUT_FILE" | tail -4
        echo ""
        echo "================================================================================================"
        
    } > "$PERF_SUMMARY_FILE"
    
    echo "✓ Performance summary saved to: $PERF_SUMMARY_FILE"
    echo ""
fi

# Cleanup
rm -f "$TEMP_OUTPUT_FILE"

exit $EXIT_CODE
