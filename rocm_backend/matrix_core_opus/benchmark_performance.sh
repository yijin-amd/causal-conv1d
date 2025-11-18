#!/bin/bash
###############################################################################
# 性能基准测试脚本
# 功能：运行 casual_conv1d_opus.exe 多次，收集性能数据并计算统计信息
# 作者：AI Assistant
# 日期：2025-11-15
###############################################################################

set -e  # 遇到错误立即退出

# ========== 配置参数 ==========
ITERATIONS=${1:-100}           # 迭代次数（默认100次）
EXECUTABLE="./casual_conv1d_opus.exe"
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.txt"
CSV_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.csv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========== 函数定义 ==========

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Causal Conv1D 性能基准测试                            ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo ""
    echo -e "${GREEN}▶ $1${NC}"
    echo "================================================================"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 检查可执行文件
check_executable() {
    if [ ! -f "$EXECUTABLE" ]; then
        print_error "找不到可执行文件: $EXECUTABLE"
        echo "请先编译："
        echo "  hipcc -x hip -std=c++17 casual_conv1d_opus.cpp -o casual_conv1d_opus.exe ..."
        exit 1
    fi
    print_success "找到可执行文件: $EXECUTABLE"
}

# 创建输出目录
setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    print_success "输出目录: $OUTPUT_DIR"
}

# 检测当前配置
detect_config() {
    print_section "检测当前配置"
    
    # 提取宏定义
    SILU_ENABLED=$(grep "ENABLE_SILU_ACTIVATION" casual_conv1d_opus.cpp | head -1 | grep -o "[01]" || echo "未知")
    VERIFY_ENABLED=$(grep "ENABLE_HOST_VERIFICATION" casual_conv1d_opus.cpp | head -1 | grep -o "[01]" || echo "未知")
    
    echo "ENABLE_SILU_ACTIVATION:    $SILU_ENABLED ($([ "$SILU_ENABLED" == "1" ] && echo "启用" || echo "禁用"))"
    echo "ENABLE_HOST_VERIFICATION:  $VERIFY_ENABLED ($([ "$VERIFY_ENABLED" == "1" ] && echo "启用" || echo "禁用"))"
    
    # 运行一次获取配置信息
    CONFIG_INFO=$($EXECUTABLE 2>&1 | head -20)
    BATCH=$(echo "$CONFIG_INFO" | grep -oP "batch:\K\d+" | head -1 || echo "1")
    
    echo "Batch Size:                $BATCH"
    echo ""
}

# 单次运行测试
run_single_test() {
    local iteration=$1
    local temp_output=$(mktemp)
    
    # 运行程序并捕获输出（忽略退出错误）
    $EXECUTABLE 2>&1 > "$temp_output" || true
    
    # 提取关键指标（可根据实际输出调整）
    local result=$(cat "$temp_output")
    
    # 尝试提取时间信息（需要根据实际输出格式调整）
    # 这里我们简单地标记是否成功
    if echo "$result" | grep -q "valid"; then
        echo "SUCCESS"
    else
        echo "FAILED"
    fi
    
    rm -f "$temp_output"
}

# 使用 rocprofv3 进行详细性能测试
run_with_rocprof() {
    local iteration=$1
    local prof_dir="${OUTPUT_DIR}/prof_${iteration}"
    
    mkdir -p "$prof_dir"
    
    # 使用 rocprofv3 收集性能数据
    rocprofv3 --stats -o "${prof_dir}/perf" $EXECUTABLE > /dev/null 2>&1 || true
    
    # 提取 kernel 执行时间（从 kernel_stats.csv）
    if [ -f "${prof_dir}/perf_kernel_stats.csv" ]; then
        # 提取总 kernel 时间（单位：微秒）
        local kernel_time=$(awk -F',' 'NR>1 {sum+=$5} END {print sum}' "${prof_dir}/perf_kernel_stats.csv" 2>/dev/null || echo "0")
        echo "$kernel_time"
    else
        echo "0"
    fi
}

# 快速性能测试（不使用 rocprof）
run_fast_benchmark() {
    print_section "运行快速性能测试 (${ITERATIONS} 次迭代)"
    print_info "使用程序内部计时..."
    
    local success_count=0
    local start_time=$(date +%s%N)
    
    echo "进度: "
    for i in $(seq 1 $ITERATIONS); do
        # 显示进度
        if [ $((i % 10)) -eq 0 ]; then
            printf "\r进度: [%-50s] %d/%d" $(printf '#%.0s' $(seq 1 $((i*50/ITERATIONS)))) $i $ITERATIONS
        fi
        
        # 运行测试
        result=$(run_single_test $i)
        if [ "$result" == "SUCCESS" ]; then
            ((success_count++))
        fi
    done
    
    local end_time=$(date +%s%N)
    local total_time=$(( (end_time - start_time) / 1000000 ))  # 转换为毫秒
    
    echo ""
    print_success "完成 ${ITERATIONS} 次迭代"
    echo ""
    echo "总耗时:       ${total_time} ms"
    echo "平均耗时:     $((total_time / ITERATIONS)) ms/次"
    echo "成功次数:     ${success_count}/${ITERATIONS}"
    echo "成功率:       $(awk "BEGIN {printf \"%.2f\", ${success_count}*100/${ITERATIONS}}")%"
}

# 详细性能测试（使用 rocprof）
run_detailed_benchmark() {
    print_section "运行详细性能测试 (${ITERATIONS} 次迭代)"
    print_info "使用 rocprofv3 收集详细性能数据..."
    print_info "这可能需要较长时间..."
    
    local -a kernel_times=()
    local start_time=$(date +%s)
    
    echo "进度: "
    for i in $(seq 1 $ITERATIONS); do
        # 显示进度
        if [ $((i % 5)) -eq 0 ] || [ $i -eq 1 ]; then
            printf "\r进度: [%-50s] %d/%d" $(printf '#%.0s' $(seq 1 $((i*50/ITERATIONS)))) $i $ITERATIONS
        fi
        
        # 使用 rocprof 运行
        kernel_time=$(run_with_rocprof $i)
        kernel_times+=($kernel_time)
    done
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    echo ""
    print_success "完成 ${ITERATIONS} 次迭代"
    
    # 计算统计信息
    calculate_statistics "${kernel_times[@]}"
    
    echo ""
    echo "总耗时: ${total_time} 秒"
}

# 计算统计信息
calculate_statistics() {
    local values=("$@")
    local count=${#values[@]}
    
    if [ $count -eq 0 ]; then
        print_error "没有有效数据"
        return
    fi
    
    # 使用 Python 计算统计信息（如果可用）
    if command -v python &> /dev/null; then
        python - <<EOF
import sys
values = [float(x) for x in """${values[*]}""".split()]
values = [v for v in values if v > 0]  # 过滤掉无效值

if len(values) > 0:
    avg = sum(values) / len(values)
    sorted_values = sorted(values)
    min_val = sorted_values[0]
    max_val = sorted_values[-1]
    median = sorted_values[len(values)//2]
    
    # 计算标准差
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5
    
    print("")
    print("=== 性能统计 (Kernel 时间) ===")
    print(f"样本数量:     {len(values)}")
    print(f"平均值:       {avg:.2f} μs")
    print(f"标准差:       {std_dev:.2f} μs")
    print(f"最小值:       {min_val:.2f} μs")
    print(f"最大值:       {max_val:.2f} μs")
    print(f"中位数:       {median:.2f} μs")
    print(f"变异系数:     {(std_dev/avg*100):.2f}%")
else:
    print("没有有效的性能数据")
EOF
    else
        # 如果没有 Python，使用简单的 awk 计算
        echo "${values[@]}" | tr ' ' '\n' | awk '
            $1 > 0 {
                sum += $1
                count++
                if (min == "" || $1 < min) min = $1
                if (max == "" || $1 > max) max = $1
            }
            END {
                if (count > 0) {
                    print ""
                    print "=== 性能统计 (Kernel 时间) ==="
                    printf "样本数量:     %d\n", count
                    printf "平均值:       %.2f μs\n", sum/count
                    printf "最小值:       %.2f μs\n", min
                    printf "最大值:       %.2f μs\n", max
                }
            }'
    fi
}

# 生成报告
generate_report() {
    print_section "生成报告"
    
    {
        echo "================================================================"
        echo "  Causal Conv1D 性能基准测试报告"
        echo "================================================================"
        echo ""
        echo "测试时间: $(date)"
        echo "迭代次数: $ITERATIONS"
        echo ""
        echo "配置信息:"
        echo "  ENABLE_SILU_ACTIVATION:    $SILU_ENABLED"
        echo "  ENABLE_HOST_VERIFICATION:  $VERIFY_ENABLED"
        echo "  Batch Size:                $BATCH"
        echo ""
        echo "可执行文件: $EXECUTABLE"
        echo "输出目录:   $OUTPUT_DIR"
        echo ""
        echo "================================================================"
    } | tee "$RESULT_FILE"
    
    print_success "报告已保存: $RESULT_FILE"
}

# 清理临时文件
cleanup() {
    print_section "清理临时文件"
    
    # 询问是否保留 rocprof 输出
    if [ -d "${OUTPUT_DIR}/prof_1" ]; then
        echo -n "是否删除 rocprof 临时文件? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "${OUTPUT_DIR}"/prof_*
            print_success "已删除临时文件"
        else
            print_info "保留 rocprof 输出: ${OUTPUT_DIR}/prof_*"
        fi
    fi
}

# ========== 主程序 ==========

main() {
    print_header
    
    # 解析参数
    MODE="fast"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --detailed|-d)
                MODE="detailed"
                shift
                ;;
            --iterations|-n)
                ITERATIONS="$2"
                shift 2
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  -n, --iterations N    运行 N 次迭代 (默认: 100)"
                echo "  -d, --detailed        使用 rocprofv3 进行详细性能分析"
                echo "  -h, --help            显示此帮助信息"
                echo ""
                echo "示例:"
                echo "  $0                    # 快速测试 100 次"
                echo "  $0 -n 50              # 快速测试 50 次"
                echo "  $0 -d -n 20           # 详细测试 20 次"
                exit 0
                ;;
            *)
                ITERATIONS="$1"
                shift
                ;;
        esac
    done
    
    # 检查环境
    check_executable
    setup_output_dir
    detect_config
    
    # 运行测试
    if [ "$MODE" == "detailed" ]; then
        # 检查 rocprofv3 是否可用
        if ! command -v rocprofv3 &> /dev/null; then
            print_error "rocprofv3 未找到，请安装 ROCm 或使用快速模式"
            exit 1
        fi
        run_detailed_benchmark
    else
        run_fast_benchmark
    fi
    
    # 生成报告
    generate_report
    
    # 清理
    if [ "$MODE" == "detailed" ]; then
        cleanup
    fi
    
    echo ""
    print_success "基准测试完成！"
    echo ""
}

# 运行主程序
main "$@"

