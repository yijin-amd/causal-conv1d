#!/bin/bash

# ============================================================================
# HIP Kernel性能对比测试 - 编译和运行脚本
# 
# 功能:
#   对比两个kernel的性能:
#   1. causal_conv1d_fwd_kernel (channel-first布局)
#   2. causal_conv1d_channellast_fwd_kernel (channel-last布局)
#
# 使用方法:
#   ./test_hip_kernel_comparison.sh              # 编译并运行测试
#   ./test_hip_kernel_comparison.sh --compile    # 仅编译
#   ./test_hip_kernel_comparison.sh --run        # 仅运行(需先编译)
#   ./test_hip_kernel_comparison.sh --clean      # 清理生成的文件
# ============================================================================

set -e  # 遇到错误立即退出

echo "════════════════════════════════════════════════════════════════"
echo "  HIP Kernel 性能对比测试"
echo "  causal_conv1d_fwd_kernel vs causal_conv1d_channellast_fwd_kernel"
echo "════════════════════════════════════════════════════════════════"

# ==================== 配置参数 ====================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 源文件和输出文件
SOURCE_FILE="${SCRIPT_DIR}/test_hip_kernel_comparison.cpp"
OUTPUT_BIN="${SCRIPT_DIR}/test_hip_kernel_comparison"
OUTPUT_LOG="${SCRIPT_DIR}/hip_kernel_comparison_results.txt"

# HIP编译器
HIPCC=${HIPCC:-hipcc}

# 解析命令行参数
MODE="all"  # all, compile, run, clean
if [ "$1" = "--compile" ]; then
    MODE="compile"
elif [ "$1" = "--run" ]; then
    MODE="run"
elif [ "$1" = "--clean" ]; then
    MODE="clean"
fi

# ==================== 清理模式 ====================
if [ "${MODE}" = "clean" ]; then
    echo ""
    echo "【清理文件】"
    if [ -f "${OUTPUT_BIN}" ]; then
        rm -f "${OUTPUT_BIN}"
        echo "  ✓ 已删除: ${OUTPUT_BIN}"
    fi
    if [ -f "${OUTPUT_LOG}" ]; then
        rm -f "${OUTPUT_LOG}"
        echo "  ✓ 已删除: ${OUTPUT_LOG}"
    fi
    echo ""
    echo "清理完成!"
    exit 0
fi

# 检查hipcc是否可用
if ! command -v ${HIPCC} &> /dev/null; then
    echo "错误: 未找到 ${HIPCC}"
    echo "请确保已安装 ROCm 并设置了正确的环境变量"
    exit 1
fi

echo ""
echo "【编译配置】"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  源文件: ${SOURCE_FILE}"
echo "  输出可执行文件: ${OUTPUT_BIN}"
echo "  结果日志: ${OUTPUT_LOG}"
echo ""

# ==================== 检查文件是否存在 ====================

echo "【检查依赖文件】"

REQUIRED_FILES=(
    "${SOURCE_FILE}"
    "${PROJECT_ROOT}/rocm_backend/hip_backend/fwd/causal_conv1d_kernel.hip"
    "${PROJECT_ROOT}/rocm_backend/hip_backend/fwd/causal_conv1d_kernel_channellast.hip"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "  ✗ 缺失文件: ${file}"
        exit 1
    else
        echo "  ✓ 找到文件: ${file}"
    fi
done

echo ""

# ==================== 编译 ====================

if [ "${MODE}" = "all" ] || [ "${MODE}" = "compile" ]; then
    echo ""
    echo "【开始编译】"
    echo "  使用编译器: ${HIPCC}"

    # 编译选项
    COMPILE_FLAGS=(
        -O3
        -std=c++17
        -I${PROJECT_ROOT}
        -I${PROJECT_ROOT}/rocm_backend/hip_backend/fwd
        --offload-arch=gfx90a  # MI200系列 (MI250X, MI308等)
        # --offload-arch=gfx942  # 如果使用MI300系列，取消此行注释
        -ffast-math
        -munsafe-fp-atomics
    )

    echo "  编译选项: -O3 -std=c++17 --offload-arch=gfx90a"
    echo ""

    # 执行编译
    ${HIPCC} "${COMPILE_FLAGS[@]}" \
        "${SOURCE_FILE}" \
        -o "${OUTPUT_BIN}"

    if [ $? -eq 0 ]; then
        echo "  ✓ 编译成功!"
        echo "  生成可执行文件: ${OUTPUT_BIN}"
    else
        echo "  ✗ 编译失败!"
        exit 1
    fi

    echo ""
    
    # 如果只是编译模式，则退出
    if [ "${MODE}" = "compile" ]; then
        echo "仅编译模式，完成!"
        exit 0
    fi
fi

# ==================== 运行测试 ====================

echo "【开始性能测试】"
echo "  结果将保存到: ${OUTPUT_LOG}"
echo ""

# 检查GPU是否可用
if ! command -v rocm-smi &> /dev/null; then
    echo "警告: 未找到 rocm-smi，无法检查GPU状态"
else
    echo "【GPU状态】"
    rocm-smi --showtemp --showmeminfo vram --showuse
    echo ""
fi

# 运行测试并同时输出到终端和日志文件
"${OUTPUT_BIN}" 2>&1 | tee "${OUTPUT_LOG}"

EXIT_CODE=$?

echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✓ 测试完成!"
    echo "════════════════════════════════════════════════════════════════"
    echo "  结果已保存到: ${OUTPUT_LOG}"
    echo ""
    
    # 提取关键性能数据
    echo "【性能摘要】"
    if grep -q "Channel-Last 更快" "${OUTPUT_LOG}"; then
        echo "  总体趋势: Channel-Last kernel 性能更优"
    elif grep -q "Channel-First 更快" "${OUTPUT_LOG}"; then
        echo "  总体趋势: Channel-First kernel 性能更优"
    else
        echo "  总体趋势: 性能相当"
    fi
    echo ""
    
    # 统计测试数量
    TOTAL_TESTS=$(grep -c "测试配置" "${OUTPUT_LOG}" || echo "0")
    echo "  完成测试数量: ${TOTAL_TESTS}"
    echo ""
else
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✗ 测试失败 (退出码: ${EXIT_CODE})"
    echo "════════════════════════════════════════════════════════════════"
    exit ${EXIT_CODE}
fi

# ==================== 清理选项 ====================

# 如果设置了CLEANUP环境变量，删除可执行文件
if [ "${CLEANUP}" = "1" ]; then
    echo "【清理】"
    echo "  删除可执行文件: ${OUTPUT_BIN}"
    rm -f "${OUTPUT_BIN}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  完成!"
echo "════════════════════════════════════════════════════════════════"

