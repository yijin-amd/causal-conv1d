#!/bin/bash
# 使用 g++ 编译 LibTorch 程序并运行
 
# LibTorch 安装路径 (请修改为你自己的路径)
LIBTORCH_PATH=/root/libtorch
ARCH=gfx942

# 源文件# 输出可执行文件
SRC=casual_conv1d_opus.cpp
OUT=casual_conv1d_opus.exe

rm $OUT

# 编译
# g++ -std=c++17 $SRC -o $OUT \
#     -I${LIBTORCH_PATH}/include \
#     -I${LIBTORCH_PATH}/include/torch/csrc/api/include \
#     -L${LIBTORCH_PATH}/lib \
#     -ltorch -lc10 -ltorch_cpu \
#     -Wl,-rpath=${LIBTORCH_PATH}/lib

# /opt/rocm/bin/hipcc -std=c++17 $SRC -o $OUT \
/opt/rocm/bin/hipcc -x hip -std=c++17 $SRC -o $OUT \
    --offload-arch=$ARCH \
    -I/workspace/aiter/csrc/include \
    -I${LIBTORCH_PATH}/include \
    -I${LIBTORCH_PATH}/include/torch/csrc/api/include \
    -L${LIBTORCH_PATH}/lib \
    -Wl,-rpath=/root/libtorch/lib \
    -ltorch -lc10 -ltorch_cpu
# 运行
./$OUT

# 性能
HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace_casual_conv1d -o casual_conv1d_nch_4_64_2048_k_4 --stats -- ./$OUT
# HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace_casual_conv1d -o casual_conv1d_nch_2_64_2048_k_4 --stats -- python benchmark_performance.py -n 100
# HIP_VISIBLE_DEVICES=7 rocprofv3 -i input_att.yaml -- ./$OUT