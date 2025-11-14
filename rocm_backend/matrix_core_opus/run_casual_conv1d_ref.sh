#!/bin/bash
# 使用 g++ 编译 LibTorch 程序并运行
 
# LibTorch 安装路径 (请修改为你自己的路径)
LIBTORCH_PATH=/root/libtorch
ARCH=gfx942

# 源文件# 输出可执行文件
SRC=casual_conv1d_opus.cpp
OUT=casual_conv1d_opus.exe
# SRC=casual_conv1d_copilot.cpp
# OUT=casual_conv1d_copilot.exe
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
# HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace -o matrix_out --stats -- ./build/matrix_core.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace_casual_conv1d -o matrix_casual_conv1d_out_batch_4_c_64_l_2048_kernel_size_4 --stats -- ./casual_conv1d_opus.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace_casual_conv1d -o casual_conv1d_nch_1_64_2048_k_4 --stats -- ./conv1d_libtorch_ref.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 -i input_att.yaml -- ./build/matrix_core.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 -i input_att.yaml -- ./build_casual_conv1d_test/matrix_core_casual_conv1d.exe