rm -f ./test_rocm_fwd_conv1d_hip
hipcc test_rocm_fwd_conv1d_hip.cpp -o test_rocm_fwd_conv1d_hip
hipcc -O3 -std=c++17 test_rocm_fwd_conv1d_hip.cpp \
    -o test_rocm_fwd_conv1d_hip \
    --offload-arch=gfx942
    # -ffast-math \              # 保留！提升 8-12%
    # -munsafe-fp-atomics        # 保留！保险+匹配CUDA
./test_rocm_fwd_conv1d_hip
