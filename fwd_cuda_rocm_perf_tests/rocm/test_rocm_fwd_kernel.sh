rm -f ./test_rocm_fwd_conv1d_hip
hipcc test_rocm_fwd_conv1d_hip.cpp -o test_rocm_fwd_conv1d_hip
./test_rocm_fwd_conv1d_hip
