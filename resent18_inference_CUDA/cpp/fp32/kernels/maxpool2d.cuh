#pragma once
#include <cuda_runtime.h>

extern "C" __global__
void maxpool2d_3x3_s2p1_nchw(const float* __restrict__ x,
                             int N, int C, int H, int W,
                             float* __restrict__ y);
