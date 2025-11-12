#pragma once
extern "C" __global__
void gap_global(const float* __restrict__ x, // (C,H,W) 또는 (N=1,C,H,W)의 연속 메모리
                int C, int H, int W,
                float* __restrict__ out);    // out[C], 채널별 평균
