#include <cuda_runtime.h>
extern "C" __global__
void gap_global(const float* __restrict__ x, // [C,H,W] contiguous
                int C, int H, int W,
                float* __restrict__ y)       // [C]
{
    int c = blockIdx.x;                // 채널당 1 block
    if (c >= C) return;

    int HW = H * W;
    // 1D block에서 여러 원소를 순회하며 합산
    float sum = 0.f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        sum += x[c*HW + i];
    }

    // block 내 reduce (warp-unaware 간단 합)
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = sum;
    __syncthreads();

    // blockDim.x는 256 가정(launch에서 보장)
    for (int s = blockDim.x/2; s > 1; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // 마지막 2개를 정리
    if (tid == 0) {
        float tot = smem[0] + smem[1];
        y[c] = tot / (float)HW;
    }
}
