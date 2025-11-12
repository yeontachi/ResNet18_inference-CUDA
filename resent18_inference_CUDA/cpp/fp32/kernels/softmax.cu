#include <cuda_runtime.h>
#include <math_constants.h>

// N=1, K=classes(1000) 벡터에 대해 softmax 수행
extern "C" __global__
void softmax_1d(const float* __restrict__ x, // [K]
                int K,
                float* __restrict__ y)       // [K]
{
    // 단일 block 처리 가정 (K<=2048 정도)
    __shared__ float smax;
    __shared__ float ssum;
    int tid = threadIdx.x;

    // 1) 최대값 찾기
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < K; i += blockDim.x)
        local_max = fmaxf(local_max, x[i]);

    __shared__ float tmp[256];
    tmp[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] = fmaxf(tmp[tid], tmp[tid + s]);
        __syncthreads();
    }
    if (tid == 0) smax = tmp[0];
    __syncthreads();

    // 2) exp(x - max) 합
    float local_sum = 0.f;
    for (int i = tid; i < K; i += blockDim.x)
        local_sum += __expf(x[i] - smax);

    tmp[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] += tmp[tid + s];
        __syncthreads();
    }
    if (tid == 0) ssum = tmp[0];
    __syncthreads();

    // 3) normalize
    for (int i = tid; i < K; i += blockDim.x)
        y[i] = __expf(x[i] - smax) / ssum;
}
