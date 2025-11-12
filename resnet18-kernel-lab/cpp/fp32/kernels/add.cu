#include <cuda_runtime.h>
extern "C" __global__
void add_inplace(float* __restrict__ y,
                 const float* __restrict__ x,
                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
