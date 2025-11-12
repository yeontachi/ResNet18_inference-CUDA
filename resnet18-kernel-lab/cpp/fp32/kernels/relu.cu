#include <cuda_runtime.h>

// x: 입력/출력(같은 버퍼, in-place), n: 요소 개수
extern "C" __global__
void relu_forward(float* x, int n){
    // 전역 인덱스
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 경계 체크 + 음수이면 0으로 클램프
    if (i < n && x[i] < 0.f) x[i] = 0.f;
}
