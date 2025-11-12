#include <cuda_runtime.h>

// x: 입력/출력 텐서 (in-place). 1배치 가정: (C, OH, OW) 혹은 평탄화(C*OH*OW).
// g: gamma[C], b: beta[C], m: running_mean[C], v: running_var[C] (모두 채널별 파라미터)
extern "C" __global__
void bn_inference(float* x,          // (C,OH,OW) 혹은 (C*OH*OW) 1배치 가정
                  const float* g,    // gamma[C]
                  const float* b,    // beta[C]
                  const float* m,    // mean[C]
                  const float* v,    // var[C]
                  float eps,         // 수치 안정화용 epsilon
                  int C,int OH,int OW)
{
    // 전역 인덱스
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * OH * OW;
    if (idx >= total) return;

    // 채널 인덱스 복원: (C, OH, OW) 평탄화 기준
    // 한 채널이 OH*OW 연속 블록을 차지하므로,
    // idx / (OH*OW) → 채널 번호 c
    int c = (idx / (OH * OW)) % C;

    // BN 추론 공식:
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    float y = (x[idx] - m[c]) / sqrtf(v[c] + eps);
    x[idx] = g[c] * y + b[c]; // in-place 업데이트
}
