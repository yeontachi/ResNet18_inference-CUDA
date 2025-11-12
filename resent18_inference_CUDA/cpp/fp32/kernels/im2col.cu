#include <cuda_runtime.h>

// x: 입력(N*C*H*W, 여기선 N=1만 처리)
// col: 출력((C*kH*kW) x (N*OH*OW)) 행렬을 1D(row-major)로 저장
extern "C" __global__
void im2col_nchw(const float* __restrict__ x, // 입력: N*C*H*W (N=1 가정)
                 int N,int C,int H,int W,    // 입력 크기
                 int kH,int kW,int sH,int sW,int pH,int pW, // 커널/스트라이드/패딩
                 float* __restrict__ col)    // 출력: (C*kH*kW, N*OH*OW) row-major
{
    // 단순화를 위해 배치 차원은 N=1만 지원(필요시 n 차원 병렬화 가능)
    int n = 0;

    // 출력 특성맵 크기 계산
    int OH = (H + 2*pH - kH)/sH + 1;
    int OW = (W + 2*pW - kW)/sW + 1;

    // 블록/스레드에서 출력 위치(oh, ow) 담당
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    // 경계 밖은 바로 반환
    if (oh >= OH || ow >= OW) return;

    // 현재 (oh, ow)에 대한 열(column) 인덱스
    // 열 수 = N*OH*OW, N=1이므로 OH*OW
    int out_index = oh * OW + ow;
    int colStride = OH * OW; // row-major에서 한 행(row)의 stride(=열 수)

    // 모든 채널과 커널 위치(kh, kw)를 순회하면서
    // 해당 수용영역 픽셀을 col의 (row=r, col=out_index)에 기록
    for (int c = 0; c < C; ++c){
        for (int kh = 0; kh < kH; ++kh){
            for (int kw = 0; kw < kW; ++kw){

                // 입력 좌표(ih, iw): 스트라이드/패딩 적용
                int ih = oh * sH - pH + kh;
                int iw = ow * sW - pW + kw;

                // 패딩 영역(경계 밖)은 0으로 채움
                float v = 0.f;
                if (ih >= 0 && iw >= 0 && ih < H && iw < W){
                    // N=1 가정이므로 오프셋은 c*H*W + ih*W + iw
                    int idx = c * H * W + ih * W + iw;
                    v = x[idx];
                }

                // (row=r) = 채널과 커널 위치를 평탄화한 인덱스
                // r ∈ [0, C*kH*kW)
                int r = c * kH * kW + kh * kW + kw;

                // row-major 2D -> 1D: row * colStride + col
                // colStride = 전체 열 수 = N*OH*OW (N=1 → OH*OW)
                col[r * colStride + out_index] = v;
            }
        }
    }
}
