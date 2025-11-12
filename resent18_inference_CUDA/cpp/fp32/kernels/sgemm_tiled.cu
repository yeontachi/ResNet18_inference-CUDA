#include <cuda_runtime.h>
#define TILE 32

// A: MxK (row-major), B: KxN (row-major), C: MxN (row-major)
extern "C" __global__
void sgemm_tiled(const float* __restrict__ A, // MxK
                 const float* __restrict__ B, // KxN
                 float* __restrict__ C,       // MxN
                 int M,int N,int K)
{
    // 블록이 사용할 A/B의 타일(공유메모리) 버퍼
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];

    // 이 스레드가 담당하는 C의 좌표 (row, col)
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // 누적 레지스터
    float acc = 0.f;

    // K 차원을 TILE씩 순회하며 A와 B의 대응 타일을 로드→곱→누적
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // 현재 타일에서 A의 열, B의 행 인덱스
        int aCol = t * TILE + threadIdx.x;   // A[row, aCol]
        int bRow = t * TILE + threadIdx.y;   // B[bRow, col]

        // 경계 체크 후 공유메모리에 로드 (out-of-range는 0 패딩)
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.f;

        __syncthreads(); // 타일 로드 동기화

        // 타일 내 곱-누적: (TILE 길이 내적)
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads(); // 다음 타일 로드 전 동기화
    }

    // 경계 내면 결과 저장
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
