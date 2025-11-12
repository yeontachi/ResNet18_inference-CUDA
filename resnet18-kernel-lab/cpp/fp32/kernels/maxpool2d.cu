#include <float.h>
#include "maxpool2d.cuh"

extern "C" __global__
void maxpool2d_3x3_s2p1_nchw(const float* __restrict__ x,
                             int N, int C, int H, int W,
                             float* __restrict__ y)
{
    // grid: (oh*ow, C, N), block: (1,1,1)
    int nc = blockIdx.y;   // channel
    int n  = blockIdx.z;   // batch
    int ohow = blockIdx.x; // packed (oh,ow)

    int OH = (H + 2*1 - 3)/2 + 1; // p=1, k=3, s=2
    int OW = (W + 2*1 - 3)/2 + 1;

    int oh = ohow / OW;
    int ow = ohow % OW;

    const float* xnc = x + ((n*C + nc) * H * W);
    float* ync      = y + ((n*C + nc) * OH * OW);
    float vmax = -FLT_MAX;

    // 창의 좌측상단 입력 좌표 (패딩 포함)
    int ih0 = oh * 2 - 1;
    int iw0 = ow * 2 - 1;

    #pragma unroll
    for (int kh=0; kh<3; ++kh){
        int ih = ih0 + kh;
        if (ih < 0 || ih >= H) continue;
        #pragma unroll
        for (int kw=0; kw<3; ++kw){
            int iw = iw0 + kw;
            if (iw < 0 || iw >= W) continue;
            float v = xnc[ih*W + iw];
            vmax = v > vmax ? v : vmax;
        }
    }
    ync[oh*OW + ow] = vmax;
}
