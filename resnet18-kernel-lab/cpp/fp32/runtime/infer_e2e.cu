// cpp/fp32/runtime/infer_e2e.cu
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "utils.hpp"
#include "../kernels/gap_global.cuh"
#include "../kernels/maxpool2d.cuh"

// ==== 외부 커널 (이미 구현) ====
extern "C" __global__
void im2col_nchw(const float* x, int N,int C,int H,int W,
                 int kH,int kW,int sH,int sW,int pH,int pW,
                 float* col);

extern "C" __global__
void sgemm_tiled(const float* A, const float* B, float* C,
                 int M,int N,int K);

extern "C" __global__
void bn_inference(float* x, const float* g, const float* b,
                  const float* m, const float* v, float eps,
                  int C,int OH,int OW);

extern "C" __global__
void relu_forward(float* x, int n);

extern "C" __global__
void add_inplace(float* y, const float* x, int n);


// --- Reliable GAP (N=1, NCHW) ---
// out[c] = mean_{h,w} in[c,h,w]
__global__ void gap_global_ref(const float* __restrict__ x,
                               int C, int H, int W,
                               float* __restrict__ out) {
    int c = blockIdx.x;      // 1 block per channel
    if (c >= C) return;
    const int HW = H * W;
    const float* base = x + c * HW;  // N=1 가정, 채널별 연속
    float sum = 0.0f;
    // 병렬 누적
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        sum += base[i];
    }
    // warp/thread block reduce
    __shared__ float ssum[256];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    // 간단한 트리 리덕션(256 가정)
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[c] = ssum[0] / (float)HW;  // 평균
    }
}

// ---- 런처 유틸 ----
static inline void gemm_launch(const float* A, const float* B, float* C,
                               int M, int N, int K)
{
    dim3 blk(32,32);
    dim3 grd( (N+blk.x-1)/blk.x, (M+blk.y-1)/blk.y );
    sgemm_tiled<<<grd,blk>>>(A, B, C, M, N, K);
}

static inline void im2col_launch(const float* x,
                                 int N,int C,int H,int W,
                                 int kH,int kW,int sH,int sW,int pH,int pW,
                                 int OH,int OW,
                                 float* col)
{
    dim3 blk(16,16);
    dim3 grd( (OW+blk.x-1)/blk.x, (OH+blk.y-1)/blk.y );
    im2col_nchw<<<grd,blk>>>(x, N,C,H,W, kH,kW,sH,sW,pH,pW, col);
}

static inline void bn_launch(float* y,
                             const std::vector<float>& G,
                             const std::vector<float>& B,
                             const std::vector<float>& M,
                             const std::vector<float>& V,
                             int C, int OH, int OW,
                             float eps=1e-5f)
{
    auto dG = copy_to_device(G);
    auto dB = copy_to_device(B);
    auto dM = copy_to_device(M);
    auto dV = copy_to_device(V);
    int total = C*OH*OW;
    bn_inference<<< div_up(total,256), 256 >>>(y, dG.get(), dB.get(), dM.get(), dV.get(), eps, C, OH, OW);
}

// ---- conv2d (im2col+gemm) ----
//   W layout: [OC, IC, kH, kW] (torch)
//   내부에서 W를 [OC, IC*kH*kW]로 변환 후 GEMM
static std::unique_ptr<float, DeviceDeleter>
conv2d_nchw_im2col_gemm(const float* dX,
                        int N, int IC, int H_in, int W_in,
                        const std::vector<float>& Wk,   // weights [OC*IC*kH*kW]
                        int OC, int kH, int kW, int sH, int sW, int pH, int pW,
                        int& OH, int& OW)
{
    OH = (H_in + 2*pH - kH)/sH + 1;
    OW = (W_in + 2*pW - kW)/sW + 1;

    const int KCOL = IC * kH * kW;

    // Wk -> Wcol(OC x KCOL)
    std::vector<float> Wcol(OC * KCOL);
    for (int o = 0; o < OC; ++o){
        for (int c = 0; c < IC; ++c){
            for (int kh = 0; kh < kH; ++kh){
                for (int kw = 0; kw < kW; ++kw){
                    int r   = c*kH*kW + kh*kW + kw;
                    int src = o*IC*kH*kW + c*kH*kW + kh*kW + kw;
                    Wcol[o*KCOL + r] = Wk[src];
                }
            }
        }
    }

    auto dW   = copy_to_device(Wcol);
    auto dCol = make_device_f32((size_t)KCOL * OH * OW);
    auto dY   = make_device_f32((size_t)OC   * OH * OW);

    im2col_launch(dX, N, IC, H_in, W_in, kH, kW, sH, sW, pH, pW, OH, OW, dCol.get());
    gemm_launch(dW.get(), dCol.get(), dY.get(), OC, OH*OW, KCOL);

    return dY; // [OC, OH, OW]
}

// ---- BasicBlock: (conv-bn-relu) -> (conv-bn) + skip(add) -> relu ----
struct BlockParams {
    // main path conv1
    std::vector<float> w1, g1, b1, m1, v1;
    int k1H=3, k1W=3, s1H=1, s1W=1, p1H=1, p1W=1;

    // main path conv2
    std::vector<float> w2, g2, b2, m2, v2;
    int k2H=3, k2W=3, s2H=1, s2W=1, p2H=1, p2W=1;

    // downsample?
    bool use_down = false;
    std::vector<float> wd, gd, bd, md, vd; // 1x1 conv + bn
    int sdH=1, sdW=1, pdH=0, pdW=0;       // stride/pad for 1x1
};

// 입력 dIn: [IC, H, W]
// 출력:    [OC, OH, OW]
static std::unique_ptr<float, DeviceDeleter>
basic_block_forward(const std::unique_ptr<float, DeviceDeleter>& dIn,
                    int N, int IC, int H, int W,
                    int OC,
                    const BlockParams& P)
{
    // --- conv1 ---
    int c1OH=0, c1OW=0;
    auto dC1 = conv2d_nchw_im2col_gemm(dIn.get(), N,IC,H,W,
                                       P.w1, OC, P.k1H,P.k1W, P.s1H,P.s1W, P.p1H,P.p1W,
                                       c1OH, c1OW);
    // bn1 + relu
    bn_launch(dC1.get(), P.g1, P.b1, P.m1, P.v1, OC, c1OH, c1OW);
    relu_forward<<< div_up(OC*c1OH*c1OW,256), 256 >>>(dC1.get(), OC*c1OH*c1OW);

    // --- conv2 ---
    int c2OH=0, c2OW=0;
    auto dC2 = conv2d_nchw_im2col_gemm(dC1.get(), N,OC,c1OH,c1OW,
                                       P.w2, OC, P.k2H,P.k2W, P.s2H,P.s2W, P.p2H,P.p2W,
                                       c2OH, c2OW);
    // bn2
    bn_launch(dC2.get(), P.g2, P.b2, P.m2, P.v2, OC, c2OH, c2OW);

    // --- skip ---
    std::unique_ptr<float, DeviceDeleter> dSkip;
    if (!P.use_down) {
        // identity: 입력과 출력의 공간/채널 동일해야 함
        dSkip = make_device_f32((size_t)OC*c2OH*c2OW);
        CUDA_CHECK(cudaMemcpy(dSkip.get(), dIn.get(),
                              (size_t)OC*c2OH*c2OW*sizeof(float),
                              cudaMemcpyDeviceToDevice));
    } else {
        // 1x1 conv (stride=sdH,sdW) + bn
        int dsOH=0, dsOW=0;
        auto dDS = conv2d_nchw_im2col_gemm(dIn.get(), N,IC,H,W,
                                           P.wd, OC, 1,1, P.sdH,P.sdW, P.pdH,P.pdW,
                                           dsOH, dsOW);
        bn_launch(dDS.get(), P.gd, P.bd, P.md, P.vd, OC, dsOH, dsOW);
        dSkip = std::move(dDS);
        assert(dsOH==c2OH && dsOW==c2OW);
    }

    // --- add + relu ---
    add_inplace<<< div_up(OC*c2OH*c2OW,256), 256 >>>(dC2.get(), dSkip.get(), OC*c2OH*c2OW);
    relu_forward<<< div_up(OC*c2OH*c2OW,256), 256 >>>(dC2.get(), OC*c2OH*c2OW);

    return dC2; // [OC, c2OH, c2OW]
}

// ---- FC ----
static inline void fc_forward(const float* gap,            // [512]
                              const std::vector<float>& W, // [1000,512]
                              const std::vector<float>& B, // [1000]
                              std::vector<float>& out)     // [1000]
{
    const int O = 1000, I = 512;
    auto dGap = copy_to_device(gap, (size_t)I);
    auto dW   = copy_to_device(W);
    auto dOut = make_device_f32((size_t)O);
    // W[O x I] * x[I] = y[O x 1]
    gemm_launch(dW.get(), dGap.get(), dOut.get(), O, 1, I);
    out = copy_to_host(dOut, (size_t)O);
    for (int o=0; o<O; ++o) out[o] += B[o];
}

static void usage(){
    std::cout << "step8_e2e --manifest <dir> --input <input.bin> [--dump_dir <dir>]\n";
}

// ---- 편의 로더 ----
static inline std::vector<float> L(const std::string& root, const std::string& name, size_t n){
    return load_bin_f32(root + "/" + name + ".bin", n);
}

int main(int argc, char** argv)
{
    std::string mani, input_path, dump_dir;

    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest" && i+1<argc) mani = argv[++i];
        else if (a=="--input" && i+1<argc) input_path = argv[++i];
        else if (a=="--dump_dir" && i+1<argc) dump_dir = argv[++i];
    }
    if (mani.empty() || input_path.empty()) { usage(); return 1; }

    // 저장 헬퍼 (덤프 요청 시 파일로 저장)
    auto maybe_save = [&](const std::string& name, const std::vector<float>& v){
        if (!dump_dir.empty()){
            ensure_out_dir(dump_dir.c_str());
            save_bin_f32(dump_dir + "/" + name, v);
        }
    };

    ensure_out_dir(); // 기본 out/ 보장
    Timer T;

    // ---------- 0) 입력 ----------
    const int N=1, C=3, H=224, W=224;
    std::vector<float> X = load_bin_f32(input_path, (size_t)N*C*H*W);
    auto dX = copy_to_device(X);

    // ---------- 1) Stem: conv1(7x7,s2,p3) ----------
    const int OC0 = 64, kH0 = 7, kW0 = 7, sH0 = 2, sW0 = 2, pH0 = 3, pW0 = 3;
    int OH0 = 0, OW0 = 0;

    std::vector<float> Wc1 = load_bin_f32(mani + "/conv1.weight.bin",
                                          (size_t)OC0*C*kH0*kW0);

    auto dY0 = conv2d_nchw_im2col_gemm(
        dX.get(), N, C, H, W,
        Wc1,
        OC0, kH0, kW0, sH0, sW0, pH0, pW0,
        OH0, OW0
    );

    // BN1 + ReLU
    std::vector<float> G1 = load_bin_f32(mani + "/bn1.weight.bin",       OC0);
    std::vector<float> B1 = load_bin_f32(mani + "/bn1.bias.bin",         OC0);
    std::vector<float> M1 = load_bin_f32(mani + "/bn1.running_mean.bin", OC0);
    std::vector<float> V1 = load_bin_f32(mani + "/bn1.running_var.bin",  OC0);

    bn_launch(dY0.get(), G1, B1, M1, V1, OC0, OH0, OW0, /*eps=*/1e-5f);
    relu_forward<<< div_up(OC0*OH0*OW0,256), 256 >>>(dY0.get(), OC0*OH0*OW0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- MaxPool(3x3, s=2, p=1) ----------
    const int MPH=3, MPW=3, MPS=2, MPP=1;
    const int POH = (OH0 + 2*MPP - MPH)/MPS + 1;  // 56
    const int POW = (OW0 + 2*MPP - MPW)/MPS + 1;  // 56

    auto dStemPool = make_device_f32((size_t)OC0*POH*POW);
    {
        dim3 blk(1,1,1);
        dim3 grd(POH*POW, OC0, N); // (oh,ow) in grid.x, C in grid.y, N in grid.z
        maxpool2d_3x3_s2p1_nchw<<<grd, blk>>>(dY0.get(), N, OC0, OH0, OW0, dStemPool.get());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 덤프: stem_pool
    {
        auto hStem = copy_to_host(dStemPool, (size_t)64*56*56);
        maybe_save("stem_pool.bin", hStem);
    }

    // ---------- 2) Layer1 (64ch, 2 blocks, stride=1) ----------
    auto dCur = std::move(dStemPool);
    int IC = 64, Hc = POH, Wc = POW;

    auto load_block = [&](const std::string& base, int inC, int outC,
                          int sH, int sW, bool downsample)->BlockParams {
        BlockParams P;
        // conv1: 3x3
        P.w1 = L(mani, base + ".conv1.weight", (size_t)outC*inC*3*3);
        P.g1 = L(mani, base + ".bn1.weight", outC);
        P.b1 = L(mani, base + ".bn1.bias",   outC);
        P.m1 = L(mani, base + ".bn1.running_mean", outC);
        P.v1 = L(mani, base + ".bn1.running_var",  outC);
        P.k1H=3; P.k1W=3; P.s1H=sH; P.s1W=sW; P.p1H=1; P.p1W=1;

        // conv2: 3x3 stride=1
        P.w2 = L(mani, base + ".conv2.weight", (size_t)outC*outC*3*3);
        P.g2 = L(mani, base + ".bn2.weight", outC);
        P.b2 = L(mani, base + ".bn2.bias",   outC);
        P.m2 = L(mani, base + ".bn2.running_mean", outC);
        P.v2 = L(mani, base + ".bn2.running_var",  outC);
        P.k2H=3; P.k2W=3; P.s2H=1; P.s2W=1; P.p2H=1; P.p2W=1;

        P.use_down = downsample;
        if (downsample) {
            // downsample: 1x1 conv stride=(sH,sW), no padding
            P.wd = L(mani, base + ".downsample.0.weight", (size_t)outC*inC*1*1);
            P.gd = L(mani, base + ".downsample.1.weight", outC);
            P.bd = L(mani, base + ".downsample.1.bias",   outC);
            P.md = L(mani, base + ".downsample.1.running_mean", outC);
            P.vd = L(mani, base + ".downsample.1.running_var",  outC);
            P.sdH=sH; P.sdW=sW; P.pdH=0; P.pdW=0;
        }
        return P;
    };

    // layer1.0
    {
        BlockParams P = load_block("layer1.0", /*inC=*/64, /*outC=*/64, /*sH=*/1,/*sW=*/1, /*down=*/false);
        dCur = basic_block_forward(dCur, N, /*IC*/64, Hc, Wc, /*OC*/64, P);
        // Hc,Wc 유지 (56x56)
    }
    // layer1.1
    {
        BlockParams P = load_block("layer1.1", /*inC=*/64, /*outC=*/64, /*sH=*/1,/*sW=*/1, /*down=*/false);
        dCur = basic_block_forward(dCur, N, /*IC*/64, Hc, Wc, /*OC*/64, P);
    }
    // 덤프: layer1
    {
        auto hL1 = copy_to_host(dCur, (size_t)64*56*56);
        maybe_save("layer1.bin", hL1);
    }
    IC=64;

    // ---------- 3) Layer2 (128ch, 첫 블록 stride=2/downsample, 다음 stride=1) ----------
    // layer2.0
    {
        BlockParams P = load_block("layer2.0", /*inC=*/IC, /*outC=*/128, /*sH=*/2,/*sW=*/2, /*down=*/true);
        dCur = basic_block_forward(dCur, N, /*IC*/IC, Hc, Wc, /*OC*/128, P);
        Hc = (Hc + 2*1 - 3)/2 + 1;   // 56->28
        Wc = (Wc + 2*1 - 3)/2 + 1;
        IC = 128;
    }
    // layer2.1
    {
        BlockParams P = load_block("layer2.1", /*inC=*/128, /*outC=*/128, /*sH=*/1,/*sW=*/1, /*down=*/false);
        dCur = basic_block_forward(dCur, N, /*IC*/128, Hc, Wc, /*OC*/128, P);
    }
    // 덤프: layer2
    {
        auto hL2 = copy_to_host(dCur, (size_t)128*28*28);
        maybe_save("layer2.bin", hL2);
    }

    // ---------- 4) Layer3 (256ch, 첫 블록 stride=2/downsample) ----------
    // layer3.0
    {
        BlockParams P = load_block("layer3.0", /*inC=*/128, /*outC=*/256, /*sH=*/2,/*sW=*/2, /*down=*/true);
        dCur = basic_block_forward(dCur, N, /*IC*/128, Hc, Wc, /*OC*/256, P);
        Hc = (Hc + 2*1 - 3)/2 + 1;   // 28->14
        Wc = (Wc + 2*1 - 3)/2 + 1;
        IC = 256;
    }
    // layer3.1
    {
        BlockParams P = load_block("layer3.1", /*inC=*/256, /*outC=*/256, /*sH=*/1,/*sW=*/1, /*down=*/false);
        dCur = basic_block_forward(dCur, N, /*IC*/256, Hc, Wc, /*OC*/256, P);
    }
    // 덤프: layer3
    {
        auto hL3 = copy_to_host(dCur, (size_t)256*14*14);
        maybe_save("layer3.bin", hL3);
    }

    // ---------- 5) Layer4 (512ch, 첫 블록 stride=2/downsample) ----------
    // layer4.0
    {
        BlockParams P = load_block("layer4.0", /*inC=*/256, /*outC=*/512, /*sH=*/2,/*sW=*/2, /*down=*/true);
        dCur = basic_block_forward(dCur, N, /*IC*/256, Hc, Wc, /*OC*/512, P);
        Hc = (Hc + 2*1 - 3)/2 + 1;   // 14->7
        Wc = (Wc + 2*1 - 3)/2 + 1;
        IC = 512;
    }
    // layer4.1
    {
        BlockParams P = load_block("layer4.1", /*inC=*/512, /*outC=*/512, /*sH=*/1,/*sW=*/1, /*down=*/false);
        dCur = basic_block_forward(dCur, N, /*IC*/512, Hc, Wc, /*OC*/512, P);
    }
    // 최종: [1,512,7,7]
    assert(IC==512 && Hc==7 && Wc==7);

    // 덤프: layer4
    {
        auto hL4 = copy_to_host(dCur, (size_t)512*7*7);
        maybe_save("layer4.bin", hL4);
    }

    // ---------- 6) GAP + FC ----------
    auto dGAP = make_device_f32((size_t)IC);
    {
        dim3 blk(256);
        dim3 grd(IC); // 1 block per channel
        gap_global_ref<<<grd, blk>>>(dCur.get(), IC, Hc, Wc, dGAP.get());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto hGAP = copy_to_host(dGAP, (size_t)IC);
    maybe_save("gap.bin", hGAP);

    auto Wfc = L(mani, "fc.weight", (size_t)1000*512);
    auto Bfc = L(mani, "fc.bias",   (size_t)1000);

    std::vector<float> logits;
    fc_forward(hGAP.data(), Wfc, Bfc, logits);
    maybe_save("logits.bin", logits);

    // top-1
    int top=-1; float best=-1e30f;
    for (int i=0;i<1000;i++) if (logits[i] > best){ best=logits[i]; top=i; }
    std::cout<<"[E2E] top-1 class index = "<<top<<", logit="<<best<<"\n";
    std::cout<<"(주의: synset 매핑은 별도)\n";
    return 0;
}
