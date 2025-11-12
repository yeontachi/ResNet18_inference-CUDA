#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include "utils.hpp"   // CUDA_CHECK, Timer, load_bin_f32, 래퍼(CmdArgs/Manifest/Tensor/parse_args) 포함

// ---- 외부 커널 선언 (kernels/*.cu 에 구현) ----
extern "C" __global__
void im2col_nchw(const float* __restrict__ x, // N*C*H*W (N=1 가정)
                 int N,int C,int H,int W,
                 int kH,int kW,int sH,int sW,int pH,int pW,
                 float* __restrict__ col);   // (C*kH*kW, N*OH*OW)

extern "C" __global__
void sgemm_tiled(const float* __restrict__ A, // (M x K)
                 const float* __restrict__ B, // (K x N)
                 float* __restrict__ C,       // (M x N)
                 int M,int N,int K);

extern "C" __global__
void bn_inference(float* x,          // in-place, (C,OH,OW)
                  const float* g,    // gamma[C]
                  const float* b,    // beta[C]
                  const float* m,    // mean[C]
                  const float* v,    // var[C]
                  float eps,
                  int C,int OH,int OW);

extern "C" __global__
void relu_forward(float* x, int n);

extern "C" __global__
void add_inplace(float* __restrict__ y, const float* __restrict__ x, int n);

// ---- layer1 공통 파라미터 (stem 이후) ----
struct L1Cfg {
    // 입력: stem(maxpool) 출력 기준 (1, 64, 56, 56)
    static constexpr int N   = 1;
    static constexpr int C   = 64;
    static constexpr int H   = 56;
    static constexpr int W   = 56;
    // conv: 3x3, stride=1, pad=1, 채널 동일(64 -> 64)
    static constexpr int OC  = 64;
    static constexpr int KH  = 3;
    static constexpr int KW  = 3;
    static constexpr int SH  = 1;
    static constexpr int SW  = 1;
    static constexpr int PH  = 1;
    static constexpr int PW  = 1;
    static constexpr int OH  = (H + 2*PH - KH)/SH + 1; // 56
    static constexpr int OW  = (W + 2*PW - KW)/SW + 1; // 56
    static constexpr int KCOL= C*KH*KW;   // 64*3*3 = 576
    static constexpr int NCOL= OH*OW;     // 56*56   = 3136
    static constexpr int ELEMS = OC*OH*OW; // 64*56*56 = 200704
};

// (OIHW -> O x (I*kH*kW))로 변환
static void make_Wcol_OxK(const std::vector<float>& w_oi_hw, // size = OC*IC*KH*KW
                          int OC, int IC, int KH, int KW,
                          std::vector<float>& Wcol)          // size = OC*(IC*KH*KW)
{
    const int K = IC*KH*KW;
    Wcol.resize(OC * K);
    for (int o=0;o<OC;o++){
        for (int c=0;c<IC;c++){
            for (int kh=0;kh<KH;kh++){
                for (int kw=0;kw<KW;kw++){
                    int r = c*KH*KW + kh*KW + kw; // row in Wcol
                    int src = o*IC*KH*KW + c*KH*KW + kh*KW + kw; // OIHW
                    Wcol[o*K + r] = w_oi_hw[src];
                }
            }
        }
    }
}

// 한 번의 conv(+bn) 실행: in -> im2col -> gemm -> bn -> (선택) relu
static void conv_bn_forward(float* dIn,          // (C,H,W) with N=1
                            float* dCol,         // (KCOL, NCOL)
                            float* dWcol,        // (OC, KCOL)
                            float* dOut,         // (OC, NCOL)
                            float* dGamma, float* dBeta,
                            float* dMean,  float* dVar,
                            bool with_relu,
                            Timer& T,
                            float& t_im2col, float& t_gemm, float& t_bn, float& t_relu)
{
    // 1) im2col
    {
        dim3 blk(16,16);
        dim3 grd( (L1Cfg::OW+blk.x-1)/blk.x, (L1Cfg::OH+blk.y-1)/blk.y );
        T.start();
        im2col_nchw<<<grd,blk>>>(dIn, L1Cfg::N, L1Cfg::C, L1Cfg::H, L1Cfg::W,
                                 L1Cfg::KH, L1Cfg::KW, L1Cfg::SH, L1Cfg::SW,
                                 L1Cfg::PH, L1Cfg::PW, dCol);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_im2col += T.stop();
    }
    // 2) GEMM: (OC x KCOL) * (KCOL x NCOL) -> (OC x NCOL)
    {
        dim3 blk(32,32);
        dim3 grd( (L1Cfg::NCOL+blk.x-1)/blk.x, (L1Cfg::OC+blk.y-1)/blk.y );
        T.start();
        sgemm_tiled<<<grd,blk>>>(dWcol, dCol, dOut, L1Cfg::OC, L1Cfg::NCOL, L1Cfg::KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_gemm += T.stop();
    }
    // 3) BN
    {
        T.start();
        bn_inference<<< (L1Cfg::ELEMS+255)/256, 256 >>>(dOut, dGamma, dBeta, dMean, dVar,
                                                        1e-5f, L1Cfg::OC, L1Cfg::OH, L1Cfg::OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_bn += T.stop();
    }
    // 4) ReLU (옵션)
    if (with_relu){
        T.start();
        relu_forward<<< (L1Cfg::ELEMS+255)/256, 256 >>>(dOut, L1Cfg::ELEMS);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_relu += T.stop();
    }
}

static void usage(){
    std::cout << "usage: step3_layer1 --manifest <exports/resnet18/fp32>\n";
}

int main(int argc, char** argv)
{
    // --- 인자 파싱 (래퍼 사용 가능하나, 여기선 간단히 문자열만)
    std::string manifest;
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest" && i+1<argc) manifest = argv[++i];
    }
    if (manifest.empty()){ usage(); return 1; }

    // ---- 1) 입력(Stem 이후) 및 기대 출력(블록0/1) 로드 ----
    auto hX0   = load_bin_f32(manifest + "/fixtures_step3/stem_after_maxpool.bin", L1Cfg::ELEMS);
    auto yE_b0 = load_bin_f32(manifest + "/fixtures_step3/layer1_block0_out.bin",  L1Cfg::ELEMS);
    auto yE_b1 = load_bin_f32(manifest + "/fixtures_step3/layer1_block1_out.bin",  L1Cfg::ELEMS);

    // ---- 2) 가중치 로드 & Wcol 변환 (4개의 conv) ----
    // layer1.0
    auto W10 = load_bin_f32(manifest + "/layer1.0.conv1.weight.bin", L1Cfg::OC*L1Cfg::C*L1Cfg::KH*L1Cfg::KW);
    auto G10 = load_bin_f32(manifest + "/layer1.0.bn1.weight.bin",   L1Cfg::OC);
    auto B10 = load_bin_f32(manifest + "/layer1.0.bn1.bias.bin",     L1Cfg::OC);
    auto M10 = load_bin_f32(manifest + "/layer1.0.bn1.running_mean.bin", L1Cfg::OC);
    auto V10 = load_bin_f32(manifest + "/layer1.0.bn1.running_var.bin",  L1Cfg::OC);

    auto W11 = load_bin_f32(manifest + "/layer1.0.conv2.weight.bin", L1Cfg::OC*L1Cfg::C*L1Cfg::KH*L1Cfg::KW);
    auto G11 = load_bin_f32(manifest + "/layer1.0.bn2.weight.bin",   L1Cfg::OC);
    auto B11 = load_bin_f32(manifest + "/layer1.0.bn2.bias.bin",     L1Cfg::OC);
    auto M11 = load_bin_f32(manifest + "/layer1.0.bn2.running_mean.bin", L1Cfg::OC);
    auto V11 = load_bin_f32(manifest + "/layer1.0.bn2.running_var.bin",  L1Cfg::OC);

    // layer1.1
    auto W20 = load_bin_f32(manifest + "/layer1.1.conv1.weight.bin", L1Cfg::OC*L1Cfg::C*L1Cfg::KH*L1Cfg::KW);
    auto G20 = load_bin_f32(manifest + "/layer1.1.bn1.weight.bin",   L1Cfg::OC);
    auto B20 = load_bin_f32(manifest + "/layer1.1.bn1.bias.bin",     L1Cfg::OC);
    auto M20 = load_bin_f32(manifest + "/layer1.1.bn1.running_mean.bin", L1Cfg::OC);
    auto V20 = load_bin_f32(manifest + "/layer1.1.bn1.running_var.bin",  L1Cfg::OC);

    auto W21 = load_bin_f32(manifest + "/layer1.1.conv2.weight.bin", L1Cfg::OC*L1Cfg::C*L1Cfg::KH*L1Cfg::KW);
    auto G21 = load_bin_f32(manifest + "/layer1.1.bn2.weight.bin",   L1Cfg::OC);
    auto B21 = load_bin_f32(manifest + "/layer1.1.bn2.bias.bin",     L1Cfg::OC);
    auto M21 = load_bin_f32(manifest + "/layer1.1.bn2.running_mean.bin", L1Cfg::OC);
    auto V21 = load_bin_f32(manifest + "/layer1.1.bn2.running_var.bin",  L1Cfg::OC);

    // Wcol 변환
    std::vector<float> W10col, W11col, W20col, W21col;
    make_Wcol_OxK(W10, L1Cfg::OC, L1Cfg::C, L1Cfg::KH, L1Cfg::KW, W10col);
    make_Wcol_OxK(W11, L1Cfg::OC, L1Cfg::C, L1Cfg::KH, L1Cfg::KW, W11col);
    make_Wcol_OxK(W20, L1Cfg::OC, L1Cfg::C, L1Cfg::KH, L1Cfg::KW, W20col);
    make_Wcol_OxK(W21, L1Cfg::OC, L1Cfg::C, L1Cfg::KH, L1Cfg::KW, W21col);

    // ---- 3) 디바이스 버퍼 할당/복사 ----
    float *dIn, *dCol, *dW, *dOut, *dSkip; // dSkip = identity용 입력 보관
    CUDA_CHECK(cudaMalloc(&dIn,   L1Cfg::ELEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol,  L1Cfg::KCOL * L1Cfg::NCOL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW,    L1Cfg::OC * L1Cfg::KCOL     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut,  L1Cfg::OC * L1Cfg::NCOL     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSkip, L1Cfg::ELEMS * sizeof(float)));

    auto upload_vec = [](const std::vector<float>& v, float* d){
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice));
    };

    // BN 파라미터 (필요시 덮어씀)
    float *dG, *dB, *dM, *dV;
    CUDA_CHECK(cudaMalloc(&dG, L1Cfg::OC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, L1Cfg::OC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dM, L1Cfg::OC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, L1Cfg::OC*sizeof(float)));

    // 입력 업로드
    upload_vec(hX0, dIn);

    // ---- 4) Block0: (conv1+bn+relu) -> (conv2+bn) -> +identity -> relu ----
    Timer T; float t_im2col=0, t_gemm=0, t_bn=0, t_relu=0;

    // identity 보관
    CUDA_CHECK(cudaMemcpy(dSkip, dIn, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToDevice));

    // conv1
    upload_vec(W10col, dW);
    upload_vec(G10, dG); upload_vec(B10, dB); upload_vec(M10, dM); upload_vec(V10, dV);
    conv_bn_forward(dIn, dCol, dW, dOut, dG, dB, dM, dV, /*with_relu=*/true,
                    T, t_im2col, t_gemm, t_bn, t_relu);

    // conv2 (ReLU 없음)
    // 다음 입력은 방금 출력(dOut)
    CUDA_CHECK(cudaMemcpy(dIn, dOut, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToDevice));
    upload_vec(W11col, dW);
    upload_vec(G11, dG); upload_vec(B11, dB); upload_vec(M11, dM); upload_vec(V11, dV);
    conv_bn_forward(dIn, dCol, dW, dOut, dG, dB, dM, dV, /*with_relu=*/false,
                    T, t_im2col, t_gemm, t_bn, t_relu);

    // Add + ReLU
    add_inplace<<< (L1Cfg::ELEMS+255)/256, 256 >>>(dOut, dSkip, L1Cfg::ELEMS);
    CUDA_CHECK(cudaDeviceSynchronize());
    relu_forward<<< (L1Cfg::ELEMS+255)/256, 256 >>>(dOut, L1Cfg::ELEMS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // host로 복사 & 비교
    std::vector<float> yB0(L1Cfg::ELEMS);
    CUDA_CHECK(cudaMemcpy(yB0.data(), dOut, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs0=0.0, mean_abs0=0.0;
    for (size_t i=0;i<yB0.size();++i){
        double d = std::fabs((double)yB0[i] - (double)yE_b0[i]);
        max_abs0 = std::max(max_abs0, d);
        mean_abs0 += d;
    }
    mean_abs0 /= (double)yB0.size();

    std::cout << "[Layer1 Block0]\n";
    std::cout << "  im2col: " << t_im2col << " ms,  gemm: " << t_gemm
              << " ms,  bn: " << t_bn << " ms,  relu: " << t_relu << " ms\n";
    std::cout << "  diff  : max_abs=" << max_abs0 << " mean_abs=" << mean_abs0 << "\n";
    if (max_abs0 > 1e-4) { std::cerr << "  [FAIL] exceed atol 1e-4\n"; return 2; }

    // ---- 5) Block1: 입력은 Block0 출력, 동일한 구조 ----
    t_im2col=t_gemm=t_bn=t_relu=0.0f;

    // block1 identity = block0 출력
    CUDA_CHECK(cudaMemcpy(dIn,   dOut, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dSkip, dOut, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToDevice));

    // conv1
    upload_vec(W20col, dW);
    upload_vec(G20, dG); upload_vec(B20, dB); upload_vec(M20, dM); upload_vec(V20, dV);
    conv_bn_forward(dIn, dCol, dW, dOut, dG, dB, dM, dV, /*with_relu=*/true,
                    T, t_im2col, t_gemm, t_bn, t_relu);

    // conv2 (ReLU 없음)
    CUDA_CHECK(cudaMemcpy(dIn, dOut, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToDevice));
    upload_vec(W21col, dW);
    upload_vec(G21, dG); upload_vec(B21, dB); upload_vec(M21, dM); upload_vec(V21, dV);
    conv_bn_forward(dIn, dCol, dW, dOut, dG, dB, dM, dV, /*with_relu=*/false,
                    T, t_im2col, t_gemm, t_bn, t_relu);

    // Add + ReLU
    add_inplace<<< (L1Cfg::ELEMS+255)/256, 256 >>>(dOut, dSkip, L1Cfg::ELEMS);
    CUDA_CHECK(cudaDeviceSynchronize());
    relu_forward<<< (L1Cfg::ELEMS+255)/256, 256 >>>(dOut, L1Cfg::ELEMS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // host로 복사 & 비교
    std::vector<float> yB1(L1Cfg::ELEMS);
    CUDA_CHECK(cudaMemcpy(yB1.data(), dOut, L1Cfg::ELEMS*sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs1=0.0, mean_abs1=0.0;
    for (size_t i=0;i<yB1.size();++i){
        double d = std::fabs((double)yB1[i] - (double)yE_b1[i]);
        max_abs1 = std::max(max_abs1, d);
        mean_abs1 += d;
    }
    mean_abs1 /= (double)yB1.size();

    std::cout << "[Layer1 Block1]\n";
    std::cout << "  im2col: " << t_im2col << " ms,  gemm: " << t_gemm
              << " ms,  bn: " << t_bn << " ms,  relu: " << t_relu << " ms\n";
    std::cout << "  diff  : max_abs=" << max_abs1 << " mean_abs=" << mean_abs1 << "\n";
    if (max_abs1 > 1e-4) { std::cerr << "  [FAIL] exceed atol 1e-4\n"; return 3; }

    std::cout << "[OK] Step3 layer1 matched within atol 1e-4\n";

    // ---- 6) 자원 해제 ----
    cudaFree(dIn); cudaFree(dCol); cudaFree(dW); cudaFree(dOut); cudaFree(dSkip);
    cudaFree(dG);  cudaFree(dB);   cudaFree(dM); cudaFree(dV);
    return 0;
}