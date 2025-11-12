// cpp/fp32/runtime/infer_layer2.cu
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sys/stat.h>

#include "utils.hpp" // CUDA_CHECK, Timer, load_bin_f32 등 사용

// ===== 커널 선언 =====
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);

extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);

extern "C" __global__
void bn_inference(float*,const float*,const float*,const float*,const float*,float,int,int,int);

extern "C" __global__
void relu_forward(float*,int);

extern "C" __global__
void add_inplace(float* y, const float* x, int n); // elemwise add (y+=x)

#define CUDA_LAUNCH(kernel, grid, block, ...) \
do { \
  kernel<<<grid, block>>>(__VA_ARGS__); \
  cudaError_t __e = cudaGetLastError(); \
  if (__e != cudaSuccess){ \
    fprintf(stderr, "CUDA launch error %s @ %s:%d\n", cudaGetErrorString(__e), __FILE__, __LINE__); \
    return 3; \
  } \
  CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

static void make_Wcol_OIHW(const std::vector<float>& W_oi_hw,
                           int OC, int IC, int KH, int KW,
                           std::vector<float>& Wcol)
{
    const int KCOL = IC * KH * KW;
    Wcol.resize(OC * KCOL);
    for (int o=0;o<OC;++o){
        for (int c=0;c<IC;++c){
            for (int kh=0;kh<KH;++kh){
                for (int kw=0;kw<KW;++kw){
                    int r   = c*KH*KW + kh*KW + kw;                 // row in Wcol
                    int src = o*IC*KH*KW + c*KH*KW + kh*KW + kw;    // flat OIHW
                    Wcol[o*KCOL + r] = W_oi_hw[src];
                }
            }
        }
    }
}

static bool file_exists(const std::string& p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

static void print_diff(const char* tag,
                       const std::vector<float>& y,
                       const std::vector<float>& yE)
{
    double max_abs=0.0, mean_abs=0.0;
    size_t n = std::min(y.size(), yE.size());
    for (size_t i=0;i<n;++i){
        double d = std::fabs((double)y[i] - (double)yE[i]);
        max_abs = std::max(max_abs, d);
        mean_abs += d;
    }
    mean_abs /= std::max<size_t>(1,n);
    std::cout<<"  diff  : max_abs="<<max_abs<<" mean_abs="<<mean_abs<<"\n";
    if (max_abs <= 1e-4) std::cout<<"[OK] "<<tag<<" within atol 1e-4\n";
    else                 std::cout<<"[WARN] "<<tag<<" exceeded atol 1e-4\n";
}

// ===== 메인 =====
int main(int argc, char** argv)
{
    // 인자 파싱: --manifest <path>
    std::string mani;
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest" && i+1<argc) mani = argv[++i];
    }
    if (mani.empty()){
        std::cerr<<"Usage: step4_layer2 --manifest exports/resnet18/fp32\n";
        return 1;
    }

    // 입력/기대치 경로
    const std::string in_path   = mani + "/fixtures_step3/layer1_block1_out.bin";
    const std::string exp_b0    = mani + "/fixtures_step4/layer2_block0_out.bin";
    const std::string exp_b1    = mani + "/fixtures_step4/layer2_block1_out.bin";

    // ===== 입력 로드 =====
    // layer1 출력 = (C=64, H=56, W=56)
    const int C_in=64, H_in=56, W_in=56;
    std::vector<float> x = load_bin_f32(in_path, C_in*H_in*W_in);

    // ===== 가중치 로드 (layer2) =====
    // layer2.0
    auto W20_1 = load_bin_f32(mani+"/layer2.0.conv1.weight.bin", 128*64*3*3);
    auto B20_1_w = load_bin_f32(mani+"/layer2.0.bn1.weight.bin", 128);
    auto B20_1_b = load_bin_f32(mani+"/layer2.0.bn1.bias.bin",   128);
    auto B20_1_m = load_bin_f32(mani+"/layer2.0.bn1.running_mean.bin", 128);
    auto B20_1_v = load_bin_f32(mani+"/layer2.0.bn1.running_var.bin",  128);

    auto W20_2 = load_bin_f32(mani+"/layer2.0.conv2.weight.bin", 128*128*3*3);
    auto B20_2_w = load_bin_f32(mani+"/layer2.0.bn2.weight.bin", 128);
    auto B20_2_b = load_bin_f32(mani+"/layer2.0.bn2.bias.bin",   128);
    auto B20_2_m = load_bin_f32(mani+"/layer2.0.bn2.running_mean.bin", 128);
    auto B20_2_v = load_bin_f32(mani+"/layer2.0.bn2.running_var.bin",  128);

    // downsample (1x1, s=2)
    auto W20_ds = load_bin_f32(mani+"/layer2.0.downsample.0.weight.bin", 128*64*1*1);
    auto B20_ds_w = load_bin_f32(mani+"/layer2.0.downsample.1.weight.bin", 128);
    auto B20_ds_b = load_bin_f32(mani+"/layer2.0.downsample.1.bias.bin",   128);
    auto B20_ds_m = load_bin_f32(mani+"/layer2.0.downsample.1.running_mean.bin", 128);
    auto B20_ds_v = load_bin_f32(mani+"/layer2.0.downsample.1.running_var.bin",  128);

    // layer2.1
    auto W21_1 = load_bin_f32(mani+"/layer2.1.conv1.weight.bin", 128*128*3*3);
    auto B21_1_w = load_bin_f32(mani+"/layer2.1.bn1.weight.bin", 128);
    auto B21_1_b = load_bin_f32(mani+"/layer2.1.bn1.bias.bin",   128);
    auto B21_1_m = load_bin_f32(mani+"/layer2.1.bn1.running_mean.bin", 128);
    auto B21_1_v = load_bin_f32(mani+"/layer2.1.bn1.running_var.bin",  128);

    auto W21_2 = load_bin_f32(mani+"/layer2.1.conv2.weight.bin", 128*128*3*3);
    auto B21_2_w = load_bin_f32(mani+"/layer2.1.bn2.weight.bin", 128);
    auto B21_2_b = load_bin_f32(mani+"/layer2.1.bn2.bias.bin",   128);
    auto B21_2_m = load_bin_f32(mani+"/layer2.1.bn2.running_mean.bin", 128);
    auto B21_2_v = load_bin_f32(mani+"/layer2.1.bn2.running_var.bin",  128);

    // ===== Wcol 준비 =====
    std::vector<float> W20_1_col, W20_2_col, W20_ds_col;
    std::vector<float> W21_1_col, W21_2_col;
    make_Wcol_OIHW(W20_1, 128,  64, 3,3, W20_1_col);
    make_Wcol_OIHW(W20_2, 128, 128, 3,3, W20_2_col);
    make_Wcol_OIHW(W20_ds,128,  64, 1,1, W20_ds_col);
    make_Wcol_OIHW(W21_1, 128, 128, 3,3, W21_1_col);
    make_Wcol_OIHW(W21_2, 128, 128, 3,3, W21_2_col);

    // ===== 디바이스 메모리 =====
    // 최대 필요량을 고려해 버퍼 재사용
    float *dX=nullptr, *dCol=nullptr, *dY=nullptr, *dTmp=nullptr;
    float *dW1=nullptr, *dW2=nullptr; // 가변 바인딩용
    float *dBn_w=nullptr,*dBn_b=nullptr,*dBn_m=nullptr,*dBn_v=nullptr;

    // 입력 업로드
    CUDA_CHECK(cudaMalloc(&dX, x.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));

    // layer2에서의 최대 KCOL, OC, NCOL
    const int KCOL_max = max3(128*3*3, 64*3*3, 64*1*1); // 1152, 576, 64 => 1152
    const int OC_max   = 128;
    const int NCOL_max = 28*28; // stride=2 후 28x28

    CUDA_CHECK(cudaMalloc(&dCol, KCOL_max * NCOL_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY,   OC_max   * NCOL_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTmp, OC_max   * NCOL_max * sizeof(float))); // add/임시 목적

    CUDA_CHECK(cudaMalloc(&dW1, OC_max * KCOL_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, OC_max * KCOL_max * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dBn_w, OC_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_b, OC_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_m, OC_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_v, OC_max*sizeof(float)));

    Timer T;
    float ms_im2, ms_gemm, ms_bn, ms_relu;

    // ===== Block0 =====
    // conv1: in(64,56,56)->out(128,28,28), 3x3 s=2 p=1
    {
        const int IC=64, OC=128, KH=3,KW=3, SH=2,SW=2, PH=1,PW=1;
        const int OH = (H_in + 2*PH - KH)/SH + 1; // 28
        const int OW = (W_in + 2*PW - KW)/SW + 1; // 28
        const int KCOL = IC*KH*KW;                 // 576
        const int NCOL = OH*OW;                    // 784

        // 업로드: Wcol, BN1
        CUDA_CHECK(cudaMemcpy(dW1, W20_1_col.data(), OC*KCOL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_w, B20_1_w.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_b, B20_1_b.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_m, B20_1_m.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_v, B20_1_v.data(), OC*sizeof(float), cudaMemcpyHostToDevice));

        dim3 blk_im2(16,16), grd_im2(div_up(OW,blk_im2.x), div_up(OH,blk_im2.y));
        T.start();
        CUDA_LAUNCH(im2col_nchw, grd_im2, blk_im2, dX, 1, IC, H_in, W_in, KH,KW, SH,SW, PH,PW, dCol);
        ms_im2 = T.stop();

        // GEMM: (OC x NCOL) = (OC x KCOL) * (KCOL x NCOL)
        dim3 blk_g(32,32), grd_g(div_up(NCOL,blk_g.x), div_up(OC,blk_g.y));
        T.start();
        CUDA_LAUNCH(sgemm_tiled, grd_g, blk_g, dW1, dCol, dY, OC, NCOL, KCOL);
        ms_gemm = T.stop();

        // BN + ReLU
        int n = OC*OH*OW;
        T.start();
        CUDA_LAUNCH(bn_inference, dim3(div_up(n,256)), dim3(256), dY, dBn_w, dBn_b, dBn_m, dBn_v, 1e-5f, OC, OH, OW);
        ms_bn = T.stop();

        T.start();
        CUDA_LAUNCH(relu_forward, dim3(div_up(n,256)), dim3(256), dY, n);
        ms_relu = T.stop();
    }

    // conv2: in(128,28,28)->out(128,28,28), 3x3 s=1 p=1  (ReLU는 residual add 이후)
    {
        const int IC=128, OC=128, KH=3,KW=3, SH=1,SW=1, PH=1,PW=1;
        const int OH = 28, OW = 28;
        const int KCOL = IC*KH*KW; // 1152
        const int NCOL = OH*OW;    // 784

        // 업로드: Wcol, BN2
        CUDA_CHECK(cudaMemcpy(dW2, W20_2_col.data(), OC*KCOL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_w, B20_2_w.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_b, B20_2_b.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_m, B20_2_m.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_v, B20_2_v.data(), OC*sizeof(float), cudaMemcpyHostToDevice));

        // im2col (입력은 dY: 이전 conv1 결과)
        dim3 blk_im2(16,16), grd_im2(div_up(OW,blk_im2.x), div_up(OH,blk_im2.y));
        T.start();
        CUDA_LAUNCH(im2col_nchw, grd_im2, blk_im2, dY, 1, IC, 28, 28, KH,KW, SH,SW, PH,PW, dCol);
        ms_im2 += T.stop();

        // gemm
        dim3 blk_g(32,32), grd_g(div_up(NCOL,blk_g.x), div_up(OC,blk_g.y));
        T.start();
        CUDA_LAUNCH(sgemm_tiled, grd_g, blk_g, dW2, dCol, dTmp, OC, NCOL, KCOL); // conv2 출력은 dTmp
        ms_gemm += T.stop();

        // bn2
        int n = OC*OH*OW;
        T.start();
        CUDA_LAUNCH(bn_inference, dim3(div_up(n,256)), dim3(256), dTmp, dBn_w, dBn_b, dBn_m, dBn_v, 1e-5f, OC, OH, OW);
        ms_bn += T.stop();
    }

    // shortcut (1x1, s=2) + BN
    {
        const int IC=64, OC=128, KH=1,KW=1, SH=2,SW=2, PH=0,PW=0;
        const int OH = 28, OW = 28;
        const int KCOL = IC*KH*KW; // 64
        const int NCOL = OH*OW;    // 784

        // Wcol, BN(downsample) 업로드
        CUDA_CHECK(cudaMemcpy(dW1, W20_ds_col.data(), OC*KCOL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_w, B20_ds_w.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_b, B20_ds_b.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_m, B20_ds_m.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_v, B20_ds_v.data(), OC*sizeof(float), cudaMemcpyHostToDevice));

        // im2col (입력은 원본 dX: (64,56,56))
        dim3 blk_im2(16,16), grd_im2(div_up(OW,blk_im2.x), div_up(OH,blk_im2.y));
        T.start();
        CUDA_LAUNCH(im2col_nchw, grd_im2, blk_im2, dX, 1, IC, 56, 56, KH,KW, SH,SW, PH,PW, dCol);
        ms_im2 += T.stop();

        // gemm: dW1 * dCol -> dY (shortcut 결과를 dY에 저장)
        dim3 blk_g(32,32), grd_g(div_up(NCOL,blk_g.x), div_up(OC,blk_g.y));
        T.start();
        CUDA_LAUNCH(sgemm_tiled, grd_g, blk_g, dW1, dCol, dY, OC, NCOL, KCOL);
        ms_gemm += T.stop();

        // BN
        int n = OC*OH*OW;
        T.start();
        CUDA_LAUNCH(bn_inference, dim3(div_up(n,256)), dim3(256), dY, dBn_w, dBn_b, dBn_m, dBn_v, 1e-5f, OC, OH, OW);
        ms_bn += T.stop();
    }

    // residual add: dTmp(conv2+bn2) + dY(shortcut+bn) -> dTmp에 축적, ReLU(dTmp)
    {
        const int OC=128, H=28, W=28;
        const int n = OC*H*W;
        T.start();
        CUDA_LAUNCH(add_inplace, dim3(div_up(n,256)), dim3(256), dTmp, dY, n);
        CUDA_LAUNCH(relu_forward, dim3(div_up(n,256)), dim3(256), dTmp, n);
        ms_relu += T.stop();
    }

    // Block0 결과를 host로
    std::vector<float> out_b0(128*28*28);
    CUDA_CHECK(cudaMemcpy(out_b0.data(), dTmp, out_b0.size()*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout<<"[Layer2 Block0]\n";
    std::cout<<"  im2col: "<<ms_im2<<" ms,  gemm: "<<ms_gemm<<" ms,  bn: "<<ms_bn<<" ms,  relu: "<<ms_relu<<" ms\n";
    if (file_exists(exp_b0)) {
        auto yE0 = load_bin_f32(exp_b0, out_b0.size());
        print_diff("Block0", out_b0, yE0);
    }

    // ===== Block1 ===== (입력: Block0 출력 out_b0, C=128,H=28,W=28, downsample 없음)
    CUDA_CHECK(cudaMemcpy(dX, out_b0.data(), out_b0.size()*sizeof(float), cudaMemcpyHostToDevice));
    ms_im2=ms_gemm=ms_bn=ms_relu=0.0f;

    // conv1: 3x3 s=1 p=1
    {
        const int IC=128, OC=128, KH=3,KW=3, SH=1,SW=1, PH=1,PW=1;
        const int OH=28, OW=28;
        const int KCOL=IC*KH*KW; // 1152
        const int NCOL=OH*OW;    // 784

        CUDA_CHECK(cudaMemcpy(dW1, W21_1_col.data(), OC*KCOL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_w, B21_1_w.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_b, B21_1_b.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_m, B21_1_m.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_v, B21_1_v.data(), OC*sizeof(float), cudaMemcpyHostToDevice));

        dim3 blk_im2(16,16), grd_im2(div_up(OW,blk_im2.x), div_up(OH,blk_im2.y));
        T.start();
        CUDA_LAUNCH(im2col_nchw, grd_im2, blk_im2, dX, 1, IC, 28, 28, KH,KW, SH,SW, PH,PW, dCol);
        ms_im2 += T.stop();

        dim3 blk_g(32,32), grd_g(div_up(NCOL,blk_g.x), div_up(OC,blk_g.y));
        T.start();
        CUDA_LAUNCH(sgemm_tiled, grd_g, blk_g, dW1, dCol, dY, OC, NCOL, KCOL);
        ms_gemm += T.stop();

        int n = OC*OH*OW;
        T.start();
        CUDA_LAUNCH(bn_inference, dim3(div_up(n,256)), dim3(256), dY, dBn_w, dBn_b, dBn_m, dBn_v, 1e-5f, OC, OH, OW);
        ms_bn += T.stop();

        T.start();
        CUDA_LAUNCH(relu_forward, dim3(div_up(n,256)), dim3(256), dY, n);
        ms_relu += T.stop();
    }

    // conv2: 3x3 s=1 p=1 (ReLU는 residual add 이후)
    {
        const int IC=128, OC=128, KH=3,KW=3, SH=1,SW=1, PH=1,PW=1;
        const int OH=28, OW=28;
        const int KCOL=IC*KH*KW; // 1152
        const int NCOL=OH*OW;    // 784

        CUDA_CHECK(cudaMemcpy(dW2, W21_2_col.data(), OC*KCOL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_w, B21_2_w.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_b, B21_2_b.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_m, B21_2_m.data(), OC*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBn_v, B21_2_v.data(), OC*sizeof(float), cudaMemcpyHostToDevice));

        dim3 blk_im2(16,16), grd_im2(div_up(OW,blk_im2.x), div_up(OH,blk_im2.y));
        T.start();
        CUDA_LAUNCH(im2col_nchw, grd_im2, blk_im2, dY, 1, IC, 28, 28, KH,KW, SH,SW, PH,PW, dCol);
        ms_im2 += T.stop();

        dim3 blk_g(32,32), grd_g(div_up(NCOL,blk_g.x), div_up(OC,blk_g.y));
        T.start();
        CUDA_LAUNCH(sgemm_tiled, grd_g, blk_g, dW2, dCol, dTmp, OC, NCOL, KCOL);
        ms_gemm += T.stop();

        int n = OC*OH*OW;
        T.start();
        CUDA_LAUNCH(bn_inference, dim3(div_up(n,256)), dim3(256), dTmp, dBn_w, dBn_b, dBn_m, dBn_v, 1e-5f, OC, OH, OW);
        ms_bn += T.stop();
    }

    // residual add (identity) + ReLU  (입력 dX == shortcut)
    {
        const int OC=128, H=28, W=28;
        const int n = OC*H*W;
        T.start();
        CUDA_LAUNCH(add_inplace, dim3(div_up(n,256)), dim3(256), dTmp, dX, n);
        CUDA_LAUNCH(relu_forward, dim3(div_up(n,256)), dim3(256), dTmp, n);
        ms_relu += T.stop();
    }

    std::vector<float> out_b1(128*28*28);
    CUDA_CHECK(cudaMemcpy(out_b1.data(), dTmp, out_b1.size()*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout<<"[Layer2 Block1]\n";
    std::cout<<"  im2col: "<<ms_im2<<" ms,  gemm: "<<ms_gemm<<" ms,  bn: "<<ms_bn<<" ms,  relu: "<<ms_relu<<" ms\n";
    if (file_exists(exp_b1)) {
        auto yE1 = load_bin_f32(exp_b1, out_b1.size());
        print_diff("Block1", out_b1, yE1);
    }

    // clean
    cudaFree(dX); cudaFree(dCol); cudaFree(dY); cudaFree(dTmp);
    cudaFree(dW1); cudaFree(dW2);
    cudaFree(dBn_w); cudaFree(dBn_b); cudaFree(dBn_m); cudaFree(dBn_v);

    return 0;
}
