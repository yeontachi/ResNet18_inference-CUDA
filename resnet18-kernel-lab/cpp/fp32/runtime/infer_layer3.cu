// cpp/fp32/runtime/infer_layer3.cu
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "utils.hpp"

// =====(필요시) 호스트 복사 헬퍼=====
// utils.hpp에 copy_to_host가 없다면 아래를 사용.
// 이미 있다면 이 블록 삭제해도 됨.
static inline std::vector<float> copy_to_host_fallback(const float* dptr, size_t n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), dptr, n*sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}
#define COPY_TO_HOST(ptr, n) copy_to_host_fallback((ptr), (n))

// ===== kernels =====
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);

extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);

extern "C" __global__
void bn_inference(float*,const float*,const float*,const float*,const float*,float,int,int,int);

extern "C" __global__
void relu_forward(float*,int);

extern "C" __global__
void add_inplace(float* __restrict__ y, const float* __restrict__ x, int n);

// ===== layer3 shapes (ResNet18) =====
struct L3Shape {
    static constexpr int N = 1;

    // in from layer2
    static constexpr int C_in  = 128;
    static constexpr int H_in  = 28;
    static constexpr int W_in  = 28;

    // block0 conv1
    static constexpr int OC0_1 = 256, KH0_1 = 3, KW0_1 = 3, SH0_1 = 2, SW0_1 = 2, PH0_1 = 1, PW0_1 = 1;
    static constexpr int OH0_1 = (H_in + 2*PH0_1 - KH0_1)/SH0_1 + 1;
    static constexpr int OW0_1 = (W_in + 2*PW0_1 - KW0_1)/SW0_1 + 1;
    static constexpr int KCOL0_1 = C_in * KH0_1 * KW0_1;
    static constexpr int NCOL0_1 = OH0_1 * OW0_1;

    // block0 conv2
    static constexpr int C0_2 = 256, OC0_2 = 256, KH0_2 = 3, KW0_2 = 3, SH0_2 = 1, SW0_2 = 1, PH0_2 = 1, PW0_2 = 1;
    static constexpr int OH0_2 = OH0_1, OW0_2 = OW0_1;
    static constexpr int KCOL0_2 = C0_2 * KH0_2 * KW0_2;
    static constexpr int NCOL0_2 = OH0_2 * OW0_2;

    // block0 downsample (1x1 s=2)
    static constexpr int DS0_OC = 256, DS0_KH = 1, DS0_KW = 1, DS0_SH = 2, DS0_SW = 2, DS0_PH = 0, DS0_PW = 0;
    static constexpr int DS0_OH = OH0_1, DS0_OW = OW0_1;
    static constexpr int DS0_KCOL = C_in * DS0_KH * DS0_KW; // 128
    static constexpr int DS0_NCOL = DS0_OH * DS0_OW;

    // block1 conv1
    static constexpr int C1_1 = 256, OC1_1 = 256, KH1_1 = 3, KW1_1 = 3, SH1_1 = 1, SW1_1 = 1, PH1_1 = 1, PW1_1 = 1;
    static constexpr int OH1_1 = OH0_2, OW1_1 = OW0_2;
    static constexpr int KCOL1_1 = C1_1 * KH1_1 * KW1_1;
    static constexpr int NCOL1_1 = OH1_1 * OW1_1;

    // block1 conv2
    static constexpr int C1_2 = 256, OC1_2 = 256, KH1_2 = 3, KW1_2 = 3, SH1_2 = 1, SW1_2 = 1, PH1_2 = 1, PW1_2 = 1;
    static constexpr int OH1_2 = OH1_1, OW1_2 = OW1_1;
    static constexpr int KCOL1_2 = C1_2 * KH1_2 * KW1_2;
    static constexpr int NCOL1_2 = OH1_2 * OW1_2;

    static constexpr int OUT_ELEMS = OC0_2 * OH0_2 * OW0_2; // 256*14*14
};

// weights reshape: OIHW -> (O, I*KH*KW)
static void make_Wcol(const std::vector<float>& W_oi_hw, int O, int I, int KH, int KW, std::vector<float>& Wcol) {
    const int KCOL = I*KH*KW;
    Wcol.assign(O*KCOL, 0.f);
    for (int o=0;o<O;o++){
        for (int c=0;c<I;c++){
            for (int kh=0;kh<KH;kh++){
                for (int kw=0;kw<KW;kw++){
                    int r = c*KH*KW + kh*KW + kw;
                    int src = o*I*KH*KW + c*KH*KW + kh*KW + kw;
                    Wcol[o*KCOL + r] = W_oi_hw[src];
                }
            }
        }
    }
}

int main(int argc, char** argv){
    CmdArgs args = parse_args(argc, argv);
    if (args.manifest.empty()){
        std::cerr<<"Usage: step5_layer3 --manifest exports/resnet18/fp32\n";
        return 1;
    }
    Manifest mani(args.manifest);

    // ===== inputs / expected =====
    const std::string FIX = args.manifest + "/fixtures_step5";
    auto Xin   = load_bin_f32(FIX + "/layer2_out.bin",            L3Shape::C_in * L3Shape::H_in * L3Shape::W_in);
    auto Yb0E  = load_bin_f32(FIX + "/layer3_block0_out.bin",     L3Shape::OUT_ELEMS);
    auto Yb1E  = load_bin_f32(FIX + "/layer3_block1_out.bin",     L3Shape::OUT_ELEMS);

    // ===== weights =====
    // block0
    auto W01 = mani.load("layer3.0.conv1.weight",   L3Shape::OC0_1 * L3Shape::KCOL0_1).buf;
    auto G01 = mani.load("layer3.0.bn1.weight",     L3Shape::OC0_1).buf;
    auto B01 = mani.load("layer3.0.bn1.bias",       L3Shape::OC0_1).buf;
    auto M01 = mani.load("layer3.0.bn1.running_mean", L3Shape::OC0_1).buf;
    auto V01 = mani.load("layer3.0.bn1.running_var",  L3Shape::OC0_1).buf;

    auto W02 = mani.load("layer3.0.conv2.weight",   L3Shape::OC0_2 * L3Shape::KCOL0_2).buf;
    auto G02 = mani.load("layer3.0.bn2.weight",     L3Shape::OC0_2).buf;
    auto B02 = mani.load("layer3.0.bn2.bias",       L3Shape::OC0_2).buf;
    auto M02 = mani.load("layer3.0.bn2.running_mean", L3Shape::OC0_2).buf;
    auto V02 = mani.load("layer3.0.bn2.running_var",  L3Shape::OC0_2).buf;

    auto W0d = mani.load("layer3.0.downsample.0.weight", L3Shape::DS0_OC * L3Shape::DS0_KCOL).buf;
    auto G0d = mani.load("layer3.0.downsample.1.weight", L3Shape::DS0_OC).buf;
    auto B0d = mani.load("layer3.0.downsample.1.bias",   L3Shape::DS0_OC).buf;
    auto M0d = mani.load("layer3.0.downsample.1.running_mean", L3Shape::DS0_OC).buf;
    auto V0d = mani.load("layer3.0.downsample.1.running_var",  L3Shape::DS0_OC).buf;

    // block1
    auto W11 = mani.load("layer3.1.conv1.weight",   L3Shape::OC1_1 * L3Shape::KCOL1_1).buf;
    auto G11 = mani.load("layer3.1.bn1.weight",     L3Shape::OC1_1).buf;
    auto B11 = mani.load("layer3.1.bn1.bias",       L3Shape::OC1_1).buf;
    auto M11 = mani.load("layer3.1.bn1.running_mean", L3Shape::OC1_1).buf;
    auto V11 = mani.load("layer3.1.bn1.running_var",  L3Shape::OC1_1).buf;

    auto W12 = mani.load("layer3.1.conv2.weight",   L3Shape::OC1_2 * L3Shape::KCOL1_2).buf;
    auto G12 = mani.load("layer3.1.bn2.weight",     L3Shape::OC1_2).buf;
    auto B12 = mani.load("layer3.1.bn2.bias",       L3Shape::OC1_2).buf;
    auto M12 = mani.load("layer3.1.bn2.running_mean", L3Shape::OC1_2).buf;
    auto V12 = mani.load("layer3.1.bn2.running_var",  L3Shape::OC1_2).buf;

    // ===== W_col =====
    std::vector<float> W01col, W02col, W0dcol, W11col, W12col;
    make_Wcol(W01, L3Shape::OC0_1, L3Shape::C_in,   L3Shape::KH0_1, L3Shape::KW0_1, W01col);
    make_Wcol(W02, L3Shape::OC0_2, L3Shape::C0_2,   L3Shape::KH0_2, L3Shape::KW0_2, W02col);
    make_Wcol(W0d, L3Shape::DS0_OC, L3Shape::C_in,  L3Shape::DS0_KH, L3Shape::DS0_KW, W0dcol);
    make_Wcol(W11, L3Shape::OC1_1, L3Shape::C1_1,   L3Shape::KH1_1, L3Shape::KW1_1, W11col);
    make_Wcol(W12, L3Shape::OC1_2, L3Shape::C1_2,   L3Shape::KH1_2, L3Shape::KW1_2, W12col);

    // ===== device buffers =====
    float *dX;  CUDA_CHECK(cudaMalloc(&dX,  Xin.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, Xin.data(), Xin.size()*sizeof(float), cudaMemcpyHostToDevice));

    float *dCol01, *dCol02, *dCol0d, *dCol11, *dCol12;
    CUDA_CHECK(cudaMalloc(&dCol01, L3Shape::KCOL0_1 * L3Shape::NCOL0_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol02, L3Shape::KCOL0_2 * L3Shape::NCOL0_2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol0d, L3Shape::DS0_KCOL * L3Shape::DS0_NCOL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol11, L3Shape::KCOL1_1 * L3Shape::NCOL1_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol12, L3Shape::KCOL1_2 * L3Shape::NCOL1_2 * sizeof(float)));

    // unique_ptr (DeviceDeleter) → .get()로 커널에 전달
    auto dW01 = copy_to_device(W01col);
    auto dW02 = copy_to_device(W02col);
    auto dW0d = copy_to_device(W0dcol);
    auto dW11 = copy_to_device(W11col);
    auto dW12 = copy_to_device(W12col);

    auto dG01 = copy_to_device(G01), dB01 = copy_to_device(B01), dM01 = copy_to_device(M01), dV01 = copy_to_device(V01);
    auto dG02 = copy_to_device(G02), dB02 = copy_to_device(B02), dM02 = copy_to_device(M02), dV02 = copy_to_device(V02);
    auto dG0d = copy_to_device(G0d), dB0d = copy_to_device(B0d), dM0d = copy_to_device(M0d), dV0d = copy_to_device(V0d);
    auto dG11 = copy_to_device(G11), dB11 = copy_to_device(B11), dM11 = copy_to_device(M11), dV11 = copy_to_device(V11);
    auto dG12 = copy_to_device(G12), dB12 = copy_to_device(B12), dM12 = copy_to_device(M12), dV12 = copy_to_device(V12);

    float *dY0_1, *dY0_2, *dY0d, *dY1_1, *dY1_2;
    CUDA_CHECK(cudaMalloc(&dY0_1, L3Shape::OC0_1 * L3Shape::NCOL0_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY0_2, L3Shape::OC0_2 * L3Shape::NCOL0_2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY0d,  L3Shape::DS0_OC * L3Shape::DS0_NCOL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY1_1, L3Shape::OC1_1 * L3Shape::NCOL1_1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY1_2, L3Shape::OC1_2 * L3Shape::NCOL1_2 * sizeof(float)));

    Timer T; float ms_im2col=0.f, ms_gemm=0.f, ms_bn=0.f, ms_relu=0.f;

    // ===== Block0 =====
    std::cout<<"[Layer3 Block0]\n";

    // conv1
    {
        dim3 blk(16,16);
        dim3 grd((L3Shape::OW0_1+15)/16, (L3Shape::OH0_1+15)/16);
        T.start();
        im2col_nchw<<<grd,blk>>>(dX, L3Shape::N, L3Shape::C_in, L3Shape::H_in, L3Shape::W_in,
                                 L3Shape::KH0_1, L3Shape::KW0_1, L3Shape::SH0_1, L3Shape::SW0_1,
                                 L3Shape::PH0_1, L3Shape::PW0_1, dCol01);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col = T.stop();

        dim3 blk2(32,32);
        dim3 grd2((L3Shape::NCOL0_1+31)/32, (L3Shape::OC0_1+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW01.get(), dCol01, dY0_1, L3Shape::OC0_1, L3Shape::NCOL0_1, L3Shape::KCOL0_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm = T.stop();

        T.start();
        bn_inference<<< div_up(L3Shape::OC0_1*L3Shape::OH0_1*L3Shape::OW0_1,256), 256 >>>(
            dY0_1, dG01.get(), dB01.get(), dM01.get(), dV01.get(), 1e-5f,
            L3Shape::OC0_1, L3Shape::OH0_1, L3Shape::OW0_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn = T.stop();

        T.start();
        relu_forward<<< div_up(L3Shape::OC0_1*L3Shape::OH0_1*L3Shape::OW0_1,256), 256 >>>(
            dY0_1, L3Shape::OC0_1*L3Shape::OH0_1*L3Shape::OW0_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu = T.stop();
    }

    // conv2
    {
        dim3 blk(16,16);
        dim3 grd((L3Shape::OW0_2+15)/16, (L3Shape::OH0_2+15)/16);
        T.start();
        im2col_nchw<<<grd,blk>>>(dY0_1, L3Shape::N, L3Shape::C0_2, L3Shape::OH0_1, L3Shape::OW0_1,
                                 L3Shape::KH0_2, L3Shape::KW0_2, L3Shape::SH0_2, L3Shape::SW0_2,
                                 L3Shape::PH0_2, L3Shape::PW0_2, dCol02);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 blk2(32,32);
        dim3 grd2((L3Shape::NCOL0_2+31)/32, (L3Shape::OC0_2+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW02.get(), dCol02, dY0_2, L3Shape::OC0_2, L3Shape::NCOL0_2, L3Shape::KCOL0_2);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        T.start();
        bn_inference<<< div_up(L3Shape::OC0_2*L3Shape::OH0_2*L3Shape::OW0_2,256), 256 >>>(
            dY0_2, dG02.get(), dB02.get(), dM02.get(), dV02.get(), 1e-5f,
            L3Shape::OC0_2, L3Shape::OH0_2, L3Shape::OW0_2);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();
    }

    // downsample(X)
    {
        dim3 blk(16,16);
        dim3 grd((L3Shape::DS0_OW+15)/16, (L3Shape::DS0_OH+15)/16);
        T.start();
        im2col_nchw<<<grd,blk>>>(dX, L3Shape::N, L3Shape::C_in, L3Shape::H_in, L3Shape::W_in,
                                 L3Shape::DS0_KH, L3Shape::DS0_KW, L3Shape::DS0_SH, L3Shape::DS0_SW,
                                 L3Shape::DS0_PH, L3Shape::DS0_PW, dCol0d);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 blk2(32,32);
        dim3 grd2((L3Shape::DS0_NCOL+31)/32, (L3Shape::DS0_OC+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW0d.get(), dCol0d, dY0d, L3Shape::DS0_OC, L3Shape::DS0_NCOL, L3Shape::DS0_KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        T.start();
        bn_inference<<< div_up(L3Shape::DS0_OC*L3Shape::DS0_OH*L3Shape::DS0_OW,256), 256 >>>(
            dY0d, dG0d.get(), dB0d.get(), dM0d.get(), dV0d.get(), 1e-5f,
            L3Shape::DS0_OC, L3Shape::DS0_OH, L3Shape::DS0_OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();
    }

    // residual add + ReLU
    T.start();
    add_inplace<<< div_up(L3Shape::OUT_ELEMS,256), 256 >>>(dY0_2, dY0d, L3Shape::OUT_ELEMS);
    relu_forward<<< div_up(L3Shape::OUT_ELEMS,256), 256 >>>(dY0_2, L3Shape::OUT_ELEMS); // <-- 오타 수정(dY0__2 -> dY0_2)
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_relu += T.stop();

    // compare block0
    {
        auto Y0 = COPY_TO_HOST(dY0_2, L3Shape::OUT_ELEMS);
        auto [mx, mn] = diff_max_mean(Y0, Yb0E);
        std::cout<<"  im2col: "<<ms_im2col<<" ms,  gemm: "<<ms_gemm<<" ms,  bn: "<<ms_bn<<" ms,  relu: "<<ms_relu<<" ms\n";
        std::cout<<"  diff  : max_abs="<<mx<<" mean_abs="<<mn<<"\n";
        if (mx <= 1e-4) std::cout<<"[OK] Block0 within atol 1e-4\n";
        else            std::cout<<"[WARN] Block0 exceed atol\n";
    }

    // ===== Block1 (identity) =====
    std::cout<<"[Layer3 Block1]\n";
    ms_im2col = ms_gemm = ms_bn = ms_relu = 0.f;

    // identity = block0 output
    float* dId = nullptr;
    CUDA_CHECK(cudaMalloc(&dId, L3Shape::OUT_ELEMS*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dId, dY0_2, L3Shape::OUT_ELEMS*sizeof(float), cudaMemcpyDeviceToDevice));

    // conv1
    {
        dim3 blk(16,16);
        dim3 grd((L3Shape::OW1_1+15)/16, (L3Shape::OH1_1+15)/16);
        T.start();
        im2col_nchw<<<grd,blk>>>(dY0_2, L3Shape::N, L3Shape::C1_1, L3Shape::OH0_2, L3Shape::OW0_2,
                                 L3Shape::KH1_1, L3Shape::KW1_1, L3Shape::SH1_1, L3Shape::SW1_1,
                                 L3Shape::PH1_1, L3Shape::PW1_1, dCol11);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col = T.stop();

        dim3 blk2(32,32);
        dim3 grd2((L3Shape::NCOL1_1+31)/32, (L3Shape::OC1_1+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW11.get(), dCol11, dY1_1, L3Shape::OC1_1, L3Shape::NCOL1_1, L3Shape::KCOL1_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm = T.stop();

        T.start();
        bn_inference<<< div_up(L3Shape::OC1_1*L3Shape::OH1_1*L3Shape::OW1_1,256), 256 >>>(
            dY1_1, dG11.get(), dB11.get(), dM11.get(), dV11.get(), 1e-5f,
            L3Shape::OC1_1, L3Shape::OH1_1, L3Shape::OW1_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn = T.stop();

        T.start();
        relu_forward<<< div_up(L3Shape::OC1_1*L3Shape::OH1_1*L3Shape::OW1_1,256), 256 >>>(
            dY1_1, L3Shape::OC1_1*L3Shape::OH1_1*L3Shape::OW1_1);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu = T.stop();
    }

    // conv2
    {
        dim3 blk(16,16);
        dim3 grd((L3Shape::OW1_2+15)/16, (L3Shape::OH1_2+15)/16);
        T.start();
        im2col_nchw<<<grd,blk>>>(dY1_1, L3Shape::N, L3Shape::C1_2, L3Shape::OH1_1, L3Shape::OW1_1,
                                 L3Shape::KH1_2, L3Shape::KW1_2, L3Shape::SH1_2, L3Shape::SW1_2,
                                 L3Shape::PH1_2, L3Shape::PW1_2, dCol12);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 blk2(32,32);
        dim3 grd2((L3Shape::NCOL1_2+31)/32, (L3Shape::OC1_2+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW12.get(), dCol12, dY1_2, L3Shape::OC1_2, L3Shape::NCOL1_2, L3Shape::KCOL1_2);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        T.start();
        bn_inference<<< div_up(L3Shape::OC1_2*L3Shape::OH1_2*L3Shape::OW1_2,256), 256 >>>(
            dY1_2, dG12.get(), dB12.get(), dM12.get(), dV12.get(), 1e-5f,
            L3Shape::OC1_2, L3Shape::OH1_2, L3Shape::OW1_2);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();
    }

    // residual add + ReLU (identity = dId)
    T.start();
    add_inplace<<< div_up(L3Shape::OUT_ELEMS,256), 256 >>>(dY1_2, dId, L3Shape::OUT_ELEMS);
    relu_forward<<< div_up(L3Shape::OUT_ELEMS,256), 256 >>>(dY1_2, L3Shape::OUT_ELEMS);
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_relu += T.stop();

    // compare block1
    {
        auto Y1 = COPY_TO_HOST(dY1_2, L3Shape::OUT_ELEMS);
        auto [mx, mn] = diff_max_mean(Y1, Yb1E);
        std::cout<<"  im2col: "<<ms_im2col<<" ms,  gemm: "<<ms_gemm<<" ms,  bn: "<<ms_bn<<" ms,  relu: "<<ms_relu<<" ms\n";
        std::cout<<"  diff  : max_abs="<<mx<<" mean_abs="<<mn<<"\n";
        if (mx <= 1e-4) std::cout<<"[OK] Step5 layer3 matched within atol 1e-4\n";
        else            std::cout<<"[FAIL] Step5 diff exceeded\n";
    }

    // free (unique_ptr는 자동 해제, cudaMalloc한 것만 free)
    cudaFree(dX);
    cudaFree(dCol01); cudaFree(dCol02); cudaFree(dCol0d); cudaFree(dCol11); cudaFree(dCol12);
    cudaFree(dY0_1); cudaFree(dY0_2); cudaFree(dY0d); cudaFree(dY1_1); cudaFree(dY1_2);
    if (dId) cudaFree(dId);
    return 0;
}
