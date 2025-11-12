// cpp/fp32/runtime/infer_layer4.cu
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "utils.hpp"

// kernels
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);

extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);

extern "C" __global__
void bn_inference(float* x, const float* g,const float* b,const float* m,const float* v,
                  float eps, int C,int OH,int OW);

extern "C" __global__
void relu_forward(float* x, int n);

extern "C" __global__
void add_inplace(float* y, const float* x, int n); // y += x

// ---- Layer4 구조 ----
// 입력: layer3_out (N=1, C=256, H=W=14)
//
// Block0: (stride=2, downsample 존재)
//   main:
//     conv1: C=256 -> OC=512, k=3x3, s=2, p=1  -> 7x7
//     bn1 -> relu
//     conv2: C=512 -> OC=512, k=3x3, s=1, p=1  -> 7x7
//     bn2
//   skip:
//     downsample: conv 1x1, C=256->OC=512, s=2 -> 7x7
//     bn
//   add + relu
//
// Block1: (stride=1, downsample 없음)
//   main:
//     conv1: 512->512, k=3x3, s=1, p=1 -> 7x7
//     bn1 -> relu
//     conv2: 512->512, k=3x3, s=1, p=1 -> 7x7
//     bn2
//   add (skip = identity) + relu

// 공통: W(OIHW) -> Wcol (O x (I*kH*kW))
static void pack_W_OIHW_to_Wcol(std::vector<float>& Wcol,
                                const std::vector<float>& W,
                                int OC, int C, int kH, int kW)
{
    const int KCOL = C*kH*kW;
    Wcol.resize(OC * KCOL);
    for (int o=0;o<OC;o++){
        for (int c=0;c<C;c++){
            for (int kh=0;kh<kH;kh++){
                for (int kw=0;kw<kW;kw++){
                    int r = c*kH*kW + kh*kW + kw; // row in Wcol
                    int src = o*C*kH*kW + c*kH*kW + kh*kW + kw; // OIHW
                    Wcol[o*KCOL + r] = W[src];
                }
            }
        }
    }
}

struct ConvCfg {
    int N=1, C, H, W;
    int OC, kH, kW, sH, sW, pH, pW;
    int OH, OW, KCOL, NCOL;
    ConvCfg(int C_,int H_,int W_,
            int OC_,int kH_,int kW_,int sH_,int sW_,int pH_,int pW_)
    : C(C_),H(H_),W(W_),OC(OC_),kH(kH_),kW(kW_),sH(sH_),sW(sW_),pH(pH_),pW(pW_)
    {
        OH = (H + 2*pH - kH)/sH + 1;
        OW = (W + 2*pW - kW)/sW + 1;
        KCOL = C*kH*kW;
        NCOL = OH*OW;
    }
};

static void usage(){
    std::cout<<"--manifest exports/resnet18/fp32\n";
}

int main(int argc, char** argv){
    std::string mani;
    for (int i=1;i<argc;i++){
        std::string a=argv[i];
        if (a=="--manifest" && i+1<argc) mani=argv[++i];
    }
    if (mani.empty()){ usage(); return 1; }

    // ----- 입력/정답 로드 -----
    // X3: layer3_out (1,256,14,14)
    auto X3 = load_bin_f32(mani+"/fixtures_step6/layer3_out.bin",
                           1*256*14*14);

    // Expected outputs
    auto Yb0E = load_bin_f32(mani+"/fixtures_step6/layer4_block0_out.bin",
                             1*512*7*7);
    auto Yb1E = load_bin_f32(mani+"/fixtures_step6/layer4_block1_out.bin",
                             1*512*7*7);

    // ----- 가중치 로드 -----
    // Block0
    auto W01 = load_bin_f32(mani+"/layer4.0.conv1.weight.bin", 512*256*3*3);
    auto G01 = load_bin_f32(mani+"/layer4.0.bn1.weight.bin", 512);
    auto B01 = load_bin_f32(mani+"/layer4.0.bn1.bias.bin", 512);
    auto M01 = load_bin_f32(mani+"/layer4.0.bn1.running_mean.bin", 512);
    auto V01 = load_bin_f32(mani+"/layer4.0.bn1.running_var.bin", 512);

    auto W02 = load_bin_f32(mani+"/layer4.0.conv2.weight.bin", 512*512*3*3);
    auto G02 = load_bin_f32(mani+"/layer4.0.bn2.weight.bin", 512);
    auto B02 = load_bin_f32(mani+"/layer4.0.bn2.bias.bin", 512);
    auto M02 = load_bin_f32(mani+"/layer4.0.bn2.running_mean.bin", 512);
    auto V02 = load_bin_f32(mani+"/layer4.0.bn2.running_var.bin", 512);

    auto W0d = load_bin_f32(mani+"/layer4.0.downsample.0.weight.bin", 512*256*1*1);
    auto G0d = load_bin_f32(mani+"/layer4.0.downsample.1.weight.bin", 512);
    auto B0d = load_bin_f32(mani+"/layer4.0.downsample.1.bias.bin", 512);
    auto M0d = load_bin_f32(mani+"/layer4.0.downsample.1.running_mean.bin", 512);
    auto V0d = load_bin_f32(mani+"/layer4.0.downsample.1.running_var.bin", 512);

    // Block1
    auto W11 = load_bin_f32(mani+"/layer4.1.conv1.weight.bin", 512*512*3*3);
    auto G11 = load_bin_f32(mani+"/layer4.1.bn1.weight.bin", 512);
    auto B11 = load_bin_f32(mani+"/layer4.1.bn1.bias.bin", 512);
    auto M11 = load_bin_f32(mani+"/layer4.1.bn1.running_mean.bin", 512);
    auto V11 = load_bin_f32(mani+"/layer4.1.bn1.running_var.bin", 512);

    auto W12 = load_bin_f32(mani+"/layer4.1.conv2.weight.bin", 512*512*3*3);
    auto G12 = load_bin_f32(mani+"/layer4.1.bn2.weight.bin", 512);
    auto B12 = load_bin_f32(mani+"/layer4.1.bn2.bias.bin", 512);
    auto M12 = load_bin_f32(mani+"/layer4.1.bn2.running_mean.bin", 512);
    auto V12 = load_bin_f32(mani+"/layer4.1.bn2.running_var.bin", 512);

    // ----- Host → Device -----
    // 입력 X3
    auto dX3 = copy_to_device(X3);

    // Block0 conv1 cfg (256->512, k3,s2,p1 on 14x14 => 7x7)
    ConvCfg cfg01(/*C=*/256, /*H=*/14, /*W=*/14,
                  /*OC=*/512, /*kH=*/3,/*kW=*/3, /*sH=*/2,/*sW=*/2, /*pH=*/1,/*pW=*/1);
    // conv2 cfg (512->512, k3,s1,p1 on 7x7 => 7x7)
    ConvCfg cfg02(/*C=*/512, /*H=*/7, /*W=*/7,
                  /*OC=*/512, /*kH=*/3,/*kW=*/3, /*sH=*/1,/*sW=*/1, /*pH=*/1,/*pW=*/1);
    // downsample conv cfg (1x1, s=2)
    ConvCfg cfg0d(/*C=*/256, /*H=*/14, /*W=*/14,
                  /*OC=*/512, /*kH=*/1,/*kW=*/1, /*sH=*/2,/*sW=*/2, /*pH=*/0,/*pW=*/0);

    // Block1 convs (512->512 on 7x7)
    ConvCfg cfg11(/*C=*/512, /*H=*/7, /*W=*/7,
                  /*OC=*/512, /*kH=*/3,/*kW=*/3, /*sH=*/1,/*sW=*/1, /*pH=*/1,/*pW=*/1);
    ConvCfg cfg12 = cfg11;

    // Pack weights to (OC x KCOL)
    std::vector<float> W01col, W02col, W0dcol, W11col, W12col;
    pack_W_OIHW_to_Wcol(W01col, W01, 512, 256, 3, 3);
    pack_W_OIHW_to_Wcol(W02col, W02, 512, 512, 3, 3);
    // 1x1 conv
    {
        W0dcol.resize(512 * (256*1*1));
        for (int o=0;o<512;o++){
            for (int c=0;c<256;c++){
                int r = c; // kH=kW=1
                int src = o*256 + c;
                W0dcol[o*(256) + r] = W0d[src];
            }
        }
    }
    pack_W_OIHW_to_Wcol(W11col, W11, 512, 512, 3, 3);
    pack_W_OIHW_to_Wcol(W12col, W12, 512, 512, 3, 3);

    // Device buffers
    auto dW01 = copy_to_device(W01col);
    auto dW02 = copy_to_device(W02col);
    auto dW0d = copy_to_device(W0dcol);
    auto dG01 = copy_to_device(G01), dB01 = copy_to_device(B01),
         dM01 = copy_to_device(M01), dV01 = copy_to_device(V01);
    auto dG02 = copy_to_device(G02), dB02 = copy_to_device(B02),
         dM02 = copy_to_device(M02), dV02 = copy_to_device(V02);
    auto dG0d = copy_to_device(G0d), dB0d = copy_to_device(B0d),
         dM0d = copy_to_device(M0d), dV0d = copy_to_device(V0d);

    auto dW11 = copy_to_device(W11col);
    auto dW12 = copy_to_device(W12col);
    auto dG11 = copy_to_device(G11), dB11 = copy_to_device(B11),
         dM11 = copy_to_device(M11), dV11 = copy_to_device(V11);
    auto dG12 = copy_to_device(G12), dB12 = copy_to_device(B12),
         dM12 = copy_to_device(M12), dV12 = copy_to_device(V12);

    // Workspaces
    auto dCol01 = make_device_f32(cfg01.KCOL * cfg01.NCOL);
    auto dY01   = make_device_f32(cfg01.OC   * cfg01.NCOL); // after conv1
    auto dSkip0 = make_device_f32(cfg0d.OC   * cfg0d.NCOL); // downsample path
    auto dCol02 = make_device_f32(cfg02.KCOL * cfg02.NCOL);
    auto dY02   = make_device_f32(cfg02.OC   * cfg02.NCOL); // block0 output

    auto dCol11 = make_device_f32(cfg11.KCOL * cfg11.NCOL);
    auto dY11   = make_device_f32(cfg11.OC   * cfg11.NCOL);
    auto dCol12 = make_device_f32(cfg12.KCOL * cfg12.NCOL);
    auto dY12   = make_device_f32(cfg12.OC   * cfg12.NCOL); // block1 output

    Timer T;
    float ms_im2col=0, ms_gemm=0, ms_bn=0, ms_relu=0, ms_add=0;

    // ===== Block0 =====
    {
        // main path conv1: im2col
        dim3 blk1(16,16);
        dim3 grd1((cfg01.OW+blk1.x-1)/blk1.x, (cfg01.OH+blk1.y-1)/blk1.y);
        T.start();
        im2col_nchw<<<grd1,blk1>>>(dX3.get(), 1,cfg01.C,cfg01.H,cfg01.W,
                                   cfg01.kH,cfg01.kW,cfg01.sH,cfg01.sW,cfg01.pH,cfg01.pW,
                                   dCol01.get());
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        // GEMM1: Y01 = W01 * Col01
        dim3 blk2(32,32);
        dim3 grd2((cfg01.NCOL+31)/32, (cfg01.OC+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW01.get(), dCol01.get(), dY01.get(),
                                   cfg01.OC, cfg01.NCOL, cfg01.KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        // BN1 + ReLU (C=512, OH=7, OW=7)
        T.start();
        bn_inference<<< div_up(cfg01.OC*cfg01.OH*cfg01.OW,256), 256 >>>(
            dY01.get(), dG01.get(), dB01.get(), dM01.get(), dV01.get(),
            1e-5f, cfg01.OC, cfg01.OH, cfg01.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();

        T.start();
        relu_forward<<< div_up(cfg01.OC*cfg01.OH*cfg01.OW,256), 256 >>>(
            dY01.get(), cfg01.OC*cfg01.OH*cfg01.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu += T.stop();

        // skip: downsample 1x1 s2
        dim3 grd1d((cfg0d.OW+blk1.x-1)/blk1.x, (cfg0d.OH+blk1.y-1)/blk1.y);
        T.start();
        im2col_nchw<<<grd1d,blk1>>>(dX3.get(), 1,cfg0d.C,cfg0d.H,cfg0d.W,
                                    cfg0d.kH,cfg0d.kW,cfg0d.sH,cfg0d.sW,cfg0d.pH,cfg0d.pW,
                                    dCol01.get()); // 재사용 (KCOL=256, NCOL=49)
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 grd2d((cfg0d.NCOL+31)/32, (cfg0d.OC+31)/32);
        T.start();
        sgemm_tiled<<<grd2d,blk2>>>(dW0d.get(), dCol01.get(), dSkip0.get(),
                                    cfg0d.OC, cfg0d.NCOL, cfg0d.KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        T.start();
        bn_inference<<< div_up(cfg0d.OC*cfg0d.OH*cfg0d.OW,256), 256 >>>(
            dSkip0.get(), dG0d.get(), dB0d.get(), dM0d.get(), dV0d.get(),
            1e-5f, cfg0d.OC, cfg0d.OH, cfg0d.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();

        // main path conv2 (input = Y01)
        dim3 grd1b((cfg02.OW+blk1.x-1)/blk1.x, (cfg02.OH+blk1.y-1)/blk1.y);
        T.start();
        im2col_nchw<<<grd1b,blk1>>>(dY01.get(), 1,cfg02.C,cfg02.H,cfg02.W,
                                    cfg02.kH,cfg02.kW,cfg02.sH,cfg02.sW,cfg02.pH,cfg02.pW,
                                    dCol02.get());
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 grd2b((cfg02.NCOL+31)/32, (cfg02.OC+31)/32);
        T.start();
        sgemm_tiled<<<grd2b,blk2>>>(dW02.get(), dCol02.get(), dY02.get(),
                                    cfg02.OC, cfg02.NCOL, cfg02.KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        // BN2
        T.start();
        bn_inference<<< div_up(cfg02.OC*cfg02.OH*cfg02.OW,256), 256 >>>(
            dY02.get(), dG02.get(), dB02.get(), dM02.get(), dV02.get(),
            1e-5f, cfg02.OC, cfg02.OH, cfg02.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();

        // add skip + relu
        T.start();
        add_inplace<<< div_up(cfg02.OC*cfg02.OH*cfg02.OW,256), 256 >>>(
            dY02.get(), dSkip0.get(), cfg02.OC*cfg02.OH*cfg02.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_add += T.stop();

        T.start();
        relu_forward<<< div_up(cfg02.OC*cfg02.OH*cfg02.OW,256), 256 >>>(
            dY02.get(), cfg02.OC*cfg02.OH*cfg02.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu += T.stop();

        // diff vs expected
        auto Yb0 = copy_to_host(dY02, cfg02.OC*cfg02.NCOL);
        auto [mx0, mn0] = diff_max_mean(Yb0, Yb0E);
        std::cout<<"[Layer4 Block0]\n";
        std::cout<<"  im2col: "<<ms_im2col<<" ms,  gemm: "<<ms_gemm
                 <<" ms,  bn: "<<ms_bn<<" ms,  add: "<<ms_add
                 <<" ms,  relu: "<<ms_relu<<" ms\n";
        std::cout<<"  diff  : max_abs="<<mx0<<" mean_abs="<<mn0<<"\n";
        if (mx0 > 1e-4){ std::cerr<<"[FAIL] Step6 diff exceeded (block0)\n"; return 2; }
    }

    // reset timing for block1 (표시만 깔끔히)
    ms_im2col=ms_gemm=ms_bn=ms_relu=ms_add=0.f;

    // ===== Block1 (identity skip) =====
    {
        // conv1
        dim3 blk1(16,16);
        dim3 grd1((cfg11.OW+blk1.x-1)/blk1.x, (cfg11.OH+blk1.y-1)/blk1.y);
        T.start();
        im2col_nchw<<<grd1,blk1>>>(dY02.get(), 1,cfg11.C,cfg11.H,cfg11.W,
                                   cfg11.kH,cfg11.kW,cfg11.sH,cfg11.sW,cfg11.pH,cfg11.pW,
                                   dCol11.get());
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 blk2(32,32);
        dim3 grd2((cfg11.NCOL+31)/32, (cfg11.OC+31)/32);
        T.start();
        sgemm_tiled<<<grd2,blk2>>>(dW11.get(), dCol11.get(), dY11.get(),
                                   cfg11.OC, cfg11.NCOL, cfg11.KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        T.start();
        bn_inference<<< div_up(cfg11.OC*cfg11.OH*cfg11.OW,256), 256 >>>(
            dY11.get(), dG11.get(), dB11.get(), dM11.get(), dV11.get(),
            1e-5f, cfg11.OC, cfg11.OH, cfg11.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();

        T.start();
        relu_forward<<< div_up(cfg11.OC*cfg11.OH*cfg11.OW,256), 256 >>>(
            dY11.get(), cfg11.OC*cfg11.OH*cfg11.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu += T.stop();

        // conv2
        dim3 grd1b((cfg12.OW+blk1.x-1)/blk1.x, (cfg12.OH+blk1.y-1)/blk1.y);
        T.start();
        im2col_nchw<<<grd1b,blk1>>>(dY11.get(), 1,cfg12.C,cfg12.H,cfg12.W,
                                    cfg12.kH,cfg12.kW,cfg12.sH,cfg12.sW,cfg12.pH,cfg12.pW,
                                    dCol12.get());
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col += T.stop();

        dim3 grd2b((cfg12.NCOL+31)/32, (cfg12.OC+31)/32);
        T.start();
        sgemm_tiled<<<grd2b,blk2>>>(dW12.get(), dCol12.get(), dY12.get(),
                                    cfg12.OC, cfg12.NCOL, cfg12.KCOL);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm += T.stop();

        T.start();
        bn_inference<<< div_up(cfg12.OC*cfg12.OH*cfg12.OW,256), 256 >>>(
            dY12.get(), dG12.get(), dB12.get(), dM12.get(), dV12.get(),
            1e-5f, cfg12.OC, cfg12.OH, cfg12.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn += T.stop();

        // add identity + relu (identity는 dY02)
        T.start();
        add_inplace<<< div_up(cfg12.OC*cfg12.OH*cfg12.OW,256), 256 >>>(
            dY12.get(), dY02.get(), cfg12.OC*cfg12.OH*cfg12.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_add += T.stop();

        T.start();
        relu_forward<<< div_up(cfg12.OC*cfg12.OH*cfg12.OW,256), 256 >>>(
            dY12.get(), cfg12.OC*cfg12.OH*cfg12.OW);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu += T.stop();

        auto Yb1 = copy_to_host(dY12, cfg12.OC*cfg12.NCOL);
        auto [mx1, mn1] = diff_max_mean(Yb1, Yb1E);
        std::cout<<"[Layer4 Block1]\n";
        std::cout<<"  im2col: "<<ms_im2col<<" ms,  gemm: "<<ms_gemm
                 <<" ms,  bn: "<<ms_bn<<" ms,  add: "<<ms_add
                 <<" ms,  relu: "<<ms_relu<<" ms\n";
        std::cout<<"  diff  : max_abs="<<mx1<<" mean_abs="<<mn1<<"\n";
        if (mx1 > 1e-4){ std::cerr<<"[FAIL] Step6 diff exceeded (block1)\n"; return 3; }
    }

    std::cout<<"[OK] Step6 layer4 matched within atol 1e-4\n";
    return 0;
}
