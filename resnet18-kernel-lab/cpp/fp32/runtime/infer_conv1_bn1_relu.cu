#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "utils.hpp"

// 커널 선언
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);

extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);

extern "C" __global__
void bn_inference(float*,const float*,const float*,const float*,const float*,float,int,int,int);

extern "C" __global__
void relu_forward(float*,int);

// 간단 파라미터 (ResNet18 conv1 설정)
struct Conv1Cfg {
    int N=1,C=3,H=224,W=224;
    int OC=64, KH=7, KW=7, SH=2, SW=2, PH=3, PW=3;
    int OH=(H+2*PH-KH)/SH+1;
    int OW=(W+2*PW-KW)/SW+1;
    int KCOL=C*KH*KW; // rows of W_col
    int NCOL=OH*OW;   // columns of im2col
};

static void usage(){
    std::cout<<"--manifest exports/resnet18/fp32 --input .../input.bin --expect .../expected.bin\n";
}

int main(int argc, char** argv){
    std::string mani, inputPath, expectPath;
    for (int i=1;i<argc;i++){
        std::string a=argv[i];
        if (a=="--manifest" && i+1<argc) mani=argv[++i];
        else if (a=="--input" && i+1<argc) inputPath=argv[++i];
        else if (a=="--expect" && i+1<argc) expectPath=argv[++i];
    }
    if (mani.empty()||inputPath.empty()||expectPath.empty()){ usage(); return 1; }

    Conv1Cfg cfg;

    // --- 1) 가중치 로드 (Step1에서 export한 파일 사용)
    auto w = load_bin_f32(mani+"/conv1.weight.bin", cfg.OC*cfg.C*cfg.KH*cfg.KW);
    auto bn_w = load_bin_f32(mani+"/bn1.weight.bin", cfg.OC);
    auto bn_b = load_bin_f32(mani+"/bn1.bias.bin", cfg.OC);
    auto bn_m = load_bin_f32(mani+"/bn1.running_mean.bin", cfg.OC);
    auto bn_v = load_bin_f32(mani+"/bn1.running_var.bin", cfg.OC);

    // --- 2) 입력/정답 로드 (토치 기준)
    auto x  = load_bin_f32(inputPath,  cfg.N*cfg.C*cfg.H*cfg.W);
    auto yE = load_bin_f32(expectPath, cfg.N*cfg.OC*cfg.OH*cfg.OW);

    // --- 3) W_col (OC x KCOL)로 재배열 (OIHW -> O x (I*KH*KW))
    std::vector<float> Wcol(cfg.OC * cfg.KCOL);
    for (int o=0;o<cfg.OC;o++){
        for (int c=0;c<cfg.C;c++){
            for (int kh=0;kh<cfg.KH;kh++){
                for (int kw=0;kw<cfg.KW;kw++){
                    int r = c*cfg.KH*cfg.KW + kh*cfg.KW + kw; // col row
                    int src = o*cfg.C*cfg.KH*cfg.KW + c*cfg.KH*cfg.KW + kh*cfg.KW + kw;
                    Wcol[o*cfg.KCOL + r] = w[src];
                }
            }
        }
    }

    // --- 4) 디바이스 메모리 할당
    float *dX,*dCol,*dW,*dY;
    CUDA_CHECK(cudaMalloc(&dX,   x.size()*4));
    CUDA_CHECK(cudaMalloc(&dCol, cfg.KCOL*cfg.NCOL*4));
    CUDA_CHECK(cudaMalloc(&dW,   Wcol.size()*4));
    CUDA_CHECK(cudaMalloc(&dY,   cfg.OC*cfg.NCOL*4));

    float *dBnW,*dBnB,*dBnM,*dBnV;
    CUDA_CHECK(cudaMalloc(&dBnW, cfg.OC*4));
    CUDA_CHECK(cudaMalloc(&dBnB, cfg.OC*4));
    CUDA_CHECK(cudaMalloc(&dBnM, cfg.OC*4));
    CUDA_CHECK(cudaMalloc(&dBnV, cfg.OC*4));

    CUDA_CHECK(cudaMemcpy(dX,   x.data(),    x.size()*4,               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW,   Wcol.data(), Wcol.size()*4,            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBnW, bn_w.data(), cfg.OC*4,                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBnB, bn_b.data(), cfg.OC*4,                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBnM, bn_m.data(), cfg.OC*4,                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBnV, bn_v.data(), cfg.OC*4,                 cudaMemcpyHostToDevice));

    // --- 5) 커널 실행
    Timer T; float ms_im2col, ms_gemm, ms_bn, ms_relu;

    // im2col
    dim3 blk1(16,16);
    dim3 grd1( (cfg.OW+blk1.x-1)/blk1.x, (cfg.OH+blk1.y-1)/blk1.y );
    T.start();
    im2col_nchw<<<grd1,blk1>>>(dX, cfg.N,cfg.C,cfg.H,cfg.W, cfg.KH,cfg.KW,cfg.SH,cfg.SW,cfg.PH,cfg.PW, dCol);
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_im2col = T.stop();

    // GEMM: Y(OC x NCOL) = W_col(OC x KCOL) * Col(KCOL x NCOL)
    dim3 blk2(32,32);
    dim3 grd2( (cfg.NCOL+31)/32, (cfg.OC+31)/32 );
    T.start();
    sgemm_tiled<<<grd2,blk2>>>(dW, dCol, dY, cfg.OC, cfg.NCOL, cfg.KCOL);
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_gemm = T.stop();

    // BN inference: in-place on Y reshaped as (C,OH,OW)
    T.start();
    bn_inference<<< (cfg.OC*cfg.OH*cfg.OW +255)/256, 256 >>>(dY, dBnW, dBnB, dBnM, dBnV, 1e-5f, cfg.OC, cfg.OH, cfg.OW);
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_bn = T.stop();

    std::vector<float> y_bn(cfg.OC*cfg.OH*cfg.OW);
    CUDA_CHECK(cudaMemcpy(y_bn.data(), dY, y_bn.size()*4, cudaMemcpyDeviceToHost));
    save_bin_f32("out/step2_bn1_cuda.bin", y_bn);
    
    // ReLU
    T.start();
    relu_forward<<< (cfg.OC*cfg.OH*cfg.OW +255)/256, 256 >>>(dY, cfg.OC*cfg.OH*cfg.OW);
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_relu = T.stop();

    // --- 6) 결과 비교
    std::vector<float> y(cfg.OC*cfg.OH*cfg.OW);
    CUDA_CHECK(cudaMemcpy(y.data(), dY, y.size()*4, cudaMemcpyDeviceToHost));

    double max_abs = 0.0, mean_abs = 0.0;
    for (size_t i=0;i<y.size();++i){
        double d = std::fabs((double)y[i] - (double)yE[i]);
        max_abs = std::max(max_abs, d);
        mean_abs += d;
    }
    mean_abs /= y.size();

    std::cout<<"Step2 conv1->bn1->relu done\n";
    std::cout<<"  im2col: "<<ms_im2col<<" ms\n";
    std::cout<<"  gemm  : "<<ms_gemm<<" ms\n";
    std::cout<<"  bn    : "<<ms_bn<<" ms\n";
    std::cout<<"  relu  : "<<ms_relu<<" ms\n";
    std::cout<<"Diff    : max_abs="<<max_abs<<" mean_abs="<<mean_abs<<"\n";

    // --- 7) 자원 해제
    cudaFree(dX); cudaFree(dCol); cudaFree(dW); cudaFree(dY);
    cudaFree(dBnW); cudaFree(dBnB); cudaFree(dBnM); cudaFree(dBnV);

    // 통과 기준: atol <= 1e-4
    if (max_abs <= 1e-4) {
        std::cout<<"[OK] within atol 1e-4\n";
        return 0;
    } else {
        std::cerr<<"[FAIL] exceed atol 1e-4\n";
        return 2;
    }
}