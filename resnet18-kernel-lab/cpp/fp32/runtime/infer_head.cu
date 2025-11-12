#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include "utils.hpp"

// 커널 선언
extern "C" __global__ void gap_global(const float*, int,int,int, float*);
extern "C" __global__ void softmax_1d(const float*, int, float*);

static void usage(){
    std::cout<<"Usage: step7_head --manifest exports/resnet18/fp32\n";
}

int main(int argc, char** argv){
    std::string mani;
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest" && i+1<argc) mani = argv[++i];
    }
    if (mani.empty()){ usage(); return 1; }

    // ---- 1) 입력(feature) & 정답 로드
    const std::string fixdir = mani + "/fixtures_step7";
    auto y4 = load_bin_f32(fixdir + "/after_layer4.bin");      // [C,H,W] flat
    auto ref_logits = load_bin_f32(fixdir + "/fc_logits.bin"); // [1000]
    auto ref_prob   = load_bin_f32(fixdir + "/prob_softmax.bin");

    // ResNet18 표준: C=512, H=W=7 (일반적). 파일 크기로 H,W 추정(보정 가능).
    size_t n = y4.size();
    const int C = 512;
    int HW_guess = static_cast<int>(n / C);
    int H = 1, W = 1;
    // 보통 7*7=49
    if (HW_guess == 49) { H = 7; W = 7; }
    else { // 혹시 다른 입력 크기면 정사각 가정
        int r = (int)std::lround(std::sqrt((double)HW_guess));
        H = W = r;
    }
    assert((size_t)(C*H*W) == y4.size());

    // FC 차원
    const int K_in = C;           // 512
    const int K_out = 1000;

    // ---- 2) 가중치/바이어스 로드 (Step1 export 산출물)
    auto fcW = load_bin_f32(mani + "/fc.weight.bin", (size_t)K_out * K_in); // [1000,512] (O,I)
    auto fcB = load_bin_f32(mani + "/fc.bias.bin",   (size_t)K_out);

    // ---- 3) 디바이스 메모리 준비
    // 입력 after_layer4
    float *dY4 = nullptr, *dGap = nullptr, *dW = nullptr, *dLogits = nullptr, *dProb = nullptr;
    CUDA_CHECK(cudaMalloc(&dY4, y4.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGap, K_in*sizeof(float)));            // [512]
    CUDA_CHECK(cudaMalloc(&dW,   fcW.size()*sizeof(float)));      // [1000*512]
    CUDA_CHECK(cudaMalloc(&dLogits, K_out*sizeof(float)));        // [1000]
    CUDA_CHECK(cudaMalloc(&dProb,   K_out*sizeof(float)));        // [1000]

    CUDA_CHECK(cudaMemcpy(dY4, y4.data(), y4.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW,  fcW.data(), fcW.size()*sizeof(float), cudaMemcpyHostToDevice));

    Timer T; float ms_gap, ms_fc, ms_softmax;

    // ---- 4) GAP (C,H,W → C)
    {
        dim3 grd(C), blk(256);
        T.start();
        gap_global<<<grd, blk>>>(dY4, C, H, W, dGap);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gap = T.stop();
    }

    // ---- 5) FC: logits = W(1000x512) * gap(512x1) + bias
    // 기존 sgemm_tiled(M,N,K): C[MxN] = A[MxK] * B[KxN]
    {
        dim3 blk(32,32);
        dim3 grd( (1 + 31)/32, (K_out + 31)/32 ); // N=1
        T.start();
        sgemm_tiled<<<grd, blk>>>(dW, dGap, dLogits, K_out, 1, K_in);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_fc = T.stop();

        // bias add
        // 간단 커널 대체: host에서 복사 후 더할 수도 있지만 커널 한 번 더 쓰자
        // 커널 없이 cudaMemcpy + thrust 도 가능하나 의존성 없애기 위해 간단 구현
        // 여기서는 작은 N=1이므로 host에 옮겨 더해도 무방하지만 일관성 위해 커널 작성 생략
        // -> 간단 host add:
        std::vector<float> logits_host(K_out);
        CUDA_CHECK(cudaMemcpy(logits_host.data(), dLogits, K_out*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i=0;i<K_out;i++) logits_host[i] += fcB[i];
        CUDA_CHECK(cudaMemcpy(dLogits, logits_host.data(), K_out*sizeof(float), cudaMemcpyHostToDevice));
    }

    // ---- 6) Softmax
    {
        dim3 blk(256), grd(1);
        T.start();
        softmax_1d<<<grd, blk>>>(dLogits, K_out, dProb);
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_softmax = T.stop();
    }

    // ---- 7) 결과 비교
    std::vector<float> logits(K_out), prob(K_out);
    CUDA_CHECK(cudaMemcpy(logits.data(), dLogits, K_out*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(prob.data(),   dProb,   K_out*sizeof(float), cudaMemcpyDeviceToHost));

    auto [l_max, l_mean] = diff_max_mean(logits, ref_logits);
    auto [p_max, p_mean] = diff_max_mean(prob,   ref_prob);

    std::cout<<"Head done\n";
    std::cout<<"  gap     : "<<ms_gap<<" ms\n";
    std::cout<<"  fc(gemm): "<<ms_fc<<" ms\n";
    std::cout<<"  softmax : "<<ms_softmax<<" ms\n";
    std::cout<<"Diff logits : max_abs="<<l_max<<" mean_abs="<<l_mean<<"\n";
    std::cout<<"Diff prob   : max_abs="<<p_max<<" mean_abs="<<p_mean<<"\n";

    CUDA_CHECK(cudaFree(dY4));
    CUDA_CHECK(cudaFree(dGap));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dLogits));
    CUDA_CHECK(cudaFree(dProb));

    const double atol = 1e-4;
    if (l_max<=atol && p_max<=atol) {
        std::cout<<"[OK] Step7 head matched within atol 1e-4\n";
        return 0;
    } else {
        std::cerr<<"[FAIL] Step7 diff exceeded\n";
        return 2;
    }
}
