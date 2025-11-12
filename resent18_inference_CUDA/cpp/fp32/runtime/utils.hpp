#pragma once
// utils.hpp (clean single-source version)

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <utility>   // std::pair
#include <cassert>
#include <cmath>
#include <cstdlib>

#include <cuda_runtime.h>

// -------------------- small helpers --------------------
static inline int div_up(int a, int b) { return (a + b - 1) / b; }
static inline int max3(int a, int b, int c){ return std::max(a, std::max(b,c)); }

// -------------------- CUDA error/launch helpers --------------------
inline void __cuda_check(cudaError_t e, const char* f, int l){
    if (e != cudaSuccess){
        std::cerr << "CUDA " << cudaGetErrorString(e)
                  << " @ " << f << ":" << l << "\n";
        std::exit(1);
    }
}
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) __cuda_check((expr), __FILE__, __LINE__)
#endif

#ifndef CUDA_LAUNCH
#define CUDA_LAUNCH(kernel, grid, block, ...) \
do { \
  kernel<<<(grid), (block)>>>(__VA_ARGS__); \
  cudaError_t __e = cudaGetLastError(); \
  if (__e != cudaSuccess){ \
    fprintf(stderr, "CUDA launch error %s @ %s:%d\n", \
            cudaGetErrorString(__e), __FILE__, __LINE__); \
    return 3; \
  } \
} while(0)
#endif

// -------------------- binary IO --------------------
inline std::vector<float> load_bin_f32(const std::string& path, size_t expected = 0){
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { std::cerr<<"open fail: "<<path<<"\n"; std::exit(1); }
    ifs.seekg(0, std::ios::end); size_t bytes = static_cast<size_t>(ifs.tellg()); ifs.seekg(0);
    if (bytes % 4){ std::cerr<<"size not float-aligned: "<<path<<"\n"; std::exit(1); }
    size_t n = bytes/4;
    std::vector<float> v(n);
    if (n) ifs.read(reinterpret_cast<char*>(v.data()), bytes);
    if (expected && n!=expected){
        std::cerr<<"unexpected size: "<<path<<" got "<<n<<" expected "<<expected<<"\n"; std::exit(1);
    }
    return v;
}

inline void save_bin_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) { std::cerr << "open fail (write): " << path << "\n"; std::exit(1); }
    if (!v.empty())
        ofs.write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(float));
}

inline void ensure_out_dir(const char* dir = "out") {
    std::string cmd = std::string("mkdir -p ") + dir;
    std::system(cmd.c_str());
}

// -------------------- extern CUDA kernels (prototypes) --------------------
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);
extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);
extern "C" __global__
void bn_inference(float*,const float*,const float*,const float*,const float*,float,int,int,int);
extern "C" __global__
void relu_forward(float*,int);

// -------------------- Timer --------------------
struct Timer {
    cudaEvent_t a,b;
    Timer(){ CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(){ CUDA_CHECK(cudaEventRecord(a)); }
    float stop(){ CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
                  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};

// -------------------- simple wrappers (Step2/compat) --------------------
struct CmdArgs { std::string manifest; };
inline CmdArgs parse_args(int argc, char** argv) {
    CmdArgs a;
    for (int i=1;i<argc;i++){
        std::string s = argv[i];
        if (s=="--manifest" && i+1<argc) a.manifest = argv[++i];
    }
    return a;
}
struct Tensor {
    std::vector<float> buf;
    float* data() { return buf.data(); }
    const float* data() const { return buf.data(); }
    size_t size() const { return buf.size(); }
};
inline Tensor load_bin(const std::string& path, size_t n_elems) {
    Tensor t; t.buf = load_bin_f32(path, n_elems); return t;
}
struct Manifest {
    std::string root;
    explicit Manifest(std::string r): root(std::move(r)){}
    Tensor load(const std::string& name, size_t n_elems) {
        return load_bin(root + "/" + name + ".bin", n_elems);
    }
};

// -------------------- device memory helpers (unique_ptr based) --------------------
struct DeviceDeleter {
    void operator()(float* p) const noexcept {
        if (p) cudaFree(p);
    }
};

using DevicePtr = std::unique_ptr<float, DeviceDeleter>;

// allocate device float[n] and return unique_ptr
inline DevicePtr make_device_f32(size_t n_elems) {
    if (n_elems == 0) return DevicePtr(nullptr);
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n_elems * sizeof(float)));
    return DevicePtr(p);
}

// host -> device (vector)
inline DevicePtr copy_to_device(const std::vector<float>& host) {
    DevicePtr d = make_device_f32(host.size());
    if (!host.empty())
        CUDA_CHECK(cudaMemcpy(d.get(), host.data(), host.size()*sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// host -> device (raw ptr, count)
inline DevicePtr copy_to_device(const float* src, size_t n_elems) {
    DevicePtr d = make_device_f32(n_elems);
    if (n_elems && src)
        CUDA_CHECK(cudaMemcpy(d.get(), src, n_elems*sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// device -> host
inline std::vector<float> copy_to_host(const DevicePtr& dptr, size_t n_elems) {
    std::vector<float> h(n_elems);
    if (n_elems)
        CUDA_CHECK(cudaMemcpy(h.data(), dptr.get(), n_elems * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// -------------------- diff utilities --------------------
inline std::pair<double,double>
diff_max_mean(const std::vector<float>& a, const std::vector<float>& b)
{
    assert(a.size() == b.size());
    const size_t n = a.size();
    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::abs((double)a[i] - (double)b[i]);
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    if (n) mean_abs /= (double)n;
    return {max_abs, mean_abs};
}
