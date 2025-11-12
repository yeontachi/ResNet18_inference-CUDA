# ResNet18 CUDA (FP32)

This project re-implements **ResNet-18** inference with **pure CUDA kernels** and reproduces PyTorch results on ImageNet val.
We match PyTorch **Top-1 predictions 100%** on a 500-image subset while verifying layer-by-layer numerical equivalence.

## Highlights

* NCHW layout, **im2col + SGEMM** convolution
* BatchNorm Inference (**eps = 1e-5**) + in-place ReLU
* Residual connections (including downsample 1×1)
* Full **end-to-end (E2E)**: Stem → Layer1–4 → GAP → FC
* **Stepwise verification tools** and **tensor dump diff tools**
* FP32 only (quantization / Winograd etc. planned as future work)

---

## Repository Structure

```
ResNet18_CUDA_FP32/
├─ cpp/
│  └─ fp32/
│     ├─ kernels/
│     │  ├─ im2col.cu
│     │  ├─ sgemm_tiled.cu
│     │  ├─ bn_inference.cu
│     │  ├─ relu.cu
│     │  ├─ add.cu
│     │  ├─ gap_global.cu
│     │  └─ maxpool2d.cu
│     ├─ runtime/
│     │  ├─ utils.hpp
│     │  ├─ infer_conv1_bn1_relu.cu   # Step2
│     │  ├─ infer_layer1.cu           # Step3
│     │  ├─ infer_layer2.cu           # Step4
│     │  ├─ infer_layer3.cu           # Step5
│     │  ├─ infer_layer4.cu           # Step6
│     │  ├─ infer_head.cu             # Step7 (GAP+FC+Softmax)
│     │  └─ infer_e2e.cu              # Step8 (full pipeline)
│     └─ CMakeLists.txt
├─ exports/
│  └─ resnet18/fp32/
│     ├─ conv1.weight.bin
│     ├─ bn1.(weight|bias|running_mean|running_var).bin
│     ├─ layer{1..4}/* (weights & BN stats for all blocks)
│     ├─ fc.(weight|bias).bin
│     ├─ fixtures/                 # per-step inputs/targets
│     └─ fixtures_e2e/             # torch dumps for E2E comparison
├─ data/
│  └─ imagenet_val/ILSVRC2012_img_val/
├─ tools/
│  ├─ make_step2_fixture.py
│  ├─ make_e2e_fixtures.py
│  ├─ bench_fp32_vs_torch_e2e.py
│  └─ diag_e2e_compare.py
├─ scripts/
│  ├─ build_fp32.sh
│  ├─ run_step2.sh .. run_step7.sh
│  └─ run_step9_e2e.sh
└─ README.md
```

> We export binaries from `torchvision.models.resnet18(ResNet18_Weights.IMAGENET1K_V1)`.

---

## Requirements

* **CUDA** 12+ (recommended)
* **CMake** 3.20+
* **Python** 3.10+ with: `torch`, `torchvision`, `numpy`, `Pillow`
* NVIDIA GPU (CC 7.x+ recommended)

Install Python deps:

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision pillow numpy
```

---

## Build

```bash
# From repo root
bash scripts/build_fp32.sh
# or manually:
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../cpp
cmake --build . -j
```

Binaries will appear in `build/fp32/`:

```
step2_conv1_bn1_relu  step3_layer1  step4_layer2  step5_layer3
step6_layer4          step7_head    step8_e2e
```

---

## Data & Paths
Actually, you need to **manually download** the ImageNet validation set and place it in the following directory before running the scripts:

```bash
data/imagenet_val/ILSVRC2012_img_val/
```

In other words, this directory must be **created manually** after downloading the dataset.
Once the data is placed correctly, you can set the environment variable for convenience:

```bash
export ROOT="$(pwd)"   # from the repository root
```

---

## Fixtures & Stepwise Verification

### 1) Generate Step2 fixture

```bash
python tools/make_step2_fixture.py \
  --outdir "$ROOT/exports/resnet18/fp32/fixtures"
```

### 2) Run Step2 (conv1 + BN + ReLU)

```bash
build/fp32/step2_conv1_bn1_relu \
  --manifest "$ROOT/exports/resnet18/fp32" \
  --input    "$ROOT/exports/resnet18/fp32/fixtures/input.bin" \
  --expect   "$ROOT/exports/resnet18/fp32/fixtures/expected.bin"
# Expect: [OK] within atol 1e-4
```

### 3) Run Step3–7

```bash
bash scripts/run_step3.sh   # layer1
bash scripts/run_step4.sh   # layer2
bash scripts/run_step5.sh   # layer3
bash scripts/run_step6.sh   # layer4
bash scripts/run_step7.sh   # head (GAP+FC+Softmax)
# Expect per-step diffs: max_abs ~ 1e-6..1e-5, cosine = 1.0
```

---

## End-to-End (E2E) Verification

### 1) Dump torch reference tensors (for a single input)

```bash
python tools/make_e2e_fixtures.py \
  --outdir "$ROOT/exports/resnet18/fp32/fixtures_e2e"
```

### 2) Run CUDA E2E and dump the same checkpoints

```bash
build/fp32/step8_e2e \
  --manifest "$ROOT/exports/resnet18/fp32" \
  --input    "$ROOT/exports/resnet18/fp32/fixtures/input.bin" \
  --dump_dir "$ROOT/exports/resnet18/fp32/cuda_e2e_dumps"
# Dumps: stem_pool.bin, layer1.bin, ..., layer4.bin, gap.bin, logits.bin
```

### 3) Compare dumps (torch vs CUDA)

```bash
python tools/diag_e2e_compare.py \
  --torch_dir "$ROOT/exports/resnet18/fp32/fixtures_e2e" \
  --cuda_dir  "$ROOT/exports/resnet18/fp32/cuda_e2e_dumps"
# Expect cosine = 1.0 and tiny max_abs (~1e-6..1e-5) for all checkpoints
```

---

## ImageNet Val 500-Image Benchmark (Correctness)

```bash
bash scripts/run_step9_e2e.sh
# Example:
# images=500
# agree_top1=500 (100.00%)
# torch_ms=2.63
# cuda_ms=221.48
# speedup=0.01x
```

* **agree_top1** = fraction of images where PyTorch and CUDA Top-1 class match.
* Timing shows **reference (pre-optimization)** performance.

---

## Preprocessing (Critical for Matching)

* **Resize** (shorter side = 256) → **CenterCrop(224)** → **ToTensor** → **Normalize**
* Use `ResNet18_Weights.IMAGENET1K_V1.transforms()` for mean/std
* Convert to **NCHW** before feeding the model

> Any mismatch in preprocessing, stride, padding, epsilon, or layout will break equivalence.

---

## Implementation Notes

* **Convolution**

  * `im2col_nchw`: shape → (KCOL, OH×OW)
  * `sgemm_tiled`: (OC×KCOL) × (KCOL×OH×OW) → (OC×OH×OW)
* **BatchNorm Inference**

  * Uses `running_mean` / `running_var` (variance), **eps = 1e-5**
* **Residual**

  * Downsample path: **1×1** conv with stride (usually 2) + BN
* **GAP**

  * Channel-wise mean over H×W
* **FC**

  * `[1000×512] @ [512] + [1000]`

---

## Troubleshooting

* **Large diff in Step2**
  Check preprocessing (mean/std), conv1 params (7×7, stride 2, pad 3), BN eps and stats.
* **Only E2E mismatches**
  Use dump-diff to locate first bad checkpoint.
  Verify GAP reduction order/divisor and FC matrix interpretation (row/col).
* **`copy_to_device` errors**
  Make sure you’re calling the correct overload (vector vs pointer+length) and that `utils.hpp` has no duplicate declarations.
* **BN eps / running stats**
  Use **eps = 1e-5** and **running_var** (variance), not stddev.

---

## Performance Roadmap (Next)

1. Stronger GEMM tiling (e.g., 128×64/64×64), double buffering, register tiling
2. **Implicit GEMM** (im2col-less) or **Winograd (3×3)**
3. Kernel fusion (BN+ReLU, add+ReLU)
4. Batched inference (N>1) & I/O-free pipelines
5. Streams & workspace reuse

---

## Reproducible Results (Example)

* **Per-step diffs**: max_abs ≈ 1e-6..1e-5, cosine = 1.0
* **E2E on 500 images**: `agree_top1 = 100.00%`
* **Timing**: PyTorch ≈ 2.63 ms/img, CUDA ≈ 221.48 ms/img (before optimization)

---

## Weights & License

* Weights: `torchvision.models.resnet18(ResNet18_Weights.IMAGENET1K_V1)`
* Code: MIT (adjust if you prefer)
* ImageNet data follows its own license.

---

## Acknowledgements

* PyTorch / Torchvision teams
* NVIDIA CUDA documentation and samples


If you want, I can also generate a short **README_kr.md**, add badges, and include screenshots or profiler tables.
