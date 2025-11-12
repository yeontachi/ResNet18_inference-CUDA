# Step 2 보고서 — ResNet18 FP32 커널: conv1 → bn1 → relu 구현 및 검증

## 1) 목표(Goal)

* Step 1에서 내보낸 FP32 가중치(`manifest.json`)를 **CUDA 런타임**에서 로드한다.
* ResNet18 **stem** 경로인 `conv1 → bn1 → relu`를 **자체 FP32 커널**로 실행한다.
* PyTorch 기준 출력과 **수치 일치(absolute tolerance, `atol ≤ 1e-4`)** 를 달성한다.
* 각 단계의 **실행 시간(ms)** 을 로깅한다.

---

## 2) 산출물(Deliverables)

* **코드**

  ```
  cpp/fp32/
    ├─ kernels/
    │   ├─ im2col.cu          # 입력 NCHW → (C*kH*kW, OH*OW) 열 행렬
    │   ├─ sgemm_tiled.cu     # 공유메모리 타일링 SGEMM (W_col × col → Y)
    │   ├─ bn_inference.cu    # 추론용 BN (γ, β, running mean/var)
    │   └─ relu.cu            # in-place ReLU
    └─ runtime/
        ├─ infer_conv1_bn1_relu.cu   # 로더+실행+검증(단일 실행 바이너리)
        └─ utils.hpp                 # 파일 IO/타이머/에러체크
  ```
* **빌드/실행 스크립트**

  ```
  scripts/build_fp32.sh
  scripts/run_step2.sh
  ```
* **픽스처 생성 도구**

  ```
  tools/make_step2_fixture.py
  ```

  * 입력: `1×3×224×224` 랜덤 텐서(고정 시드)
  * 정답: PyTorch의 `relu(bn1(conv1(x)))` 결과(`1×64×112×112`)

---

## 3) 구현 개요(What we did)

### (1) Conv를 im2col + GEMM으로

* **im2col**: NCHW 입력에서 각 출력 위치의 `7×7` 윈도우를 **열(column)** 로 펼쳐 `col ∈ ℝ^{(C·7·7)×(OH·OW)}` 생성.
* **가중치 재배열**: `OIHW` → `W_col ∈ ℝ^{OC×(C·7·7)}`.
* **GEMM**: `Y = W_col · col` → `ℝ^{OC×(OH·OW)}`를 `(1, OC, OH, OW)`로 해석.

### (2) BN → ReLU

* **BN 추론**: 채널별 `y = γ·(x−μ)/√(σ²+ε) + β` (ε=1e-5, PyTorch 기본 값과 동일).
* **ReLU**: in-place로 `max(0, x)`.

### (3) 정확도/타이밍

* CUDA events로 `im2col / gemm / bn / relu` 각 구간 ms 측정.
* Host로 결과를 가져와 Torch 기준값과 **`max_abs`, `mean_abs`** 차이 계산.
* **판정**: `max_abs ≤ 1e-4` 이면 **[OK]**.

---

## 4) 사용법(How to run)

### 4.1 (한 번만) FP32 가중치 내보내기

(이미 Step 1에서 완료했다면 생략 가능)

```bash
python tools/export_resnet18.py --out exports/resnet18/fp32
```

### 4.2 픽스처(입력/정답) 생성

```bash
python tools/make_step2_fixture.py --manifest exports/resnet18/fp32
```

생성물:

```
exports/resnet18/fp32/fixtures/input.bin     # 1×3×224×224
exports/resnet18/fp32/fixtures/expected.bin  # 1×64×112×112
```

### 4.3 빌드

```bash
bash scripts/build_fp32.sh
```

빌드 산출물(기본): `build/fp32/step2_conv1_bn1_relu`

### 4.4 실행

```bash
bash scripts/run_step2.sh
```

---

## 5) 실행 로그 예시 & 해석

```
Step2 conv1->bn1->relu done
  im2col: 65.0256 ms
  gemm  : 0.519968 ms
  bn    : 0.0552 ms
  relu  : 0.028288 ms
Diff    : max_abs=2.38419e-06 mean_abs=3.43917e-08
[OK] within atol 1e-4
```

* **정확도**: `max_abs=2.38e-06` (허용치 1e-4보다 훨씬 작음) → **PyTorch 기준과 사실상 동일**.
* **성능 프로파일**: `im2col`이 **지배적 오버헤드**, `gemm/bn/relu`는 상대적으로 매우 빠름.
  → 이후 최적화/대체(implicit/direct conv, BN+ReLU fusion, cuBLAS) 타깃.

---

## 6) Acceptance Criteria 달성 여부

* ✅ **수치 일치**: `atol ≤ 1e-4` 통과 (`max_abs=2.38e-06`)
* ✅ **빌드/실행**: `build_fp32.sh`, `run_step2.sh` 정상 수행
* ✅ **로그**: 각 단계 ms 및 diff 요약 출력
* ✅ **구조화**: 커널·로더 코드가 `cpp/fp32/`에 정리

---

## 7) 성능/정확도 참고

* Conv1 파라미터: `C_in=3, C_out=64, k=7, s=2, p=3`, 출력 `OH=OW=112`.
* `col` 크기: `(3·7·7) × (112·112) = 147 × 12544 ≈ 1.85M` floats (약 7.4MB, FP32).
* **오차 기준**: FP32에서는 연산 순서/라운딩 차이를 고려해 `atol=1e-4`를 사용(충분히 엄격).
  이번 결과는 **1e-6 수준**으로 매우 우수.

---

## 8) 트러블슈팅 메모

* **g++가 `<<< >>>` 인식 실패**: 런타임 파일을 `.cu`로 빌드하거나, CMake에서 `LANGUAGE CUDA` 지정.
* **`cuda_runtime.h` not found**: `find_package(CUDAToolkit REQUIRED)` 후 `CUDA::cudart`와 `${CUDAToolkit_INCLUDE_DIRS}` 연결.
* **커널 선언/정의 불일치**: 런타임 선언에도 `extern "C" __global__` 포함(공용 헤더 `.cuh` 권장).
* **실행 파일 경로 오류**: 산출물이 `build/fp32/` 아래 생성됨 → 스크립트 경로 맞추기.

---

## 9) 로그 저장(옵션)

로그를 파일로 남기려면:

```bash
mkdir -p logs
bash scripts/run_step2.sh | tee logs/step2_$(date +'%Y%m%d_%H%M%S').log
```

또는 `scripts/run_step2.sh`에 자동 저장 기능을 추가.

---

## 10) 다음 단계(Preview)

* **Step 3**: ResNet18 **Layer1 residual block** 구현

  * `maxpool` 이후 입력(`1×64×56×56`) 기준, `layer1.0(conv-bn-relu) → layer1.1(conv-bn) + skip add`
  * Torch와 수치 일치(`atol ≤ 1e-4`), 단계별 타이밍 로그
  * 필요한 커널: 공용 3×3 conv(im2col 공용화), elementwise add(skip), (선택) BN+ReLU fusion

