# Step 1 보고서 — ResNet18 FP32 가중치 내보내기(Export) 및 매니페스트 생성

## 1) 목표(Goal)

* torchvision의 **사전학습 ResNet18**을 로드하여 **FP32 바이너리(.bin)** 파일로 내보내고,
* 모든 텐서(Conv/BN/FC)와 **전처리 파이프라인** 정보를 담은 **`manifest.json`** 을 생성한다.
* 이후 CUDA 런타임에서 manifest를 기준으로 정확하게 가중치를 로드/배치할 수 있게 한다.

---

## 2) 산출물(Deliverables)

* 디렉토리: `exports/resnet18/fp32/`

  * 개별 텐서: `*.bin` (예: `conv1.weight.bin`, `bn1.running_mean.bin`, `fc.weight.bin` …)
  * 메타 파일: `manifest.json`
* 도구 스크립트: `tools/export_resnet18.py`

---

## 3) 구현 개요(What we did)

* PyTorch로 `resnet18(weights=IMAGENET1K_V1)` 로드 → `state_dict()` 접근
* 텐서별로 **형상/레이아웃/종류(kind)** 판별:

  * Conv: **OIHW** (out, in, h, w) → `kind="conv_weight"`
  * BN: `weight/bias`는 `bn_param`, `running_mean/var`는 `bn_buffer`, 레이아웃 `O`
  * FC: `fc.weight`는 **OI**, `fc.bias`는 `O`
* 모든 텐서를 **FP32 row-major**로 `.bin` 저장 (`tofile`)
* `manifest.json`에 다음 필드를 기록:

  * `model`, `dtype=fp32`, `layout=NCHW`, `version`
  * **preprocess**: `resize=256`, `center_crop=224`, `mean/std` (ImageNet 표준)
  * **tensors**: 각 텐서의 `path`(상대경로), `shape`, `layout`, `kind`

---

## 4) 사용법(How to run)

### 4.1 준비

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision
```

### 4.2 내보내기

```bash
python tools/export_resnet18.py --out exports/resnet18/fp32
```

### 4.3 결과 예시(로그)

```
saved: conv1.weight                    [64, 3, 7, 7] -> conv1.weight.bin
saved: bn1.running_mean                [64]          -> bn1.running_mean.bin
...
saved: fc.weight                       [1000, 512]   -> fc.weight.bin

Export complete -> exports/resnet18/fp32/manifest.json
```

### 4.4 폴더 구조 예시

```
exports/resnet18/fp32/
├─ conv1.weight.bin
├─ bn1.weight.bin
├─ bn1.bias.bin
├─ bn1.running_mean.bin
├─ bn1.running_var.bin
├─ layer1.0.conv1.weight.bin
├─ ...
├─ fc.weight.bin
├─ fc.bias.bin
└─ manifest.json
```

---

## 5) manifest란? (핵심 역할)

* **가중치/전처리의 단일 출처(SSOT)**: CUDA 로더가 **이 파일 하나**만 보면,

  * 어떤 파일을
  * 어떤 **shape/레이아웃**으로
  * 어떤 연산(kind)로 쓸지
    정확히 알 수 있다.
* **언어/환경 독립**: PyTorch(파이썬) → C++/CUDA 간 경계에서 **명세서** 역할.
* **확장 용이**: 이후 INT8/INT2로 갈 때 `quant` 블록만 추가하면 동일 파이프라인 재사용 가능.

**예시(JSON 일부)**

```json
{
  "model":  "resnet18",
  "dtype":  "fp32",
  "layout": "NCHW",
  "version": 1,
  "preprocess": {
    "resize": 256, "center_crop": 224,
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
  },
  "tensors": {
    "conv1.weight": {
      "path": "conv1.weight.bin",
      "shape": [64, 3, 7, 7],
      "layout": "OIHW",
      "kind": "conv_weight"
    },
    "bn1.running_mean": {
      "path": "bn1.running_mean.bin",
      "shape": [64],
      "layout": "O",
      "kind": "bn_buffer"
    }
  }
}
```

---

## 6) 검증(Validation)

* 파일 크기 = `shape` 곱 × 4 bytes(float32)인지 확인
* manifest의 **`path`는 상대경로**, **레이아웃 일치(OIHW/OI/O)** 확인
* 전처리 파라미터(mean/std/resize/crop) 기록 → 추후 C++/CUDA 추론과 **완전 동일 전처리** 보장

---

## 7) .gitignore 권장

대용량 바이너리/빌드 산출물을 깃에서 제외:

```
# python venv
venv/
# build
build/
# exported weights & fixtures
exports/
# logs
logs/
```

(필요 시 `exports/.gitkeep`만 커밋해 폴더 존재 유지)

---

## 8) 트러블슈팅 메모

* `os.path.realpath` 인자 오류 → `os.path.join`, `os.path.relpath` 사용
* 로그 포맷: 저장 시 텐서 이름/shape/경로를 함께 출력
* Windows 경로 구분자 이슈 회피: manifest에는 **상대경로**만 기록

---

## 9) 다음 단계(Preview)

* **Step 2**: manifest 기반 **CUDA 로더 + 첫 커널(conv1→bn1→relu)** 구현

  * Torch 결과와 **수치 일치(atol ≤ 1e-4)**, **단계별 타이밍(ms)** 로그
* 이후: **Residual block**(Layer1) → **INT8/INT2** 양자화 실험으로 확장

