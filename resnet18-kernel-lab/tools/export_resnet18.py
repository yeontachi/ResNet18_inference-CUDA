import os, json, argparse, numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

# -------------------------------------------------------------
# [함수] 텐서를 binary(.bin) 파일로 저장
# -------------------------------------------------------------
def save_bin(path, tensor):
    # 텐서를 CPU로 옮기고, 연속된(float32) numpy 배열로 변환
    arr = tensor.detach().cpu().contiguous().float().numpy()
    # numpy.tofile()을 이용해 raw binary 형식으로 저장
    arr.tofile(path)

# -------------------------------------------------------------
# [함수] 각 텐서의 메타데이터(shape, layout, kind) 생성
# -------------------------------------------------------------
def tensor_meta(name, t):
    shape = list(t.shape)
    kind = "param"   # 기본 타입
    layout = None    # 데이터 레이아웃(OIHW 등)

    # Conv2d weight: (out, in, h, w)
    if "conv" in name and "weight" in name and t.ndim == 4:
        layout = "OIHW"
        kind = "conv_weight"

    # BatchNorm의 running_mean / running_var 버퍼
    elif any(k in name for k in ["running_mean", "running_var"]):
        kind = "bn_buffer"
        layout = "O"

    # BatchNorm의 weight / bias (학습 파라미터)
    elif "bn" in name:
        kind = "bn_param"
        layout = "O"

    # Fully Connected (FC) layer의 weight
    elif name.endswith("fc.weight") and t.ndim == 2:
        kind = "fc_weight"
        layout = "OI"  # (out, in)

    # FC layer의 bias
    elif name.endswith("fc.bias"):
        kind = "fc_bias"
        layout = "O"

    # 기타 파라미터 (예: num_batches_tracked)
    else:
        layout = "auto"

    # shape, layout, kind를 딕셔너리 형태로 반환
    return {"shape": shape, "layout": layout, "kind": kind}

# -------------------------------------------------------------
# [함수] 메인 익스포트 로직
# -------------------------------------------------------------
def main(out_dir):
    # 출력 폴더 생성 (이미 있으면 통과)
    os.makedirs(out_dir, exist_ok=True)

    # torchvision 사전학습 ResNet18 모델 로드 (FP32)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    # state_dict()은 모든 weight/buffer 텐서를 key-value 형태로 반환
    sd = m.state_dict()

    # manifest 기본 구조 설정
    manifest = {
        "model":  "resnet18",
        "dtype":  "fp32",
        "layout": "NCHW",
        "version": 1,
        "preprocess": {  # 이미지 전처리 정보
            "resize": 256,
            "center_crop": 224,
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225]
        },
        "tensors": {}  # 여기에 각 weight 메타데이터 추가됨
    }

    # ---------------------------------------------------------
    # 모든 텐서 순회하며 binary 저장 및 manifest 기록
    # ---------------------------------------------------------
    for name, t in sd.items():
        # 예: conv1.weight → exports/resnet18/fp32/conv1.weight.bin
        subdir = os.path.join(out_dir, os.path.dirname(name))
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.bin")

        # 실제 binary 파일 저장
        save_bin(path, t)

        # 텐서 메타데이터 생성
        meta = tensor_meta(name, t)

        # manifest 내에서 경로는 out_dir 기준 상대경로로 기록
        meta["path"] = os.path.relpath(path, out_dir)

        # manifest["tensors"] 딕셔너리에 추가
        manifest["tensors"][name] = meta

        # 콘솔 출력 (진행 로그)
        print(f"saved: {name:30s}   {meta['shape']} -> {meta['path']}")

    # ---------------------------------------------------------
    # manifest.json 저장
    # ---------------------------------------------------------
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 완료 메시지
    print("\nExport complete ->", os.path.join(out_dir, "manifest.json"))

# -------------------------------------------------------------
# [main 진입점]
# -------------------------------------------------------------
if __name__ == "__main__":
    # --out 인자로 출력 폴더 지정 (예: exports/resnet18/fp32)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory (e.g., exports/resnet18/fp32)")
    args = ap.parse_args()

    # main() 실행
    main(args.out)
