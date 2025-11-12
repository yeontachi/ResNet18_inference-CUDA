# tools/bench_fp32_vs_torch_fast.py
import os
import argparse
import time
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# --- 이미지 로딩/전처리 (ImageNet 표준) ---
_imagenet_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0,1], CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(img_path: str) -> np.ndarray:
    """이미지 파일을 읽어 (3,224,224) float32 (정규화 완료) numpy로 반환."""
    img = Image.open(img_path).convert('RGB')
    t = _imagenet_transform(img)           # torch.FloatTensor, CHW
    return t.numpy().astype('float32')     # CHW float32


# --- 모델을 layer4까지 전진시켜 [B,512,7,7] 반환 ---
@torch.no_grad()
def forward_to_layer4(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,224,224] on CUDA
    returns: [B,512,7,7] (layer4 output)
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = F.relu(x, inplace=False)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x


# --- 플랫 디렉터리에서 이미지 경로 수집 ---
def list_images_flat(root: str, limit: int = None, shuffle: bool = False):
    """
    root 아래(재귀 포함)에서 확장자가 이미지인 파일을 수집한다.
    ImageNet val 원본 구조(클래스 하위폴더)가 아닌 '평평한' 폴더여도 OK.
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = []
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    if shuffle:
        rng = np.random.default_rng(123)
        rng.shuffle(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="exports/resnet18/fp32 디렉터리")
    ap.add_argument("--bin",      required=True, help="빌드된 CUDA 실행파일 (예: build/fp32/step8_e2e)")
    ap.add_argument("--imagenet_dir", required=True, help="평평한(또는 재귀) 이미지 디렉터리")
    ap.add_argument("--limit", type=int, default=500, help="처리할 이미지 수")
    ap.add_argument("--shuffle", action="store_true", help="이미지 순서 셔플")
    ap.add_argument("--save_logits", action="store_true", help="CUDA 실행이 각 샘플 로짓을 out/ 에 저장")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)

    # 1) 이미지 경로 수집
    img_paths = list_images_flat(args.imagenet_dir, limit=args.limit, shuffle=args.shuffle)
    if not img_paths:
        print(f"[ERROR] No images found under: {args.imagenet_dir}")
        return 2
    print(f"[INFO] images={len(img_paths)} under {args.imagenet_dir}")

    # 2) Torch로 layer4 → GAP 추출하여 tmp_gaps/*.bin 저장 + gap_list.txt 생성
    repo_root = Path(__file__).resolve().parent.parent  # tools/.. -> repo root
    tmp_dir   = repo_root / "tmp_gaps"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    gap_list_path = tmp_dir / "gap_list.txt"

    t0 = time.perf_counter()
    with open(gap_list_path, "w") as f:
        for i, p in enumerate(img_paths, 1):
            x = load_image(p)                      # (3,224,224) float32 numpy
            xt = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,224,224)
            with torch.no_grad():
                l4 = forward_to_layer4(model, xt)            # (1,512,7,7)
                gap = l4.mean(dim=(2,3)).squeeze(0)          # (512,)
                gap_np = gap.detach().cpu().numpy().astype(np.float32)

            outp = tmp_dir / f"gap_{i:06d}.bin"
            gap_np.tofile(outp)
            f.write(str(outp) + "\n")

            if i % 50 == 0 or i == len(img_paths):
                print(f"  saved GAP {i}/{len(img_paths)}")

    t1 = time.perf_counter()
    print(f"[INFO] Torch GAP dump time: {(t1 - t0) * 1000:.2f} ms "
          f"(~ {(t1 - t0) * 1000 / len(img_paths):.2f} ms/img)")

    # 3) CUDA 실행파일을 단 한 번 실행: --gap_list 로 배치 처리
    cmd = [
        str(args.bin),
        "--manifest", str(args.manifest),
        "--gap_list", str(gap_list_path),
    ]
    if args.save_logits:
        cmd.append("--save_logits")

    print("[CMD]", " ".join(cmd))
    t2 = time.perf_counter()
    out = subprocess.check_output(cmd, text=True)
    t3 = time.perf_counter()

    print(out.strip())
    print(f"[WALL] E2E process time = {(t3 - t2) * 1000:.2f} ms "
          f"(~ {(t3 - t2) * 1000 / len(img_paths):.2f} ms/img)")


if __name__ == "__main__":
    raise SystemExit(main())
