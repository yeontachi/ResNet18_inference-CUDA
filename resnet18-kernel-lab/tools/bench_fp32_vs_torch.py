# tools/bench_fp32_vs_torch.py
import os
import sys
import time
import argparse
import glob
import subprocess
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


def list_images_flat(root, limit=None, shuffle=False):
    exts = ["*.JPEG", "*.JPG", "*.jpg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root, e)))
    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(paths)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def load_image(img_path):
    # Torchvision 공식 ResNet18 전처리 파이프라인과 동일(Resize 256, CenterCrop 224, Normalize)
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()  # (Resize, CenterCrop, ToTensor, Normalize)
    img = Image.open(img_path).convert('RGB')
    x = preprocess(img)  # [3,224,224], float32
    return x.numpy()


@torch.no_grad()
def forward_to_layer4(model, x_tensor):
    """
    x_tensor: torch.Tensor [1,3,224,224] on CUDA
    return: layer4 output [1,512,7,7] on CPU float32 (NCHW)
    """
    m = model
    y = m.conv1(x_tensor); y = m.bn1(y); y = m.relu(y); y = m.maxpool(y)
    y = m.layer1(y)
    y = m.layer2(y)
    y = m.layer3(y)
    y = m.layer4(y)  # [1,512,7,7]
    return y.detach().cpu().to(torch.float32)


def save_f32_bin(path, arr):
    arr.astype('float32').tofile(path)


def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aa = np.dot(a, a)
    bb = np.dot(b, b)
    ab = np.dot(a, b)
    if aa < eps or bb < eps:
        return 0.0
    return float(ab / (np.sqrt(aa) * np.sqrt(bb)))

# ... import 동일 ...
from torchvision.models import resnet18, ResNet18_Weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bin",      required=True)
    ap.add_argument("--imagenet_dir", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters",  type=int, default=20)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--save_log", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    # manifest와 동일 버전으로 고정
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(device)

    img_paths = list_images_flat(args.imagenet_dir, limit=args.limit, shuffle=args.shuffle)
    print(f"[INFO] images={len(img_paths)} under {args.imagenet_dir}")

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_e2e")
    os.makedirs(tmp_dir, exist_ok=True)

    # === Torch FC 가중치/바이어스 덤프 (한 번만) ===
    W = model.fc.weight.detach().cpu().to(torch.float32).numpy().reshape(-1)  # [1000*512]
    b = model.fc.bias.detach().cpu().to(torch.float32).numpy().reshape(-1)    # [1000]
    fcw_path = os.path.join(tmp_dir, "fc.weight.bin")
    fcb_path = os.path.join(tmp_dir, "fc.bias.bin")
    W.astype('float32').tofile(fcw_path)
    b.astype('float32').tofile(fcb_path)

    agree = 0; torch_times=[]; cuda_times=[]; cosines=[]
    for idx, img_path in enumerate(img_paths, 1):
        x_np = load_image(img_path)                 # (3,224,224) float32, mean/std 정규화 포함
        x_t  = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1,3,224,224]

        # Torch full logits
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_t = model(x_t).squeeze(0).cpu().numpy()     # [1000]
        t1 = time.perf_counter()
        torch_ms = (t1 - t0) * 1000.0
        top1_torch = int(np.argmax(logits_t))

        # Torch layer4 & GAP(=avg over H,W)
        with torch.no_grad():
            l4 = forward_to_layer4(model, x_t)     # [1,512,7,7]
            gap = l4.mean(dim=(2,3)).squeeze(0).cpu().numpy()  # [512]
        gap_bin = os.path.join(tmp_dir, "gap.bin")
        gap.astype('float32').tofile(gap_bin)

        # CUDA E2E: GAP 스킵, FC만 동일 가중치로 실행
        cmd = [args.bin, "--manifest", args.manifest,
               "--gap", gap_bin, "--fc_weight", fcw_path, "--fc_bias", fcb_path]
        c0 = time.perf_counter()
        _ = subprocess.check_output(cmd, text=True)
        c1 = time.perf_counter()
        cuda_ms = (c1 - c0) * 1000.0

        # CUDA logits 읽기
        logits_bin = os.path.join(os.path.dirname(__file__), "..", "out", "step8_logits.bin")
        logits_cuda = np.fromfile(logits_bin, dtype=np.float32)

        # 비교
        if int(np.argmax(logits_cuda)) == top1_torch: agree += 1
        cs = cosine(logits_t, logits_cuda)
        torch_times.append(torch_ms); cuda_times.append(cuda_ms); cosines.append(cs)

        if idx % 50 == 0 or idx == len(img_paths):
            print(f"[{idx}/{len(img_paths)}] agree_top1={agree} ({agree/idx:.2%}) "
                  f"torch={np.mean(torch_times):.2f} ms, cuda={np.mean(cuda_times):.2f} ms, "
                  f"cos={np.mean(cosines):.4f}")

    print(f"[BENCH] images={len(img_paths)}, agree_top1={agree} ({agree/len(img_paths):.2%})")
    print(f"         torch={np.mean(torch_times):.2f} ms, cuda={np.mean(cuda_times):.2f} ms, "
          f"speedup={np.mean(torch_times)/np.mean(cuda_times):.2f}x")
    print(f"         logits_cosine={np.mean(cosines):.4f}")

    # 최종 요약
    avg_torch = float(np.mean(torch_times)) if torch_times else 0.0
    avg_cuda  = float(np.mean(cuda_times)) if cuda_times else 0.0
    avg_cos   = float(np.mean(cosines)) if cosines else 0.0
    speedup   = (avg_torch/avg_cuda) if avg_cuda>0 else 0.0

    print(f"[BENCH] images={len(img_paths)}, agree_top1={agree} ({agree/len(img_paths):.2%})")
    print(f"         torch={avg_torch:.2f} ms, cuda={avg_cuda:.2f} ms, speedup={speedup:.2f}x")
    print(f"         logits_cosine={avg_cos:.4f}")

    if args.save_log:
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
        # 간단 저장
        with open(os.path.join(os.path.dirname(__file__), "..", "logs", "step9_summary.txt"), "w") as f:
            f.write(f"images={len(img_paths)}\n")
            f.write(f"agree_top1={agree} ({agree/len(img_paths):.2%})\n")
            f.write(f"torch_ms={avg_torch:.2f}\n")
            f.write(f"cuda_ms={avg_cuda:.2f}\n")
            f.write(f"speedup={speedup:.2f}x\n")
            f.write(f"cosine={avg_cos:.4f}\n")


if __name__ == "__main__":
    main()
