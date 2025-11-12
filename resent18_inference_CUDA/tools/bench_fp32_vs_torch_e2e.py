# tools/bench_fp32_vs_torch_e2e.py
import os, time, argparse, glob, subprocess, re, random, sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# --- 이미지 전처리: Resize(short=256) -> CenterCrop(224) -> ToTensor -> Normalize(IMNET) ---
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def center_crop(img, size=224):
    w, h = img.size
    th, tw = size, size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return img.crop((j, i, j + tw, i + th))

def resize_short(img, short=256):
    w, h = img.size
    if w < h:
        nw = short
        nh = int(round(h * short / w))
    else:
        nh = short
        nw = int(round(w * short / h))
    return img.resize((nw, nh), Image.BILINEAR)

def preprocess_to_nchw_f32(img_path):
    img = Image.open(img_path).convert("RGB")
    img = resize_short(img, 256)
    img = center_crop(img, 224)
    x = np.asarray(img, dtype=np.float32) / 255.0   # HWC, [0,1]
    x = (x - IMNET_MEAN) / IMNET_STD               # normalize
    x = np.transpose(x, (2, 0, 1))                 # CHW
    return x.astype(np.float32)                    # (3,224,224)

def list_images_flat(root, limit=None, shuffle=False):
    exts = ["*.JPEG", "*.jpeg", "*.jpg", "*.png"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root, e)))
    if shuffle:
        random.seed(1234); random.shuffle(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths

TOP1_RE = re.compile(r"top-1 class index\s*=\s*(\d+)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bin",      required=True)
    ap.add_argument("--imagenet_dir", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--save_log", action="store_true")
    args = ap.parse_args()

    img_paths = list_images_flat(args.imagenet_dir, limit=args.limit, shuffle=args.shuffle)
    if not img_paths:
        print(f"[ERROR] No images found under: {args.imagenet_dir}")
        sys.exit(1)
    print(f"[INFO] images={len(img_paths)} under {args.imagenet_dir}")

    # Torch 모델 고정 (IMAGENET1K_V1: 우리가 export한 가중치와 동일 버전 전제)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(device).float()

    # 임시 입력 저장 디렉터리
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_e2e_full")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_input = os.path.join(tmp_dir, "input.bin")  # 매 스텝마다 덮어쓰기

    agree = 0
    torch_times = []
    cuda_times  = []
    cosines     = []

    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    for idx, p in enumerate(img_paths, 1):
        # --- 공통 전처리 (CUDA & Torch 모두 동일 텐서 사용) ---
        x = preprocess_to_nchw_f32(p)             # (3,224,224), f32, NCHW without batch
        x.tofile(tmp_input)                       # CUDA 실행파일 입력으로 저장
        xt = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,3,224,224]

        # --- Torch forward (full pipeline) ---
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_t = model(xt).squeeze(0).cpu().numpy()
        t1 = time.perf_counter()
        torch_ms = (t1 - t0) * 1000.0
        top1_t = int(np.argmax(logits_t))

        # --- CUDA 실행파일 호출 (풀 파이프라인 E2E) ---
        cmd = [args.bin, "--manifest", args.manifest, "--input", tmp_input]
        c0 = time.perf_counter()
        out = subprocess.check_output(cmd, text=True)
        c1 = time.perf_counter()
        cuda_ms = (c1 - c0) * 1000.0

        m = TOP1_RE.search(out)
        if not m:
            print("[WARN] Could not parse top-1 from CUDA output; raw:\n", out)
            top1_c = -1
        else:
            top1_c = int(m.group(1))

        # --- 통계 ---
        if top1_c == top1_t: agree += 1
        # (선택) 로짓 비교를 위해서는 step8_e2e에서 logits 저장을 추가해야 함. 여기선 top1만 비교.
        torch_times.append(torch_ms); cuda_times.append(cuda_ms)

        if idx % 50 == 0 or idx == len(img_paths):
            print(f"[{idx}/{len(img_paths)}] agree_top1={agree} ({agree/idx:.2%}) "
                  f"torch={np.mean(torch_times):.2f} ms, cuda={np.mean(cuda_times):.2f} ms")

    print("-----")
    print(f"images={len(img_paths)}")
    print(f"agree_top1={agree} ({agree/len(img_paths):.2%})")
    print(f"torch_ms={np.mean(torch_times):.2f}")
    print(f"cuda_ms={np.mean(cuda_times):.2f}")
    print(f"speedup={np.mean(torch_times)/np.mean(cuda_times):.2f}x")

if __name__ == "__main__":
    main()
