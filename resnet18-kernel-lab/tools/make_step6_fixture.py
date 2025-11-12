#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

def save_bin(path: str, arr: np.ndarray):
    arr.astype(np.float32, copy=False).tofile(path)

def load_bin(path: str, shape):
    n = int(np.prod(shape))
    data = np.fromfile(path, dtype=np.float32, count=n)
    if data.size != n:
        raise RuntimeError(f"load_bin: size mismatch for {path}, expect {n}, got {data.size}")
    return data.reshape(shape)

@torch.no_grad()
def main(manifest_root: str):
    # 경로 준비
    mani = manifest_root
    fx2_dir = os.path.join(mani, "fixtures")             # step2 입력 재사용
    out_dir = os.path.join(mani, "fixtures_step6")
    os.makedirs(out_dir, exist_ok=True)

    # 입력 로드 (1,3,224,224) NCHW float32
    x_path = os.path.join(fx2_dir, "input.bin")
    x = load_bin(x_path, (1, 3, 224, 224))
    x_t = torch.from_numpy(x).float().cuda()

    # Torch 모델 로드 (FP32, eval)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().cuda()

    # Forward: stem → layer1 → layer2 → layer3
    # torchvision의 resnet18 forward 기준:
    # stem: conv1 → bn1 → relu → maxpool
    with torch.no_grad():
        o = m.conv1(x_t)
        o = m.bn1(o)
        o = torch.relu(o)
        o = torch.max_pool2d(o, kernel_size=3, stride=2, padding=1)  # 112→56

        o = m.layer1(o)   # 56x56, 64ch
        o = m.layer2(o)   # 28x28, 128ch
        o = m.layer3(o)   # 14x14, 256ch

        layer3_out = o.detach().float().cpu().numpy()  # (1,256,14,14)
        save_bin(os.path.join(out_dir, "layer3_out.bin"), layer3_out)

        # layer4: 두 개의 BasicBlock (downsample 포함, 14→7, 256→512)
        b0 = m.layer4[0](o)
        save_bin(os.path.join(out_dir, "layer4_block0_out.bin"),
                 b0.detach().float().cpu().numpy())

        b1 = m.layer4[1](b0)
        save_bin(os.path.join(out_dir, "layer4_block1_out.bin"),
                 b1.detach().float().cpu().numpy())

    print("saved:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="exports/resnet18/fp32 (manifest root)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    main(args.manifest)
