#!/usr/bin/env python3
import os, json, argparse, numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from PIL import Image

def save_bin(path, arr: np.ndarray):
    arr.astype(np.float32).tofile(path)

@torch.no_grad()
def main(manifest_root: str):
    os.makedirs(os.path.join(manifest_root, "fixtures_step7"), exist_ok=True)

    # 1) 모델 로드 (fp32, eval)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    # 2) 전처리 (manifest의 preprocess와 동일)
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # 샘플 이미지: 없으면 랜덤 텐서 사용
    sample_img = None
    candidates = [
        os.path.join(manifest_root, "..", "..", "data", "samples", "dog.jpg"),
        os.path.join("data", "samples", "dog.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            sample_img = Image.open(p).convert("RGB")
            break

    if sample_img is None:
        x = torch.randn(1,3,224,224)
    else:
        x = tfm(sample_img).unsqueeze(0)  # [1,3,224,224]

    # 3) layer4까지 통과 (stem+layer1..4)
    with torch.no_grad():
        # torchvision resnet18 구조에 맞춰 forward의 중간까지 수동 구성
        y = m.relu(m.bn1(m.conv1(x)))
        y = m.maxpool(y)
        y = m.layer1(y)
        y = m.layer2(y)
        y = m.layer3(y)
        y4 = m.layer4(y)   # [1,512,7,7]

        # GAP → FC → Softmax (Torch 기준값)
        gap = torch.mean(y4, dim=(2,3))              # [1,512]
        logits = m.fc(gap)                            # [1,1000]
        prob = torch.softmax(logits, dim=1)           # [1,1000]

    # 4) 저장
    outdir = os.path.join(manifest_root, "fixtures_step7")
    save_bin(os.path.join(outdir, "after_layer4.bin"), y4.squeeze(0).contiguous().cpu().numpy())   # [512,7,7]
    save_bin(os.path.join(outdir, "fc_logits.bin"),    logits.squeeze(0).contiguous().cpu().numpy())# [1000]
    save_bin(os.path.join(outdir, "prob_softmax.bin"), prob.squeeze(0).contiguous().cpu().numpy())  # [1000]

    # 메타 (참고용)
    meta = {"C": int(y4.shape[1]), "H": int(y4.shape[2]), "W": int(y4.shape[3]), "num_classes": 1000}
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("saved:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="exports/resnet18/fp32")
    args = ap.parse_args()
    main(args.manifest)
