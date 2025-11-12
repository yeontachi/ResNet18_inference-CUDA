# tools/make_e2e_fixtures.py
import os, numpy as np, torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANI = os.path.join(ROOT, "exports/resnet18/fp32")
INP  = os.path.join(MANI, "fixtures", "input.bin")
OUTD = os.path.join(MANI, "fixtures_e2e")
os.makedirs(OUTD, exist_ok=True)

def to_bin(path, arr):
    arr.astype("float32").tofile(path)
    print("saved:", path)

def forward_to_blocks(m, x):
    # stem: conv1→bn1→relu→maxpool
    x = m.conv1(x); x = m.bn1(x); x = torch.relu(x); x = m.maxpool(x)
    stem = x.clone()

    # layer1
    x = m.layer1(x)
    l1 = x.clone()

    # layer2
    x = m.layer2(x)
    l2 = x.clone()

    # layer3
    x = m.layer3(x)
    l3 = x.clone()

    # layer4
    x = m.layer4(x)
    l4 = x.clone()

    # GAP
    gap = torch.mean(l4, dim=(2,3))  # [N, 512]

    # logits
    logits = m.fc(gap)               # [N, 1000]
    return stem, l1, l2, l3, l4, gap, logits

def main():
    x = np.fromfile(INP, dtype=np.float32).reshape(1,3,224,224)
    x = torch.from_numpy(x).float()

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().float()
    with torch.no_grad():
        stem, l1, l2, l3, l4, gap, logits = forward_to_blocks(m, x)

    to_bin(os.path.join(OUTD, "stem_pool.bin"), stem.squeeze(0).cpu().numpy())  # [64,56,56]
    to_bin(os.path.join(OUTD, "layer1.bin"),    l1.squeeze(0).cpu().numpy())    # [64,56,56]
    to_bin(os.path.join(OUTD, "layer2.bin"),    l2.squeeze(0).cpu().numpy())    # [128,28,28]
    to_bin(os.path.join(OUTD, "layer3.bin"),    l3.squeeze(0).cpu().numpy())    # [256,14,14]
    to_bin(os.path.join(OUTD, "layer4.bin"),    l4.squeeze(0).cpu().numpy())    # [512,7,7]
    to_bin(os.path.join(OUTD, "gap.bin"),       gap.squeeze(0).cpu().numpy())   # [512]
    to_bin(os.path.join(OUTD, "logits.bin"),    logits.squeeze(0).cpu().numpy())# [1000]

if __name__ == "__main__":
    main()
