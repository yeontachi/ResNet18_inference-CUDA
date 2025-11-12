import os, json, argparse, numpy as np, torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

torch.manual_seed(0)

def load_state(m):  # pretrained resnet18
    w = ResNet18_Weights.IMAGENET1K_V1
    m.load_state_dict(resnet18(weights=w).state_dict())
    m.eval()
    return m

@torch.no_grad()
def stem_maxpool_only(x, m):  # conv1→bn1→relu→maxpool
    y = m.conv1(x); y = m.bn1(y); y = F.relu(y); y = m.maxpool(y)
    return y

@torch.no_grad()
def layer1_block(y, m):       # layer1.0 → layer1.1 (+skip)
    y0 = m.layer1[0].conv1(y); y0 = m.layer1[0].bn1(y0); y0 = F.relu(y0)
    y0 = m.layer1[0].conv2(y0); y0 = m.layer1[0].bn2(y0)
    out = F.relu(y0 + y)      # skip add + relu
    # 이어서 layer1.1
    y1 = m.layer1[1].conv1(out); y1 = m.layer1[1].bn1(y1); y1 = F.relu(y1)
    y1 = m.layer1[1].conv2(y1); y1 = m.layer1[1].bn2(y1)
    out2 = F.relu(y1 + out)   # skip add + relu
    return out, out2          # (블록0 출력, 블록1 출력 최종)

def main(manifest, outdir="exports/resnet18/fp32/fixtures_step3"):
    with open(os.path.join(manifest, "manifest.json")) as f:
        mani = json.load(f)
    os.makedirs(outdir, exist_ok=True)

    # 입력은 Step2와 동일 원본(1×3×224×224) 사용
    x = torch.from_numpy(np.fromfile(
        os.path.join(manifest, "fixtures/input.bin"), dtype=np.float32
    ).reshape(1,3,224,224))

    m = load_state(resnet18())
    stem = stem_maxpool_only(x, m)      # 1×64×56×56
    b0, b1 = layer1_block(stem, m)      # 두 블록 출력

    stem.cpu().numpy().tofile(os.path.join(outdir, "stem_after_maxpool.bin"))
    b0.cpu().numpy().tofile(os.path.join(outdir, "layer1_block0_out.bin"))
    b1.cpu().numpy().tofile(os.path.join(outdir, "layer1_block1_out.bin"))
    print("saved:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="exports/resnet18/fp32/fixtures_step3")
    args = ap.parse_args()
    main(args.manifest, args.out)
