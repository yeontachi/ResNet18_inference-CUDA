import os, argparse, torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

@torch.no_grad()
def main(manifest_dir):
    outdir = os.path.join(manifest_dir, "fixtures_step4")
    os.makedirs(outdir, exist_ok=True)

    # 1) 모델/가중치
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    # 2) Step3에서 저장했던 stem_after_maxpool 또는 layer1 block1 결과 불러오기
    x = np.fromfile(os.path.join(manifest_dir,"fixtures_step3","layer1_block1_out.bin"),
                    dtype=np.float32).reshape(1,64,56,56)
    t = torch.from_numpy(x)

    # 3) 레이어만 꺼내서 순방향
    l2_0 = m.layer2[0]
    l2_1 = m.layer2[1]

    y0 = l2_0(t)  # (1,128,28,28)
    y1 = l2_1(y0) # (1,128,28,28)

    y0.cpu().numpy().astype(np.float32).tofile(os.path.join(outdir,"layer2_block0_out.bin"))
    y1.cpu().numpy().astype(np.float32).tofile(os.path.join(outdir,"layer2_block1_out.bin"))

    print("saved:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    main(args.manifest)
