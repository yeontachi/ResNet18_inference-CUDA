import os, json, argparse, numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

def load_manifest(root):
    with open(os.path.join(root, "manifest.json"), "r") as f:
        return json.load(f)

def save_bin(path, tensor):
    arr = tensor.detach().cpu().contiguous().float().numpy()
    arr.tofile(path)

@torch.no_grad()
def main(manifest_dir):
    mani = load_manifest(manifest_dir)

    # Step2에서 쓰던 입력 그대로 사용 (전처리 완료된 NCHW float32)
    in_path = os.path.join(manifest_dir, "fixtures", "input.bin")
    x = np.fromfile(in_path, dtype=np.float32)
    x = torch.from_numpy(x).reshape(1, 3, 224, 224)

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    # stem
    y = m.conv1(x)
    y = m.bn1(y)
    y = m.relu(y)
    y = m.maxpool(y)

    # layer1 -> layer2
    y = m.layer1(y)
    y = m.layer2(y)

    out_dir = os.path.join(manifest_dir, "fixtures_step5")
    os.makedirs(out_dir, exist_ok=True)

    # layer2 출력 저장 (Step5 입력)
    save_bin(os.path.join(out_dir, "layer2_out.bin"), y)

    # layer3 block0 (downsample 포함, 블록 끝의 ReLU까지)
    y_b0 = m.layer3[0](y)
    save_bin(os.path.join(out_dir, "layer3_block0_out.bin"), y_b0)

    # layer3 block1 (identity, 블록 끝의 ReLU까지)
    y_b1 = m.layer3[1](y_b0)
    save_bin(os.path.join(out_dir, "layer3_block1_out.bin"), y_b1)

    print("saved:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="exports/.../fp32")
    args = ap.parse_args()
    main(args.manifest)
