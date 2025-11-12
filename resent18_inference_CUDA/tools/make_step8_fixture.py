import os, argparse, numpy as np, torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

def main(manifest_dir):
    # 출력 폴더
    out_dir = os.path.join(manifest_dir, "fixtures_step8")
    os.makedirs(out_dir, exist_ok=True)

    # 샘플 입력: Step2에서 쓰던 이미지 바이너리 그대로 재사용 (없으면 이미지를 하나 지정)
    #   NCHW float32, (1,3,224,224) 가정
    input_bin = os.path.join(manifest_dir, "fixtures", "input.bin")
    if not os.path.exists(input_bin):
        raise FileNotFoundError(f"not found input fixture: {input_bin}")

    x = np.fromfile(input_bin, dtype=np.float32)
    x = x.reshape(1, 3, 224, 224)
    xt = torch.from_numpy(x).contiguous()

    # 모델 로드 (torchvision 사전학습, eval)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    # stem + layer1 + layer2 + layer3 + layer4 결과를 얻어 layer4_out 저장
    with torch.no_grad():
        # stem
        y = m.relu(m.bn1(m.conv1(xt)))
        y = m.maxpool(y)
        # layer1..4
        y = m.layer1(y)
        y = m.layer2(y)
        y = m.layer3(y)
        y = m.layer4(y)

        # layer4 출력 저장: (1,512,7,7)
        y_np = y.detach().cpu().contiguous().numpy().astype(np.float32)
        (out_path := os.path.join(out_dir, "layer4_out.bin"))
        y_np.tofile(out_path)

        # e2e 검증용으로 최종 로짓도 같이 저장 (GAP + FC)
        gap = torch.mean(y, dim=(2,3))           # (1,512)
        logits = m.fc(gap)                       # (1,1000)
        (logits_path := os.path.join(out_dir, "logits.bin"))
        logits.detach().cpu().contiguous().numpy().astype(np.float32).tofile(logits_path)

    print("saved:", out_path)
    print("saved:", logits_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="exports/resnet18/fp32")
    args = ap.parse_args()
    main(args.manifest)
