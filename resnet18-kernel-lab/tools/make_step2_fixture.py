# tools/make_step2_fixture.py
import os, numpy as np, torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MANI = os.path.join(ROOT, "exports/resnet18/fp32")
os.makedirs(os.path.join(MANI, "fixtures"), exist_ok=True)

# 1) 전처리 파이프라인: 가급적 weights.transforms() 사용
try:
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()   # Resize(256)→CenterCrop(224)→ToTensor→Normalize
except Exception:
    # 혹시 매우 구버전이면 명시 상수로 대체
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    weights = ResNet18_Weights.IMAGENET1K_V1  # 모델 로딩용

# 2) 입력 이미지 준비
img_path = os.path.join(ROOT, "data", "sample.jpg")
if not os.path.exists(img_path):
    Image.new("RGB", (256, 256), (128, 128, 128)).save(img_path)  # 없으면 회색 더미 생성
img = Image.open(img_path).convert("RGB")

# 3) input.bin 저장 (NCHW float32)
x = preprocess(img).unsqueeze(0).float()  # [1,3,224,224]
x.cpu().numpy().astype("float32").tofile(os.path.join(MANI, "fixtures", "input.bin"))

# 4) Torch 모델 로드
m = resnet18(weights=weights).eval().float()

# 5) conv1 / bn1 / relu 출력 각각 저장
with torch.no_grad():
    y_conv1 = m.conv1(x)                                  # [1,64,112,112]
    y_conv1.cpu().numpy().astype("float32").tofile(os.path.join(MANI, "fixtures", "step2_conv1.bin"))

    y_bn1 = m.bn1(y_conv1)                                # BN (eps=1e-5, running_var)
    y_bn1.cpu().numpy().astype("float32").tofile(os.path.join(MANI, "fixtures", "step2_bn1.bin"))

    y_relu = torch.relu(y_bn1)                            # 최종 expected
    y_relu.cpu().numpy().astype("float32").tofile(os.path.join(MANI, "fixtures", "expected.bin"))

print("Wrote fixtures: input.bin, step2_conv1.bin, step2_bn1.bin, expected.bin")
