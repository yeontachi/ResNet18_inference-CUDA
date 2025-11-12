# tools/preprocess_to_bin.py
import argparse, os, numpy as np
from PIL import Image

MEAN = [0.485, 0.456, 0.406]  # ImageNet (IMAGENET1K_V1)
STD  = [0.229, 0.224, 0.225]

def resize_shorter_side(img: Image.Image, size=256):
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(round(h * size / w))
    else:
        new_h = size
        new_w = int(round(w * size / h))
    return img.resize((new_w, new_h), Image.BILINEAR)

def center_crop(img: Image.Image, size=224):
    w, h = img.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return img.crop((left, top, left+size, top+size))

def to_nchw_float32(img: Image.Image):
    # PIL RGB -> numpy HWC [0..255] -> [0..1] -> Normalize -> NCHW float32
    x = np.array(img).astype(np.float32) / 255.0  # HWC, 0..1
    # normalize in HWC
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    # HWC->CHW
    x = np.transpose(x, (2,0,1))  # CHW
    # add batch (N=1): NCHW
    x = x[None, ...].copy()
    return x  # (1,3,224,224)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path (jpg/png)")
    ap.add_argument("--out",   required=True, help="output .bin (NCHW float32)")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    img = resize_shorter_side(img, 256)
    img = center_crop(img, 224)
    x   = to_nchw_float32(img)  # (1,3,224,224)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    x.astype("float32").tofile(args.out)
    print(f"saved: {args.out} shape={x.shape}")

if __name__ == "__main__":
    main()
