# tools/diag_e2e_compare.py
import os, numpy as np
from numpy.linalg import norm

CKPTS = [
    ("stem_pool.bin", (64,56,56)),
    ("layer1.bin",    (64,56,56)),
    ("layer2.bin",    (128,28,28)),
    ("layer3.bin",    (256,14,14)),
    ("layer4.bin",    (512,7,7)),
    ("gap.bin",       (512,)),
    ("logits.bin",    (1000,)),
]

def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na*nb))

def diff(a, b):
    d = np.abs(a - b).reshape(-1)
    return float(d.max()), float(d.mean())

def load(path, shape):
    x = np.fromfile(path, dtype=np.float32)
    if x.size != np.prod(shape):
        raise RuntimeError(f"size mismatch for {path}: got {x.size}, want {np.prod(shape)}")
    return x.reshape(shape)

def main(torch_dir, cuda_dir):
    print(f"[COMPARE] torch_dir={torch_dir}")
    print(f"[COMPARE]  cuda_dir={cuda_dir}")
    for name, shape in CKPTS:
        t = load(os.path.join(torch_dir, name), shape)
        c = load(os.path.join(cuda_dir,  name), shape)
        mx, mn = diff(t, c)
        cs = cosine(t, c)
        print(f"{name:<14}  max_abs={mx:.6g}  mean_abs={mn:.6g}  cosine={cs:.6f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch_dir", required=True)
    ap.add_argument("--cuda_dir",  required=True)
    args = ap.parse_args()
    main(args.torch_dir, args.cuda_dir)
