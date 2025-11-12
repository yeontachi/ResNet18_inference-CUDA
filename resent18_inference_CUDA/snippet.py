import numpy as np
ref = np.fromfile("exports/resnet18/fp32/fixtures_step8/logits.bin", dtype=np.float32)  # Torch
run = np.fromfile("out/step8_logits.bin", dtype=np.float32)  
d = np.abs(ref - run)
print("max_abs:", d.max(), "mean_abs:", d.mean())
