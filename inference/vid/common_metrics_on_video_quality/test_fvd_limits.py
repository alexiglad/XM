"""
Test: does common_metrics_on_video_quality FVD support
  1. Different spatial sizes (not 64x64)?
  2. Sequences shorter than 16 frames?
"""
import sys


import torch
from calculate_fvd import calculate_fvd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 8  # batch size

def run(label, T, H, W):
    try:
        v1 = torch.zeros(B, T, 3, H, W)
        v2 = torch.ones(B, T, 3, H, W)
        result = calculate_fvd(v1, v2, device, method='styleganv', only_final=True)
        print(f"[PASS] {label:40s} -> FVD = {result['value'][0]:.2f}")
    except Exception as e:
        print(f"[FAIL] {label:40s} -> {type(e).__name__}: {e}")

print("=== Spatial size tests (T=16) ===")
run("64x64   (default)",   T=16, H=64,  W=64)
run("128x128",             T=16, H=128, W=128)
run("224x224",             T=16, H=224, W=224)
run("256x256",             T=16, H=256, W=256)
run("320x240 (non-square)",T=16, H=240, W=320)

print()
print("=== Sequence length tests (H=W=64) ===")
run("T=4  frames",  T=4,  H=64, W=64)
run("T=8  frames",  T=8,  H=64, W=64)
run("T=9  frames",  T=9,  H=64, W=64)
run("T=10 frames",  T=10, H=64, W=64)
run("T=16 frames",  T=16, H=64, W=64)
run("T=32 frames",  T=32, H=64, W=64)
