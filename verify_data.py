
import numpy as np
import os

data_dir = "data"
files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

print(f"Checking {len(files)} .npy files in {data_dir}...")

for f in files:
    path = os.path.join(data_dir, f)
    try:
        data = np.load(path)
        print(f"[OK] {f}: shape={data.shape}, dtype={data.dtype}")
    except Exception as e:
        print(f"[FAIL] {f}: {e}")
