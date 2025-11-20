import os
import random
import time
import json
import numpy as np
import torch

def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(path, obj):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)