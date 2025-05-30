# utils/device.py

import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU（用不到在 Mac）
    else:
        return torch.device("cpu")
