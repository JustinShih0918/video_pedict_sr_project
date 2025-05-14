import cv2
import numpy as np # If needed
print(f"DEBUG: Starting execution of models/super_resolution.py ({__file__})")
print("DEBUG: Imports successful in models/super_resolution.py")

# super_resolution.py

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義上採樣模型
class CNNUpsampler(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
model = CNNUpsampler(scale=2).to(device)
# model.load_state_dict(torch.load("checkpoints/upsampler.pth", map_location=device))
model.eval()

# 預處理
transform = T.Compose([
    T.ToTensor(),
])

# 主函數：輸入 NumPy 圖片 → 輸出升解析 NumPy 圖片
def upscale(frame_np, scale=2):
    with torch.no_grad():
        img = Image.fromarray(frame_np)
        tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
        # If you want to support dynamic scale, you need to re-instantiate the model here
        # model = CNNUpsampler(scale=scale).to(device)
        out = model(tensor)  # [1, 3, H*scale, W*scale]
        out_img = out.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return (out_img * 255).astype(np.uint8)
