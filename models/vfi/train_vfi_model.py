# train_interpolation.py

import os
import glob
import zipfile
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
from PIL import Image
from torchvision import transforms

# ── 1. Import model & 推論函式 （剛剛在 interpolation.py 裡） ──────────────────

from vfi import (
    ImprovedInterpNet,
    compute_flow,
    interpolate_and_evaluate_full6,
    run_private_testset,
    run_public_testset
)

# Device util

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"[Train] Using device: {device}")
print("torch.cuda.is_available():", torch.cuda.is_available())

# data paths

# traing
TRAIN_ROOT = os.path.join("data", "topic4_release", "train")

# optical flow chart
FLOW_CACHE_DIR = os.path.join("data", "topic4_release", "flow_cache")
os.makedirs(FLOW_CACHE_DIR, exist_ok=True)

# checkpoint & logs 資料夾
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOSS_LOG_PATH = "logs/interp_loss_log.txt"


# collect Training paths

def collect_paths_v2(res_type: str = "High_Resolution", limit: int | None = None):

    base_path = os.path.join(TRAIN_ROOT, res_type)
    video_folders = sorted(glob.glob(os.path.join(base_path, "*", "*")))
    path_list = []

    for folder in video_folders:
        frame_paths = [os.path.join(folder, f"im{i}.png") for i in range(1, 8)]
        if not all(os.path.exists(p) for p in frame_paths):
            continue

        inputs_6 = [frame_paths[i] for i in (0, 1, 2, 4, 5, 6)]
        gt_4 = frame_paths[3]
        path_list.append((inputs_6, gt_4))

        if limit is not None and len(path_list) >= limit:
            break

    return path_list


# Dataset Class

image_size = (256, 448)

class InterpolationDatasetWithFlowCache(Dataset):

    def __init__(self, cached_list: list[tuple[list[str], str, str]]):
        super().__init__()
        self.cache = cached_list
        self.H, self.W = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.H, self.W)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        inputs, gt_path, flow_paths = self.cache[idx]
        # 1) 處理 6 幀影格（固定不變）
        frames = [ self.transform(Image.open(p).convert('RGB')) for p in inputs ]
        # 2) 處理 flows：現在 flow_paths 長度變 6 而非 5
        flows = []
        # 將原本迭代的部分替換成直接讀取
        arr = np.load(flow_paths)  # flow_paths 為單一檔案路徑字串
        fx = arr[..., 0] / (arr.shape[1] - 1)
        fy = arr[..., 1] / (arr.shape[0] - 1)
        flow = np.stack([fx, fy], -1)
        flow_resized = cv2.resize(flow, (self.W, self.H))
        flows = [torch.from_numpy(flow_resized).permute(2, 0, 1).float()]
        # 3) 時間編碼 channels 不變
        temp_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        t_chs = [ torch.full((1, self.H, self.W), t, dtype=torch.float32) for t in temp_vals ]

        # 最後拼接：原本 frames (6×3=18) + flows (5組＝10通道) + time (6×1=6)
        # 變成 frames (18) + flows (6組＝12通道) + time (6通道) = 總共 36 通道
        x = torch.cat(frames + flows + t_chs, dim=0)

        # 4) 處理 Target
        y = transforms.Resize((self.H, self.W))(Image.open(gt_path).convert('RGB'))
        y = transforms.ToTensor()(y)
        return x, y


# Precompute & Cache Flow

def preprocess_and_cache_flow(
    path_list: list[tuple[list[str], str]],
    cache_dir: str = FLOW_CACHE_DIR
) -> list[tuple[list[str], str, str]]:

    cached = []
    for inputs_6, gt_4 in tqdm(path_list, desc="Caching optical flows"):
        im3_path = inputs_6[2]
        im5_path = inputs_6[3]
        name3 = os.path.splitext(os.path.basename(im3_path))[0]
        name5 = os.path.splitext(os.path.basename(im5_path))[0]
        flow_file = os.path.join(cache_dir, f"{name3}_{name5}.npy")

        if not os.path.exists(flow_file):
            img3 = cv2.imread(im3_path)
            img5 = cv2.imread(im5_path)
            if img3 is None or img5 is None:
                continue
            flow = compute_flow(img3, img5)  # [H_org, W_org, 2]
            np.save(flow_file, flow)

        cached.append((inputs_6, gt_4, flow_file))

    return cached

# Training loop

if __name__ == "__main__":

    # 收集 High & Low resolution 的路徑
    # 如果想跑全部資料，把 limit=None；否則 limit=3000 (和原本示範一致)
    limit = None

    print("[Train] Collecting High_Resolution paths ...")
    high_paths = collect_paths_v2("High_Resolution", limit=limit)
    print(f"[Train] Found {len(high_paths)} high‐res samples.")

    print("[Train] Collecting Low_Resolution paths ...")
    low_paths = collect_paths_v2("Low_Resolution", limit=limit)
    print(f"[Train] Found {len(low_paths)} low‐res samples.")

    # (B) Step 2: Precompute & cache flow
    print("[Train] Preprocessing & caching flows for HIGH set ...")
    cached_high = preprocess_and_cache_flow(high_paths)

    print("[Train] Preprocessing & caching flows for LOW set ...")
    cached_low = preprocess_and_cache_flow(low_paths)

    # 合併
    cached_all = cached_high + cached_low
    print(f"[Train] Total cached samples: {len(cached_all)}")

    # (C) Step 3: 建立 Dataset & DataLoader
    dataset_all = InterpolationDatasetWithFlowCache(cached_all)
    dataloader = DataLoader(
        dataset_all,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # (D) Step 4: Instantiate Model & Optimizer & Loss
    model = ImprovedInterpNet(in_ch=26).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    base_loss_fn = nn.L1Loss(reduction="none")

    epochs = 30

    # 準備 loss log 檔案
    with open(LOSS_LOG_PATH, "w") as f:
        f.write("epoch,avg_loss\n")

    def motion_weighted_loss(preds, targets, flows, base_loss):
        motion_mag = torch.norm(flows, dim=1, keepdim=True)  # shape: [B,1,H,W]
        weights = torch.tanh(motion_mag * 10) + 1.0  # [1,2] 範圍內的權重
        loss = base_loss(preds, targets)
        if isinstance(loss, torch.Tensor) and loss.shape == preds.shape:
            loss = (loss * weights).mean()
        return loss

    # (E) Step 5: Training Loop
    print("[Train] Start training ...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)   # [B,20,256,448]
            targets = targets.to(device) # [B,3,256,448]

            preds = model(inputs)        # [B,3,256,448]

            flows = inputs[:, -2:, :, :] # [B,2,256,448]

            loss = motion_weighted_loss(preds, targets, flows, base_loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Train] Epoch {epoch+1}/{epochs} => Avg Loss: {avg_loss:.6f}")

        # 寫進 log
        with open(LOSS_LOG_PATH, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f}\n")

        # 每 5 個 epoch or 最後一個 epoch 存一次 checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            ckpt_path = f"checkpoints/interp_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Train] Checkpoint saved: {ckpt_path}")

    # (F) Step 6: Save final model
    final_ckpt = "checkpoints/interp_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"[Train] Training complete. Final model -> {final_ckpt}")

    # (G) Step 7: Training
    TOPIC4_RELEASE_PATH = os.path.join("data", "topic4_release")
    PRIVATE_TEST_DIR = os.path.join(TOPIC4_RELEASE_PATH, "private_test_set")
    PUBLIC_TEST_DIR  = os.path.join(TOPIC4_RELEASE_PATH, "public_test_set")
    PRIVATE_OUTPUT_DIR = os.path.join(TOPIC4_RELEASE_PATH, "private_test_output")
    PUBLIC_OUTPUT_DIR  = os.path.join(TOPIC4_RELEASE_PATH, "public_test_output")

    ## print("\n>>> [Train] Now running PRIVATE test set ...")
    ## run_private_testset(model, PRIVATE_TEST_DIR, PRIVATE_OUTPUT_DIR)

    ## print("\n>>> [Train] Now running PUBLIC test set ...")
    ## run_public_testset(model, PUBLIC_TEST_DIR, PUBLIC_OUTPUT_DIR)

    ## print("\n>>> All done! (Training + Inference on private/public sets)")
