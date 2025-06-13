# models/interpolation/interpolation.py

import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

# DEVICE 設定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path settings

TOPIC4_RELEASE_PATH = os.path.join("data", "private")
MODEL_PATH = os.path.join("savedModel", "vfi_model.pth")
PRIVATE_TEST_DIR = os.path.join(TOPIC4_RELEASE_PATH, "private_test_set")
PUBLIC_TEST_DIR  = os.path.join(TOPIC4_RELEASE_PATH, "public_test_set")
OUTPUT_DIR = os.path.join("output", "vfi_results")
PRIVATE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "private_test_output")
PUBLIC_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, "public_test_output")

# ── 2. OPTICAL FLOW 求解 ───────────────────────────────────────────────────────

def compute_flow(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:

 
    gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow


# ── 3. 模型結構定義 ─────────────────────────────────────────────────────────

class ImprovedInterpNet(nn.Module):


    def __init__(self, in_ch: int = 20):
        super().__init__()
        # ---------------- Encoder ----------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ---------------- Decoder ----------------
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ---------------- Output ----------------
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 20, H, W]
        x1 = self.enc1(x)                    # [B,64,H,W]
        x2 = self.enc2(self.pool1(x1))       # [B,128,H/2,W/2]
        x3 = self.bottleneck(self.pool2(x2)) # [B,256,H/4,W/4]

        x_up2 = self.up2(x3)  # [B,128,H/2,W/2]
        x_dec2 = self.dec2(torch.cat([x_up2, x2], dim=1))  # [B,128,H/2,W/2]

        x_up1 = self.up1(x_dec2)  # [B,64,H,W]
        x_dec1 = self.dec1(torch.cat([x_up1, x1], dim=1))  # [B,64,H,W]

        out = self.out_conv(x_dec1)  # [B,3,H,W]
        return self.final_act(out)   # 輸出範圍 [0,1]


# ── 4. Edge Enhancement 辅助函式 ────────────────────────────────────────────

def apply_edge_enhancement(
    tensor_img: torch.Tensor,
    amount: float = 1.5,
    radius: int = 1,
    saturation_scale: float = 1.2
) -> torch.Tensor:

    # (1) Tensor → CPU PIL → Numpy RGB
    img_pil = TF.to_pil_image(tensor_img.detach().cpu().clamp(0, 1))
    img_rgb = np.array(img_pil)  # uint8, shape=(H,W,3)

    # (2) RGB→BGR (OpenCV 標準)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # (3) BGR→LAB float32
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # (4) Unsharp Mask on L channel
    L = img_lab[:, :, 0]
    blurred = cv2.GaussianBlur(L, ksize=(2 * radius + 1, 2 * radius + 1), sigmaX=0)
    L_sharp = cv2.addWeighted(L, 1 + amount, blurred, -amount, 0)
    L_sharp = np.clip(L_sharp, 0, 255)
    img_lab[:, :, 0] = L_sharp

    # (5) 飽和度增強 (a,b channels)
    a = img_lab[:, :, 1] - 128.0
    b = img_lab[:, :, 2] - 128.0
    a = np.clip(a * saturation_scale, -128, 127)
    b = np.clip(b * saturation_scale, -128, 127)
    img_lab[:, :, 1] = a + 128.0
    img_lab[:, :, 2] = b + 128.0

    # (6) LAB→BGR→RGB
    img_bgr2 = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)

    # (7) 回到 Tensor range [0,1]
    enhanced = TF.to_tensor(img_rgb2).to(tensor_img.device).clamp(0, 1)
    return enhanced


# ── 5. 圖像讀取 & 前處理 ────────────────────────────────────────────────────

# Model 在 [256,448] 的解析度上訓練與推論 → 再 ×2 upsamping
image_size = (128, 224)  # (H, W) # k modified: (256, 448) -> (128, 224)

_transform_resize = transforms.Resize(image_size)
_transform_to_tensor = transforms.ToTensor()

def load_and_preprocess_image(path: str) -> torch.Tensor:
    """
    讀取 RGB 圖檔 → Resize((256,448)) → ToTensor() → [3,256,448], range [0,1]
    """
    img = Image.open(path).convert("RGB")
    img = _transform_resize(img)
    img = _transform_to_tensor(img)
    return img


# ── 6. PSNR & SSIM 計算 ─────────────────────────────────────────────────────

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    img1, img2: [3,H,W], range [0,1]
    回傳 SSIM (scalar)
    """
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim_func(img1_np, img2_np, data_range=1.0, channel_axis=2)

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    img1, img2: [3,H,W], range [0,1]
    回傳 PSNR (scalar)
    """
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return psnr_func(img1_np, img2_np, data_range=1.0)


# ── 7. 推論主程式 ────────────────────────────────────────────────────────────

def interpolate_and_evaluate_full6(
    im_paths: list[str],
    gt_path: str | None,
    model: nn.Module
) -> tuple[torch.Tensor, float, float, torch.Tensor]:
    """
    im_paths: 長度 6 的 list，依序是
        [im1.png, im2.png, im3.png, im5.png, im6.png, im7.png] → path 字串
    gt_path: im4.png 的路徑 (Public Set 才有 GT；Private Set 可傳 None 跳過計算)
    model: 已經 load 好（call load_interpolation_model 之後）的網路

    回傳：
        - pred_tensor: [3, 512, 896], range [0,1] (模型預測 + 2× upsample + edge enhancement)
        - psnr_val (float, 若 gt_path=None，則回傳 0.0)
        - ssim_val (float, 若 gt_path=None，則回傳 0.0)
        - residual: [3, 512, 896], abs(gt - pred) (若 gt_path=None，回傳全零 Tensor)
    """

    # Ⅰ. 載入 6 張 frame → Resize → Tensor [3,256,448]
    frames: list[torch.Tensor] = []
    for p in im_paths:
        img_t = load_and_preprocess_image(p)  # [3,256,448]
        frames.append(img_t)

    # Ⅱ. 若有 im3, im5 → 計算 optical flow
    img3 = cv2.imread(im_paths[2])
    img5 = cv2.imread(im_paths[3])
    if img3 is None or img5 is None:
        raise FileNotFoundError(f"Cannot read {im_paths[2]} or {im_paths[3]}")

    flow = compute_flow(img3, img5)  # [H_org, W_org, 2]
    # 正規化 & Resize to [256,448]
    flow_x = flow[..., 0] / (flow.shape[1] - 1)
    flow_y = flow[..., 1] / (flow.shape[0] - 1)
    flow_norm = np.stack([flow_x, flow_y], axis=-1)  # [H_org,W_org,2]
    flow_resized = cv2.resize(
        flow_norm,
        dsize=(image_size[1], image_size[0]),
        interpolation=cv2.INTER_LINEAR
    )  # [256,448,2]
    flow_tensor = torch.from_numpy(flow_resized).permute(2, 0, 1).float()  # [2,256,448]

    # Ⅲ. 組合輸入：cat 6×[3,256,448] + [2,256,448] → [20,256,448]
    input_tensor = torch.cat(frames + [flow_tensor], dim=0).unsqueeze(0).to(device)  # [1,20,256,448]

    # Ⅳ. 讀 GT 並 bicubic 2× → [3,512,896] (若 gt_path=None，直接跳過)
    if gt_path is not None:
        gt_tensor = load_and_preprocess_image(gt_path)  # [3,256,448]
        gt_tensor = F.interpolate(
            gt_tensor.unsqueeze(0),
            # scale_factor=2,
            scale_factor=1,
            mode="bicubic",
            align_corners=False
        )[0].to(device)  # [3,512,896]
    else:
        gt_tensor = None

    # Ⅴ. 模型推論 + bicubic 2× → [1,3,512,896]
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # [1,3,256,448]
        upsampled = F.interpolate(
            output,
            # scale_factor=2,
            scale_factor=1,
            mode="bicubic",
            align_corners=False
        )  # [1,3,512,896]
        pred = upsampled.squeeze(0).clamp(0, 1)  # [3,512,896], range [0,1]

    # Ⅵ. Edge Enhancement
    pred_enhanced = apply_edge_enhancement(pred)  # [3,512,896]

    # Ⅶ. 計算 residual & metrics (若 gt_path=None，就全歸零)
    if gt_tensor is not None:
        residual = torch.abs(gt_tensor - pred_enhanced).clamp(0, 1)
        psnr_val = compute_psnr(pred_enhanced, gt_tensor)
        ssim_val = compute_ssim(pred_enhanced, gt_tensor)
    else:
        residual = torch.zeros_like(pred_enhanced)
        psnr_val = 0.0
        ssim_val = 0.0

    return pred_enhanced.cpu(), psnr_val, ssim_val, residual.cpu()


# ── 8. 載入模型權重函式 ─────────────────────────────────────────────────────

def load_interpolation_model(MODEL_PATH: str) -> ImprovedInterpNet:
    """
    讀取一個訓練好的 checkpoint → 回傳 model（已 load state_dict、置於 device）
    """
    net = ImprovedInterpNet().to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(ckpt)
    net.eval()
    return net


# ── 9. Private/Public Test Set 處理函式 ─────────────────────────────────────

def save_tensor_as_png(t: torch.Tensor, out_path: str):
    """
    tensor t: [3,H,W], range [0,1] → 轉成 PIL image 存成 PNG
    """
    t = t.detach().cpu().clamp(0, 1)
    pil = TF.to_pil_image(t)
    pil.save(out_path, format="PNG")


def run_private_testset(
    model: nn.Module,
    private_test_dir: str,
    private_output_dir: str
):
    """
    Updated to handle an additional level of subdirectories.
    Each sequence folder (e.g. "00081") may have one or more subfolders (e.g. "0202")
    containing the images.
    """
    os.makedirs(private_output_dir, exist_ok=True)
    # List primary sequence folders (e.g. "00081")
    primary_seqs = sorted([d for d in os.listdir(private_test_dir)
                           if os.path.isdir(os.path.join(private_test_dir, d))])
    print(f"[Private Set] Found {len(primary_seqs)} primary sequences.")

    for seq in primary_seqs:
        primary_path = os.path.join(private_test_dir, seq)
        # List subdirectories inside each primary sequence folder
        subdirs = sorted([sd for sd in os.listdir(primary_path)
                          if os.path.isdir(os.path.join(primary_path, sd))])
        # If there are no subdirectories, assume the images are directly under primary_path.
        if not subdirs:
            subdirs = [""]

        for sub in subdirs:
            src_folder = os.path.join(primary_path, sub) if sub else primary_path
            dst_folder = os.path.join(private_output_dir, seq, sub) if sub else os.path.join(private_output_dir, seq)
            os.makedirs(dst_folder, exist_ok=True)

            # 1) Collect 6 frames
            inputs = []
            for i in [1, 2, 3, 5, 6, 7]:
                p = os.path.join(src_folder, f"im{i}.png")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"[Private] Missing {p}")
                inputs.append(p)

            # 2) Inference
            pred_tensor, _, _, _ = interpolate_and_evaluate_full6(inputs, None, model)

            # 3) Save as im4.png
            out_path = os.path.join(dst_folder, "im4.png")
            save_tensor_as_png(pred_tensor, out_path)
            print(f"[Private] Seq={seq}/{sub} → saved predicted im4 to {out_path}")


def run_public_testset(
    model: nn.Module,
    public_test_dir: str,
    public_output_dir: str
):
    """
    Process the public_test_set:
    - public_test_dir: path to “data/topic4_release/public_test_set”
    - public_output_dir: path to “data/topic4_release/public_test_output”
    For each primary folder (e.g. “00081”), each subfolder (e.g. “0202”) should have im1..im7 where im4 is the GT.
      1. Gather im1,2,3,5,6,7 → infer → save as im4.png
      2. Compute PSNR, SSIM, print & accumulate metrics, then output averages.
    """
    os.makedirs(public_output_dir, exist_ok=True)
    
    # List primary sequence folders (e.g. "00081")
    primary_seqs = sorted([d for d in os.listdir(public_test_dir)
                           if os.path.isdir(os.path.join(public_test_dir, d))])
    print(f"[Public Set] Found {len(primary_seqs)} primary sequences to process.")
    
    total_psnr = 0.0
    total_ssim = 0.0
    cnt = 0
    
    for seq in primary_seqs:
        primary_path = os.path.join(public_test_dir, seq)
        # List subdirectories inside each primary sequence folder.
        subdirs = sorted([sd for sd in os.listdir(primary_path)
                          if os.path.isdir(os.path.join(primary_path, sd))])
        # If no subdirectories, assume images are directly under primary_path.
        if not subdirs:
            subdirs = [""]
    
        for sub in subdirs:
            src_folder = os.path.join(primary_path, sub) if sub else primary_path
            dst_folder = os.path.join(public_output_dir, seq, sub) if sub else os.path.join(public_output_dir, seq)
            os.makedirs(dst_folder, exist_ok=True)
    
            # 1) Collect inputs + GT
            inputs = []
            for i in [1, 2, 3, 5, 6, 7]:
                p = os.path.join(src_folder, f"im{i}.png")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"[Public] Missing {p}")
                inputs.append(p)
            gt_path = os.path.join(src_folder, "im4.png")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"[Public] Missing {gt_path}")
    
            # 2) Inference + evaluation
            pred_tensor, psnr_val, ssim_val, residual = interpolate_and_evaluate_full6(
                inputs, gt_path, model
            )
    
            total_psnr += psnr_val
            total_ssim += ssim_val
            cnt += 1
    
            # 3) Save as im4.png
            out_path = os.path.join(dst_folder, "im4.png")
            save_tensor_as_png(pred_tensor, out_path)
            print(f"[Public] Seq={seq}/{sub} → PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f} saved to {out_path}")
    
    if cnt > 0:
        avg_psnr = total_psnr / cnt
        avg_ssim = total_ssim / cnt
        print(f"\n[Public] #Sequences={cnt}, Avg PSNR={avg_psnr:.4f}, Avg SSIM={avg_ssim:.4f}\n")
    else:
        print("[Public] No sequences found.")

# 10. main function

print(f"[VFI] Loading model from: {MODEL_PATH}")
model = load_interpolation_model(MODEL_PATH)

def predict_frame(
    input_dir: str,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all image paths in the input directory
    im_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    # first copy the input images to the output directory
    for im_path in im_paths:
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image not found: {im_path}")
        im_name = os.path.basename(im_path)
        if im_name.startswith("im4"):
            continue
        dst_path = os.path.join(output_dir, im_name)
        cv2.imwrite(dst_path, cv2.imread(im_path))
    
    if len(im_paths) < 6:
        raise ValueError(f"[VFI] Not enough images found in {input_dir}. Expected at least 6 images.")
    
    # Prepare inputs (first 6 images)
    inputs = im_paths[:6]
    
    # Inference
    pred_tensor, _, _, _ = interpolate_and_evaluate_full6(inputs, None, model)
    
    # Save output
    out_path = os.path.join(output_dir, "im4.png")
    save_tensor_as_png(pred_tensor, out_path)

# ── 10. If run this file stand‐alone，示範推論 private & public ─────────────────────

if __name__ == "__main__":
    # 調整路徑
    TOPIC4_RELEASE_PATH = os.path.join("data", "private")
    MODEL_PATH = os.path.join("savedModel", "vfi_model.pth")
    PRIVATE_TEST_DIR = os.path.join(TOPIC4_RELEASE_PATH, "private_test_set")
    PUBLIC_TEST_DIR  = os.path.join(TOPIC4_RELEASE_PATH, "public_test_set")
    OUTPUT_DIR = os.path.join("output", "vfi_results")
    PRIVATE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "private_test_output")
    PUBLIC_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, "public_test_output")

    print("Loading model from:", MODEL_PATH)
    model = load_interpolation_model(MODEL_PATH)

    print("\n>>> Running PRIVATE test set inference ...")
    run_private_testset(model, PRIVATE_TEST_DIR, PRIVATE_OUTPUT_DIR)

    print("\n>>> Running PUBLIC test set inference ...")
    run_public_testset(model, PUBLIC_TEST_DIR, PUBLIC_OUTPUT_DIR)

    print("\n>>> All inference on private & public sets finished!")


