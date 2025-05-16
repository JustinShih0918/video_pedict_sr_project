from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

def compute_psnr_ssim(gt_path, pred_path):
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    print(f"DEBUG: gt shape: {None if gt is None else gt.shape}, pred shape: {None if pred is None else pred.shape}")

    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    # Use channel_axis=2 for color images, and set win_size if needed
    min_side = min(gt.shape[0], gt.shape[1], pred.shape[0], pred.shape[1])
    win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else min_side - 1)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255, win_size=win_size)

    return psnr, ssim
