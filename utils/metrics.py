from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

def compute_psnr_ssim(gt_path, pred_path):
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    ssim = structural_similarity(gt, pred, multichannel=True, data_range=255)

    return psnr, ssim
