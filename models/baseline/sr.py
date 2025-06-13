import os
import glob
import cv2
import numpy as np
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from PIL import Image

# Get project root directory (2 levels up from script location)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Project root: {project_root}")

def bilinear_upscale(img, scale=2):
    """Upscale an image using bilinear interpolation"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)

def calculate_metrics(gt, pred):
    """Calculate PSNR and SSIM between ground truth and prediction"""
    # Calculate MSE manually to check for perfect matches
    mse = np.mean(((gt.astype(float) - pred.astype(float)) ** 2))
    
    if mse < 1e-10:  # Effectively zero (perfect match)
        print(f"WARNING: Perfect match detected (MSE={mse:.10f})")
        # For debugging, save these images to inspect them
        if not os.path.exists("debug_perfect_matches"):
            os.makedirs("debug_perfect_matches")
        # Save a few samples for inspection
        debug_count = len(os.listdir("debug_perfect_matches")) // 2
        if debug_count < 5:  # Limit to 5 samples
            cv2.imwrite(f"debug_perfect_matches/gt_{debug_count}.png", gt)
            cv2.imwrite(f"debug_perfect_matches/pred_{debug_count}.png", pred)
        psnr = 100.0  # Use a high value instead of infinity
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    
    # Handle small images for SSIM
    min_side = min(gt.shape[0], gt.shape[1], pred.shape[0], pred.shape[1])
    win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else min_side - 1)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255, win_size=win_size)
    
    return psnr, ssim

def find_image_pairs(lr_dir, hr_dir):
    """
    Find matching image pairs using the same approach as SRDataset
    """
    lr_files = []
    hr_files = []
    
    # Find all subfolders (00001, 00002, etc.)
    subfolders = []
    for folder in sorted(os.listdir(lr_dir)):
        if os.path.isdir(os.path.join(lr_dir, folder)) and os.path.isdir(os.path.join(hr_dir, folder)):
            subfolders.append(folder)
    
    # Create lists of matching image pairs
    for folder in subfolders:
        lr_subfolder = os.path.join(lr_dir, folder)
        hr_subfolder = os.path.join(hr_dir, folder)
        
        # Recursively find all .png files
        lr_images = sorted(glob.glob(os.path.join(lr_subfolder, "**", "*.png"), recursive=True))
        
        # Ensure we have matching pairs (same relative paths)
        for lr_path in lr_images:
            rel_path = os.path.relpath(lr_path, lr_subfolder)
            hr_path = os.path.join(hr_subfolder, rel_path)
            if os.path.exists(hr_path):
                lr_files.append(lr_path)
                hr_files.append(hr_path)
    
    print(f"Found {len(lr_files)} matching image pairs across {len(subfolders)} folders")
    return lr_files, hr_files, subfolders

def main():
    # Use absolute paths based on project root
    lr_dir = os.path.join(project_root, 'data/private/train/Low_Resolution')
    hr_dir = os.path.join(project_root, 'data/private/train/High_Resolution')
    
    print(f"Looking for images in:\n  LR: {lr_dir}\n  HR: {hr_dir}")
    
    # Check if directories exist
    if not os.path.exists(lr_dir):
        print(f"ERROR: Low-resolution directory not found: {lr_dir}")
        return
    if not os.path.exists(hr_dir):
        print(f"ERROR: High-resolution directory not found: {hr_dir}")
        return
        
    # Find all matching image pairs
    lr_files, hr_files, subfolders = find_image_pairs(lr_dir, hr_dir)
    
    # Track metrics
    all_psnr = []
    all_ssim = []
    folder_metrics = {folder: {'psnr': [], 'ssim': []} for folder in subfolders}
    
    # Process all image pairs
    total_images = 0
    for lr_path, hr_path in zip(lr_files, hr_files):
        # Extract folder name from path
        rel_to_lr_dir = os.path.relpath(lr_path, lr_dir)
        folder = rel_to_lr_dir.split(os.sep)[0]  # First component of the relative path
        
        # Load images
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        
        # Skip if any image failed to load
        if lr_img is None or hr_img is None:
            print(f"Warning: Could not load {lr_path} or {hr_path}")
            continue
        
        # Determine scale factor from image dimensions
        scale = hr_img.shape[0] // lr_img.shape[0]  # Assuming square aspect ratio
        upscaled_img = bilinear_upscale(lr_img, scale)
        
        # Make sure dimensions match (crop if necessary)
        if upscaled_img.shape != hr_img.shape:
            min_h = min(upscaled_img.shape[0], hr_img.shape[0])
            min_w = min(upscaled_img.shape[1], hr_img.shape[1])
            upscaled_img = upscaled_img[:min_h, :min_w]
            hr_img = hr_img[:min_h, :min_w]
        
        # Calculate metrics
        psnr, ssim = calculate_metrics(hr_img, upscaled_img)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        folder_metrics[folder]['psnr'].append(psnr)
        folder_metrics[folder]['ssim'].append(ssim)
        
        total_images += 1
        if total_images % 500 == 0:
            print(f"Processed {total_images} images")
    
    # Print folder-specific metrics
    for folder in subfolders:
        if folder_metrics[folder]['psnr']:
            avg_folder_psnr = sum(folder_metrics[folder]['psnr']) / len(folder_metrics[folder]['psnr'])
            avg_folder_ssim = sum(folder_metrics[folder]['ssim']) / len(folder_metrics[folder]['ssim'])
            print(f"Folder {folder}: Avg PSNR = {avg_folder_psnr:.2f} dB, Avg SSIM = {avg_folder_ssim:.4f}")
    
    # Calculate overall average (exclude inf values)
    if all_psnr:
        # Filter out inf values
        finite_psnr = [p for p in all_psnr if not np.isinf(p)]
        if finite_psnr:
            avg_psnr = sum(finite_psnr) / len(finite_psnr)
            print(f"Finite PSNR values: {len(finite_psnr)}/{len(all_psnr)}")
        else:
            avg_psnr = float('nan')
        avg_ssim = sum(all_ssim) / len(all_ssim)
        
        print("\n--- BASELINE RESULTS (Bilinear Interpolation) ---")
        print(f"Total images processed: {total_images}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Optional: save results to a file
        with open("baseline_results.txt", "w") as f:
            f.write("--- BASELINE RESULTS (Bilinear Interpolation) ---\n")
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    else:
        print("No matching image pairs found!")

if __name__ == "__main__":
    main()