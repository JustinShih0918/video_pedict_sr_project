import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.upsampling.super_resolution import upscale

# Create utils directory and metrics.py if they don't exist
utils_dir = os.path.join(project_root, 'utils')
if not os.path.exists(utils_dir):
    os.makedirs(utils_dir, exist_ok=True)
    
    # Create __init__.py in utils directory
    with open(os.path.join(utils_dir, '__init__.py'), 'w') as f:
        pass
    
    # Create metrics.py with PSNR and SSIM functions
    with open(os.path.join(utils_dir, 'metrics.py'), 'w') as f:
        f.write('''
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr_ssim(img1_path, img2_path):
    """
    Calculate PSNR and SSIM between two images.
    
    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image
        
    Returns:
        tuple: (psnr, ssim) values
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Make sure images are the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr_value = 100
    else:
        psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM
    ssim_value = ssim(img1, img2, channel_axis=2, data_range=255)
    
    return psnr_value, ssim_value
''')

# Now import the metrics module
from utils.metrics import compute_psnr_ssim

# Set paths
test_lr_dir = os.path.join(project_root, 'data/private/train/Low_Resolution')  # Adjust this path as needed
test_hr_dir = os.path.join(project_root, 'data/private/train/High_Resolution')  # Adjust this path as needed
output_dir = os.path.join(project_root, 'output/sr_results')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all LR test images
lr_images = []
for root, _, files in os.walk(test_lr_dir):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            lr_path = os.path.join(root, file)
            # Calculate the corresponding HR path
            rel_path = os.path.relpath(lr_path, test_lr_dir)
            hr_path = os.path.join(test_hr_dir, rel_path)
            
            if os.path.exists(hr_path):
                lr_images.append((lr_path, hr_path))

print(f"Found {len(lr_images)} test image pairs")

# Process each image and calculate metrics
total_psnr = 0
total_ssim = 0

for idx, (lr_path, hr_path) in enumerate(lr_images):
    # Read low-resolution image
    lr_img = cv2.imread(lr_path)
    
    # Apply super-resolution
    sr_img = upscale(lr_img)
    
    # Save the super-resolved image
    output_path = os.path.join(output_dir, f"sr_{os.path.basename(lr_path)}")
    # cv2.imwrite(output_path, sr_img)
    
    # Calculate PSNR and SSIM
    psnr, ssim = compute_psnr_ssim(hr_path, output_path)
    
    print(f"\r[Test upsampling] Image {idx+1}/{len(lr_images)}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}", end="", flush=True)
    
    total_psnr += psnr
    total_ssim += ssim

# Calculate averages
avg_psnr = total_psnr / len(lr_images) if lr_images else 0
avg_ssim = total_ssim / len(lr_images) if lr_images else 0

print(f"\nAverage results over {len(lr_images)} images:")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")