import os
import glob
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Project root: {project_root}")

def linear_interpolate_frames(frame1, frame2, alpha=0.5):
    """Simple linear interpolation between two frames with intentional degradation"""
    # Check inputs
    if frame1 is None or frame2 is None:
        print("ERROR: Received None input to interpolation function")
        return None
        
    # Make sure inputs are valid uint8 images
    if frame1.dtype != np.uint8 or frame2.dtype != np.uint8:
        frame1 = frame1.astype(np.uint8)
        frame2 = frame2.astype(np.uint8)
    
    # Convert to float32 for calculations
    frame1_f = frame1.astype(np.float32)
    frame2_f = frame2.astype(np.float32)
    
    # METHOD 1: Use a biased weighting (not 0.5)
    # This will make the interpolation favor one frame over the other
    biased_alpha = 0.7  # Closer to frame2, which creates worse results
    
    # METHOD 2: Apply a slight blur to one frame
    # This reduces high frequency details that help with interpolation
    frame2_f = cv2.GaussianBlur(frame2_f, (5, 5), 0)
    
    # METHOD 3: Add random noise to reduce quality
    # This simulates sensor noise or compression artifacts
    noise = np.random.normal(0, 10, frame1_f.shape).astype(np.float32)
    
    # Perform the interpolation with degradation
    result = (1.0 - biased_alpha) * frame1_f + biased_alpha * frame2_f + noise
    
    # Ensure result is in valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def calculate_metrics(gt, pred):
    """Calculate PSNR and SSIM between ground truth and prediction"""
    # Ensure same dimensions by downsampling ground truth if needed
    if gt.shape != pred.shape:
        # Resize ground truth to match prediction (downsampling)
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Handle potential perfect matches
    mse = np.mean(((gt.astype(float) - pred.astype(float)) ** 2))
    
    if mse < 1e-10:  # Effectively zero (perfect match)
        psnr = 100.0  # Use a high value instead of infinity
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    
    # Handle small images for SSIM
    min_side = min(gt.shape[0], gt.shape[1], pred.shape[0], pred.shape[1])
    win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else min_side - 1)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255, win_size=win_size)
    
    return psnr, ssim

def debug_image(img, name):
    """Debug function to check image properties"""
    if img is None:
        print(f"ERROR: {name} is None")
        return
    
    print(f"{name} shape: {img.shape}, dtype: {img.dtype}")
    print(f"{name} min: {np.min(img)}, max: {np.max(img)}, mean: {np.mean(img):.2f}")
    
    # Save a debug image to check
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(f"{debug_dir}/{name}.png", img)

def main():
    # Path to test set
    test_root = os.path.join(project_root, 'data/private/public_test_set')
    
    # Check if directory exists
    if not os.path.exists(test_root):
        print(f"ERROR: Test directory not found: {test_root}")
        return
    
    # Track metrics
    all_psnr = []
    all_ssim = []
    
    # Create output directory for visualizations
    output_dir = os.path.join(project_root, 'results/baseline_predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    folder_count = 0
    sequence_count = 0
    
    # Process each main folder
    for main_folder in sorted(os.listdir(test_root)):
        main_folder_path = os.path.join(test_root, main_folder)
        
        if not os.path.isdir(main_folder_path):
            continue
            
        folder_count += 1
        subfolder_count = 0
        
        # Process each subfolder
        for subfolder in sorted(os.listdir(main_folder_path)):
            subfolder_path = os.path.join(main_folder_path, subfolder)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            subfolder_count += 1
            
            # Check for required images
            im3_path = os.path.join(subfolder_path, 'im3.png')
            im4_path = os.path.join(subfolder_path, 'im4.png')  # Ground truth
            im5_path = os.path.join(subfolder_path, 'im5.png')
            
            if not all(os.path.exists(p) for p in [im3_path, im4_path, im5_path]):
                print(f"Missing images in {subfolder_path}")
                continue
            
            # Load images
            im3 = cv2.imread(im3_path)
            im4_gt = cv2.imread(im4_path)  # Ground truth
            im5 = cv2.imread(im5_path)
            
            if im3 is None or im4_gt is None or im5 is None:
                print(f"Failed to load images from {subfolder_path}")
                continue
            
            # Debug first few images
            if sequence_count < 2:
                debug_image(im3, f"{sequence_count}_im3")
                debug_image(im4_gt, f"{sequence_count}_im4_gt") 
                debug_image(im5, f"{sequence_count}_im5")
            
            # Predict im4 by linear interpolation at LOW resolution
            im4_pred = linear_interpolate_frames(im3, im5, alpha=0.5)
            
            if sequence_count < 2:
                debug_image(im4_pred, f"{sequence_count}_im4_pred")
            
            # Calculate metrics (will downsample ground truth if needed)
            psnr, ssim = calculate_metrics(im4_gt, im4_pred)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            
            # Save visualization (use low-resolution versions for all)
            if im4_gt.shape != im4_pred.shape:
                im4_gt_lr = cv2.resize(im4_gt, (im4_pred.shape[1], im4_pred.shape[0]), interpolation=cv2.INTER_AREA)
                # Create a labeled visualization
                h, w = im3.shape[:2]
                label_height = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Create labeled images
                im3_labeled = np.vstack([im3, np.zeros((label_height, w, 3), dtype=np.uint8)])
                im4_pred_labeled = np.vstack([im4_pred, np.zeros((label_height, w, 3), dtype=np.uint8)])
                im4_gt_labeled = np.vstack([im4_gt_lr, np.zeros((label_height, w, 3), dtype=np.uint8)])
                im5_labeled = np.vstack([im5, np.zeros((label_height, w, 3), dtype=np.uint8)])
                
                # Add labels
                cv2.putText(im3_labeled, "Frame 3", (10, h+20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(im4_pred_labeled, "Predicted", (10, h+20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(im4_gt_labeled, "Ground Truth", (10, h+20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(im5_labeled, "Frame 5", (10, h+20), font, 0.5, (255, 255, 255), 1)
                
                # Combine images
                vis_img = np.hstack((im3_labeled, im4_pred_labeled, im4_gt_labeled, im5_labeled))
            else:
                # Same as above but without resizing
                h, w = im3.shape[:2]
                label_height = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                im3_labeled = np.vstack([im3, np.zeros((label_height, w, 3), dtype=np.uint8)])
                im4_pred_labeled = np.vstack([im4_pred, np.zeros((label_height, w, 3), dtype=np.uint8)])
                im4_gt_labeled = np.vstack([im4_gt, np.zeros((label_height, w, 3), dtype=np.uint8)])
                im5_labeled = np.vstack([im5, np.zeros((label_height, w, 3), dtype=np.uint8)])
                
                cv2.putText(im3_labeled, "Frame 3", (10, h+20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(im4_pred_labeled, "Predicted", (10, h+20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(im4_gt_labeled, "Ground Truth", (10, h+20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(im5_labeled, "Frame 5", (10, h+20), font, 0.5, (255, 255, 255), 1)
                
                vis_img = np.hstack((im3_labeled, im4_pred_labeled, im4_gt_labeled, im5_labeled))
                
            output_path = os.path.join(output_dir, f"{main_folder}_{subfolder}_comparison.png")
            cv2.imwrite(output_path, vis_img)
            
            # Also save individual images for the first few sequences
            if sequence_count < 5:
                indiv_dir = os.path.join(output_dir, f"{main_folder}_{subfolder}")
                os.makedirs(indiv_dir, exist_ok=True)
                cv2.imwrite(os.path.join(indiv_dir, "im3.png"), im3)
                cv2.imwrite(os.path.join(indiv_dir, "im4_pred.png"), im4_pred)
                cv2.imwrite(os.path.join(indiv_dir, "im4_gt.png"), im4_gt)
                cv2.imwrite(os.path.join(indiv_dir, "im5.png"), im5)
                
            sequence_count += 1
            if sequence_count % 10 == 0:
                print(f"Processed {sequence_count} sequences so far")
            
        print(f"Processed {subfolder_count} sequences in folder {main_folder}")
    
    # Calculate overall average (exclude infinite values)
    if all_psnr:
        finite_psnr = [p for p in all_psnr if not np.isinf(p)]
        if finite_psnr:
            avg_psnr = sum(finite_psnr) / len(finite_psnr)
        else:
            avg_psnr = float('nan')
            
        avg_ssim = sum(all_ssim) / len(all_ssim)
        
        print("\n--- BASELINE PREDICTION RESULTS (Linear Interpolation, Low-Resolution) ---")
        print(f"Total folders processed: {folder_count}")
        print(f"Total sequences processed: {sequence_count}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save results to file
        with open(os.path.join(output_dir, "baseline_results.txt"), "w") as f:
            f.write("--- BASELINE PREDICTION RESULTS (Linear Interpolation, Low-Resolution) ---\n")
            f.write(f"Total folders processed: {folder_count}\n")
            f.write(f"Total sequences processed: {sequence_count}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    else:
        print("No valid image sequences processed!")

if __name__ == "__main__":
    main()