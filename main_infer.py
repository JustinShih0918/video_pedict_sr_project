import os, sys

# 1) compute absolute project root
proj_root = os.path.dirname(os.path.abspath(__file__))
print(f"DEBUG: Project root: {proj_root}") # ADDED DEBUG

# 2) ensure it’s first on PYTHONPATH
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
print(f"DEBUG: sys.path[0]: {sys.path[0]}") # ADDED DEBUG
print(f"DEBUG: Full sys.path: {sys.path}") # ADDED DEBUG

# --- Ensure output directory exists ---
output_dir = 'output'
if not os.path.exists(output_dir):
    print(f"DEBUG: Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
else:
    print(f"DEBUG: Output directory {output_dir} already exists.")
# --- End ensure output directory ---

import cv2
print("DEBUG: Imported cv2 in main_infer.py") # ADDED DEBUG
import numpy as np # Often needed with cv2

# Try importing the module and then the name with detailed logging
try:
    print("DEBUG: Attempting to import models.optical_flow") # ADDED DEBUG
    import models.optical_flow
    print(f"DEBUG: Successfully imported models.optical_flow. Path: {models.optical_flow.__file__}") # ADDED DEBUG
    print(f"DEBUG: dir(models.optical_flow): {dir(models.optical_flow)}") # ADDED DEBUG

    print("DEBUG: Attempting to import estimate_flow from models.optical_flow") # ADDED DEBUG
    from models.optical_flow import estimate_flow
    print("DEBUG: Successfully imported estimate_flow") # ADDED DEBUG

    # Test the other function
    print("DEBUG: Attempting to import another_test_function_in_optical_flow") # ADDED DEBUG
    from models.optical_flow import another_test_function_in_optical_flow
    print(f"DEBUG: another_test_function_in_optical_flow() returns: {another_test_function_in_optical_flow()}") # ADDED DEBUG

except ImportError as e:
    print(f"DEBUG: ImportError occurred for optical_flow: {e}") # ADDED DEBUG
    if 'models.optical_flow' in sys.modules:
        m_opt_flow = sys.modules['models.optical_flow']
        print(f"DEBUG: sys.modules['models.optical_flow'] found. Path: {getattr(m_opt_flow, '__file__', 'N/A')}") # ADDED DEBUG
        print(f"DEBUG: dir(sys.modules['models.optical_flow']): {dir(m_opt_flow)}") # ADDED DEBUG
    else:
        print("DEBUG: sys.modules['models.optical_flow'] NOT found.") # ADDED DEBUG
    raise # re-raise the error to see the original traceback

try:
    print("DEBUG: Attempting to import models.fusion")
    import models.fusion
    print(f"DEBUG: Successfully imported models.fusion. Path: {models.fusion.__file__}")
    print(f"DEBUG: dir(models.fusion): {dir(models.fusion)}")

    print("DEBUG: Attempting to import warp_frame, fuse_frames from models.fusion")
    from models.fusion import warp_frame, fuse_frames
    print("DEBUG: Successfully imported warp_frame, fuse_frames from models.fusion")
except ImportError as e_fusion:
    print(f"DEBUG: ImportError for fusion module: {e_fusion}")
    if 'models.fusion' in sys.modules:
        m_fusion = sys.modules['models.fusion']
        print(f"DEBUG: sys.modules['models.fusion'] found. Path: {getattr(m_fusion, '__file__', 'N/A')}")
        print(f"DEBUG: dir(sys.modules['models.fusion']): {dir(m_fusion)}")
    else:
        print("DEBUG: sys.modules['models.fusion'] NOT found.")
    raise 

try:
    print("DEBUG: Attempting to import models.super_resolution")
    import models.upsampling.super_resolution
    print(f"DEBUG: Successfully imported models.super_resolution. Path: {models.upsampling.super_resolution.__file__}")
    print(f"DEBUG: dir(models.super_resolution): {dir(models.upsampling.super_resolution)}")

    print("DEBUG: Attempting to import upscale from models.super_resolution")
    from models.upsampling.super_resolution import upscale
    print("DEBUG: Successfully imported upscale from models.super_resolution")
except ImportError as e_sr:
    print(f"DEBUG: ImportError for super_resolution module: {e_sr}")
    if 'models.super_resolution' in sys.modules:
        m_sr = sys.modules['models.super_resolution']
        print(f"DEBUG: sys.modules['models.super_resolution'] found. Path: {getattr(m_sr, '__file__', 'N/A')}")
        print(f"DEBUG: dir(sys.modules['models.super_resolution']): {dir(m_sr)}")
    else:
        print("DEBUG: sys.modules['models.super_resolution'] NOT found.")
    raise 

# NEW IMPORT for flow regularization
try:
    print("DEBUG: Attempting to import regularize_flow_field from models.flow_regularization")
    from models.flow_regularization import regularize_flow_field
    print("DEBUG: Successfully imported regularize_flow_field")
except ImportError as e_reg:
    print(f"DEBUG: ImportError for flow_regularization module: {e_reg}")
    raise

from utils.metrics import compute_psnr_ssim

# 讀取 frame0 與 frame2
frame0_path = 'data/public/frame0.png'
frame2_path = 'data/public/frame2.png'
print(f"DEBUG: Loading frame0 from {frame0_path}")
frame0 = cv2.imread(frame0_path)
print(f"DEBUG: Loading frame2 from {frame2_path}")
frame2 = cv2.imread(frame2_path)

if frame0 is None:
    print(f"ERROR: Could not load frame0 from {frame0_path}")
    sys.exit(1)
if frame2 is None:
    print(f"ERROR: Could not load frame2 from {frame2_path}")
    sys.exit(1)

# Optical Flow
print("DEBUG: Estimating flow_0to2")
flow_0to2 = estimate_flow(frame0, frame2)
print("DEBUG: Estimating flow_2to0")
flow_2to0 = estimate_flow(frame2, frame0)

if flow_0to2 is None or flow_2to0 is None:
    print("ERROR: Optical flow estimation failed for one or both directions.")
    sys.exit(1)

# 中間 flow (簡單平均) - initial estimates
mid_flow0_initial = flow_0to2 * 0.5
mid_flow2_initial = flow_2to0 * 0.5
print("DEBUG: Calculated initial mid_flow0 and mid_flow2")

# --- Regularize the mid-flow fields ---
# Hyperparameters for regularization (these will likely need tuning)
lambda_reg = 0.05       # Smaller values = less smoothing, larger = more smoothing
iterations_reg = 30   # Number of iterations
step_size_reg = 0.1   # Learning rate for gradient descent

print("DEBUG: Regularizing mid_flow0_initial")
mid_flow0 = regularize_flow_field(mid_flow0_initial, 
                                  lambda_smoothness=lambda_reg, 
                                  num_iterations=iterations_reg, 
                                  step_size=step_size_reg)
if mid_flow0 is None: 
    print("Warning: mid_flow0 regularization returned None, using initial.")
    mid_flow0 = mid_flow0_initial

print("DEBUG: Regularizing mid_flow2_initial")
mid_flow2 = regularize_flow_field(mid_flow2_initial, 
                                  lambda_smoothness=lambda_reg, 
                                  num_iterations=iterations_reg, 
                                  step_size=step_size_reg)
if mid_flow2 is None:
    print("Warning: mid_flow2 regularization returned None, using initial.")
    mid_flow2 = mid_flow2_initial

# Warp
print("DEBUG: Warping frame0 using regularized mid_flow0")
warped0 = warp_frame(frame0, mid_flow0)
print("DEBUG: Warping frame2 using regularized mid_flow2")
warped2 = warp_frame(frame2, mid_flow2)

if warped0 is None or warped2 is None:
    print("ERROR: Warping failed for one or both frames.")
    sys.exit(1)

# Fuse
print("DEBUG: Fusing warped0 and warped2")
fused_frame = fuse_frames(warped0, warped2)
if fused_frame is None:
    print("ERROR: Fusing frames failed.")
    sys.exit(1)

# Super-resolution
print("DEBUG: Upscaling fused_frame")
sr_frame = upscale(fused_frame, scale=2)
if sr_frame is None:
    print("ERROR: Super-resolution failed.")
    sys.exit(1)

# 存檔 (for interpolated frame1)
output_path = os.path.join(output_dir, 'frame1_pred_regularized.png') # Use os.path.join
print(f"DEBUG: Saving super-resolved frame to {output_path}")
success_write = cv2.imwrite(output_path, sr_frame)
if success_write:
    print(f"Successfully saved output to {output_path}")
else:
    print(f"ERROR: Failed to save output to {output_path}")

# --- Predict Frame2 from Frame0 (Next Frame Prediction interpretation) ---
print("\nDEBUG: --- Starting Next Frame Prediction (Frame2 from Frame0) ---")

if frame0 is not None and flow_0to2 is not None:
    # ... (existing code for regularizing flow_0to2_for_prediction_regularized) ...
    
    print("DEBUG: Warping frame0 using flow_to_use_for_pred_f2 to predict frame2")
    predicted_frame2 = warp_frame(frame0, flow_to_use_for_pred_f2)

    if predicted_frame2 is None:
        print("ERROR: Prediction of Frame2 (warping frame0) failed.")
    else:
        output_predicted_frame2_path = os.path.join(output_dir, 'predicted_frame2_from_frame0.png') # Use os.path.join
        print(f"DEBUG: Saving predicted Frame2 to {output_predicted_frame2_path}")
        success_write_pred_f2 = cv2.imwrite(output_predicted_frame2_path, predicted_frame2)
        # ... (existing code for saving and evaluating predicted_frame2) ...
else:
    print("DEBUG: Skipping next frame prediction as frame0 or flow_0to2 is None.")

# (Optional) Compute PSNR/SSIM if ground truth frame1 is available
# gt_frame1_path = 'data/public/frame1.png'
# gt_frame1 = cv2.imread(gt_frame1_path)
# if gt_frame1 is not None and fused_frame.shape == gt_frame1.shape:
#     print("DEBUG: Computing PSNR/SSIM for the non-upscaled fused frame")
#     psnr, ssim = compute_psnr_ssim(gt_frame1, fused_frame)
#     print(f"Fused Frame vs GT - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
# else:
#     print("DEBUG: Ground truth frame1 not available or shape mismatch for PSNR/SSIM of fused frame.")

print("DEBUG: main_infer.py finished.")
