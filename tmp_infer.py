import os, sys
import cv2
import numpy as np

proj_root = os.path.dirname(os.path.abspath(__file__))

if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from utils.debug import set_flag, debug

try:
    from models.upsampling.super_resolution import upscale
    from models.vfi.vfi import predict_frame
    debug(f"[Main]: Imports successful in super_resolution module from {proj_root}")
except ImportError as e:
    debug(f"[Main]: ImportError: {e}")

input_dir = os.path.join(proj_root, "data/private/private_test_set/00081/0202") # modify this path as needed
output_dir = os.path.join(proj_root, "output/infer_results") # modify this path as needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    debug(f"[Main]: Created output directory: {output_dir}")

def main():
    # run vfi on input sequence to predict frame: im4
    debug(f"[Main]: Running VFI on input sequence: {input_dir}")
    predict_frame(input_dir=input_dir, output_dir=output_dir)
    
    debug(f"[Main]: VFI completed. Output saved to: {output_dir}")

    # run super resolution on the output sequence
    debug(f"[Main]: Running Super Resolution on output sequence: {output_dir}")
    for filename in os.listdir(output_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # get one image in the sequence and convert to np image
            debug(f"[Main]: Processing image {filename} for Super Resolution")
            input_image_path = os.path.join(output_dir, filename)
            input_image = cv2.imread(input_image_path)
            if input_image is None:
                debug(f"\r[Main]: Failed to read image {input_image_path}")
                continue
            output_image = upscale(input_image)
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, output_image)

    debug(f"\r[Main]: Super Resolution completed. Output saved to: {output_dir}")

if __name__ == "__main__":
    set_flag("-d" in sys.argv)
    print("[Main]: Starting main execution.")
    main()
    print(f"[Main]: Main execution completed, see output in {output_dir}")