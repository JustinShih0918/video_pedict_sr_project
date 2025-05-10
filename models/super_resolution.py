import cv2
import numpy as np # If needed
print(f"DEBUG: Starting execution of models/super_resolution.py ({__file__})")
print("DEBUG: Imports successful in models/super_resolution.py")

def upscale(frame, scale=2):
    print("DEBUG: upscale is defined")
    if frame is None:
        print("DEBUG: upscale received a None frame!")
        return None
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
print("DEBUG: Successfully defined upscale in models/super_resolution.py")
