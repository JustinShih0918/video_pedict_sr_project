print(f"DEBUG: Starting execution of models/optical_flow.py ({__file__})") # ADD THIS
import cv2
import numpy as np
print("DEBUG: Imports cv2 and numpy successful in models/optical_flow.py") # ADD THIS

def estimate_flow(frame0, frame2):
    # 灰階轉換
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Farneback Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

print("DEBUG: Successfully defined estimate_flow in models/optical_flow.py") # ADD THIS

def another_test_function_in_optical_flow(): # ADD THIS FUNCTION
    return "Hello from another_test_function_in_optical_flow"

print("DEBUG: Successfully defined another_test_function_in_optical_flow in models/optical_flow.py") # ADD THIS