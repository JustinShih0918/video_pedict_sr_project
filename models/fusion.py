import numpy as np
import cv2

print(f"DEBUG: Starting execution of models/fusion.py ({__file__})")
print("DEBUG: Imports successful in models/fusion.py")

def warp_frame(frame, flow):
    print("DEBUG: warp_frame is defined")
    h, w = frame.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(frame, flow, None, cv2.INTER_LINEAR)
    return res
print("DEBUG: Successfully defined warp_frame in models/fusion.py")

def fuse_frames(frame1, frame2):
    print("DEBUG: fuse_frames is defined")
    res = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
    return res
print("DEBUG: Successfully defined fuse_frames in models/fusion.py")