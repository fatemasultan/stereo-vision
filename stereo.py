import cv2
import numpy as np
import os

def load_images(folder):
    left_path=os.path.join(folder, "im0.png")
    right_path=os.path.join(folder, "im1.png")
    left=cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right=cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left is None:
        raise FileNotFoundError(f"Could not load: {left_path}")
    if right is None:
        raise FileNotFoundError(f"Could not load: {right_path}")
    return left, right

#compute disparity using StereoBM
def compute_disparity(left, right, num_disparities, block_size=15):
    stereo=cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity=stereo.compute(left, right)
    disparity = disparity.astype(np.float32) / 16.0
    return disparity


