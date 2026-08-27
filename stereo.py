import cv2
import numpy as np
import os

def load_images(folder):
    left_path=os.path.join(folder, "im0.png")
    right_path=os.path.join(folder, "img1.png")
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
    disparity=disparity.astype(np.float32)/16.0
    return disparity

#disparity map
def visual_disparity(disparity, output_path, color_output_path):
    #valid disparity values are greater than zero
    valid=disparity>0.
    disparity_normalized = np.zeros_like(disparity, dtype=np.uint8)

    if np.any(valid):
        min_disp=disparity[valid].min()
        max_disp=disparity[valid].max()
        disparity_normalized[valid] = cv2.normalize(
            disparity[valid],
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    cv2.imwrite(output_path, disparity_normalized)

    #color heatmap
    color_disparity = cv2.applyColorMap(
        disparity_normalized,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(color_output_path, color_disparity)
    print(f"Saved disparity map: {output_path}")
    print(f"Saved color disparity map: {color_output_path}")



