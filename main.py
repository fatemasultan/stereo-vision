import os
from stereo import load_images, compute_disparity, save_disparity, calculate_depth
os.makedirs("outputs", exist_ok=True)

#dataset 1
folder="dataset/im0"
left, right =load_images(folder)
print("Left image shape:", left.shape)
print("Right image shape:", right.shape)
#calibration parameters for dataset 1
fx=3979.911
baseline=193.001
doffs=124.343
num_disparities=272

disparity=compute_disparity(
    left,
    right,
    num_disparities=num_disparities,
    block_size=15
)

save_disparity(
    disparity,
    "outputs/im0_disparity.png",
    "outputs/im0_disparity_color.png"
)

x=left.shape[1] // 2
y=left.shape[0] // 2
depth, d=calculate_depth(
    x,
    y,
    disparity,
    fx,
    baseline,
    doffs
)
print("Selected pixel:", (x, y))
print("Disparity:", d)

if depth is not None:
    print("Estimated depth:", depth, "meters")
else:
    print("Invalid disparity at this pixel.")

#Dataset 2
folder="dataset/im1"
left, right=load_images(folder)
print("Left image shape:", left.shape)
print("Right image shape:", right.shape)
#calibration parameters for dataset 2
fx=6338.47
baseline=171.548
doffs=479.489
num_disparities=400

disparity=compute_disparity(
    left,
    right,
    num_disparities=num_disparities,
    block_size=15
)

save_disparity(
    disparity,
    "outputs/im1_disparity.png",
    "outputs/im1_disparity_color.png"
)

x=left.shape[1] // 2
y=left.shape[0] // 2
depth, d=calculate_depth(
    x,
    y,
    disparity,
    fx,
    baseline,
    doffs
)
print("Selected pixel:", (x, y))
print("Disparity:", d)

if depth is not None:
    print("Estimated depth:", depth, "meters")
else:
    print("Invalid disparity at this pixel.")
