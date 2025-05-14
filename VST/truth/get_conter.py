import cv2
import os

mask_dir = 'ground_truth_mask/'
contour_dir = 'Contour/'
os.makedirs(contour_dir, exist_ok=True)

for mask_name in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
    edge = cv2.Canny(mask, 50, 150)
    cv2.imwrite(os.path.join(contour_dir, mask_name), edge)