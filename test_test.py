import os
import cv2
import numpy as np
import json
from PIL import Image
from panopticapi.utils import rgb2id
import matplotlib.pyplot as plt


# segmentation = [
#                 [
#                     377.92,
#                     385.16,
#                     404.96,
#                     385.16,
#                     404.96,
#                     346.28,
#                     363.52,
#                     346.28,
#                     363.52,
#                     373,
#                     377.92,
#                     373,
#                     377.92,
#                     385.16
#                 ]
#             ]


# panoptic = np.asarray(Image.open('/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation/panoptic_masks/train/59_coco_png_jpg.rf.fb031620df625f57be6a25ad86d81f6c.jpg'), 
#                       dtype=np.uint32)
# plt.imshow(panoptic)


# panoptic_2 = rgb2id(panoptic)


# pts = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
# segmentation_mask = np.zeros((640, 640, 3), dtype=np.uint8)

# cv2.fillPoly(segmentation_mask, [pts], [30, 0, 0])


# output = np.zeros_like(panoptic, dtype=np.uint8) + 255

# output[panoptic == seg["id"]] = new_cat_id
# print("amin")


import cv2
import numpy as np

# Function to display the pixel value when the mouse hovers over the image
def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = img.copy()
        pixel_value = img[y, x]
        text = f'Pixel Value: {pixel_value}, i: {y}, j: {x}'
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Image', img_copy)

# Load the image
img = cv2.imread('/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation/panoptic_masks/train/59_coco_png_jpg.rf.fb031620df625f57be6a25ad86d81f6c.png') 
if img is None:
    print("Could not open or find the image.")
    exit(0)
    
print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j, 2] > 2:
            print(f"i: {i}, j: {j}")


aa = rgb2id(img)
# Create a window
cv2.namedWindow('Image')

# Set the mouse callback function to show the pixel value
cv2.setMouseCallback('Image', show_pixel_value)

# Display the image
while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(0) & 0xFF == 27:  # Exit the loop when 'Esc' key is pressed
        break

cv2.destroyAllWindows()
