import os
import cv2
import numpy as np
import json
from panopticapi.utils import rgb2id

segmentation = [
                [
                    377.92,
                    385.16,
                    404.96,
                    385.16,
                    404.96,
                    346.28,
                    363.52,
                    346.28,
                    363.52,
                    373,
                    377.92,
                    373,
                    377.92,
                    385.16
                ]
            ]

pts = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
segmentation_mask = np.zeros((640, 640, 3), dtype=np.uint8)

cv2.fillPoly(segmentation_mask, [pts], [30, 0, 0])
panoptic = rgb2id(segmentation_mask)
print("amin")