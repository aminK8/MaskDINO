import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import random

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    return [random.randint(0, 255) for _ in range(3)]

def plot_coco_segmentation(images_dir, coco_annotation_path, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load COCO annotations
    coco = COCO(coco_annotation_path)
    
    # Generate a random color for each category
    category_colors = {cat['id']: random_color(cat['id']) for cat in coco.loadCats(coco.getCatIds())}

    # Load all image metadata from COCO annotations
    img_metadata = coco.loadImgs(coco.getImgIds())

    # Process each image in the directory
    for image_file in os.listdir(images_dir):
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            image_path = os.path.join(images_dir, image_file)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find the image ID based on filename
            image_id = None
            for img in img_metadata:
                if img['file_name'] == image_file:
                    image_id = img['id']
                    break

            if image_id is None:
                print(f"No annotations found for image: {image_file}")
                continue

            ann_ids = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(ann_ids)
            
            # Create an empty mask for visualization
            segmentation_mask = np.zeros_like(image_rgb)

            # Plot each segmentation
            for ann in anns:
                category_id = ann['category_id']
                mask = coco.annToMask(ann)
                color = category_colors[category_id]
                
                segmentation_mask[mask == 1] = color

                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_rgb, contours, -1, color, 3)
            
            # Blend the original image with the segmentation mask
            blended_image = cv2.addWeighted(image_rgb, 0.7, segmentation_mask, 0.3, 0)

            # Save the result
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
            print(f"Segmented image saved to {output_path}")


# Usage
images_dir = "/Users/amin/Desktop/higharc/Datasets/unlabeled/data_pulte/pulte/floorplans"  # Directory containing input images
coco_annotation_path = "/Users/amin/Desktop/higharc/Datasets/unlabeled/data_pulte/pulte/floorplans/_annotations.coco.json"  # Update with actual path
output_dir = "/Users/amin/Desktop/higharc/Datasets/unlabeled/data_pulte/pulte/plot_output"  # Directory to save segmented images

plot_coco_segmentation(images_dir, coco_annotation_path, output_dir)
