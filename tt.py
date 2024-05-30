import os
import cv2
import numpy as np
import json

def load_annotations(annotations_path):
    # Load the annotations file in COCO format
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def generate_segmentation_masks(annotations, image_shape, image_info):
    # Initialize an empty array to store the segmentation masks
    segmentation_masks = np.zeros(image_shape[:2], dtype=np.uint8)

    # Iterate through the annotations to generate masks for each category
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_info['id']:
            category_id = annotation['category_id']
            segmentation_mask = np.zeros(image_shape[:2], dtype=np.uint8)
            segmentations = annotation['segmentation']
            # Convert segmentation to polygon format if necessary
            # (depending on the format of segmentation in your COCO annotations)
            # Then fill the polygon to generate the mask
                    # Iterate through each segmentation and fill the polygon
            for segmentation in segmentations:
                pts = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(segmentation_mask, [pts], category_id)
            # Add the mask to the overall segmentation masks
            segmentation_masks = np.maximum(segmentation_masks, segmentation_mask)
    
    return segmentation_masks

def save_panoptic_segmentation(segmentation_masks, output_path):
    # Save the panoptic segmentation mask to a PNG file
    cv2.imwrite(output_path, segmentation_masks)


key_path = "train"
base_url = "~/dataset/seg_object_detection/auto_translate_v4-3"
# Directory containing images
image_dir = os.path.join(base_url, '{}'.format(key_path))

# Path to the annotations file
annotations_path = '_annotations.coco.json'

# Output directory for panoptic segmentation masks
output_dir = os.path.join(base_url, "panoptic_masks")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

output_dir = os.path.join(output_dir, key_path)
os.makedirs(output_dir, exist_ok=True)

# Load the annotations
annotations = load_annotations(os.path.join(image_dir, annotations_path))

# Process each image in the directory
for image_info in annotations['images']:
    filename = image_info['file_name']
    image_path = os.path.join(image_dir, filename)
    output_path = os.path.join(output_dir, filename.replace('.jpg', '_panoptic.png'))

    # Load the image
    image = cv2.imread(image_path)

    # Generate segmentation masks
    segmentation_masks = generate_segmentation_masks(annotations, image.shape, image_info)

    # Save the panoptic segmentation mask
    save_panoptic_segmentation(segmentation_masks, output_path)
