import os
import cv2
import numpy as np
import json

HIGHARC_CATEGORIES = [
    {'color': [250, 141, 255], 'isthing': 1, 'id': 0, 'name': 'architectural-plans-kBh5'},
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'bath'},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bed_closet'}, 
    {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'bed_pass'},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'bedroom'}, 
    {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'chase'},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'closet'},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'dining'}, 
    {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'entry'},
    {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'fireplace'},
    {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': 'flex'},
    {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': 'foyer'},
    {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'front_porch'},
    {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'garage'},     
    {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'general'},
    {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'hall'}, 
    {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'hall_cased_opening'}, 
    {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'kitchen'}, 
    {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'laundry'},     
    {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'living'},
    {'color': [102, 102, 156], 'isthing': 1, 'id': 20, 'name': 'master_bed'},
    {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'master_closet'}, 
    {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'master_hall'}, 
    {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'master_vestibule'}, 
    {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'mech'}, 
    {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'mudroom'}, 
    {'color': [255, 179, 240], 'isthing': 1, 'id': 26, 'name': 'office'},
    {'color': [0, 125, 92], 'isthing': 1, 'id': 27, 'name': 'pantry'}, 
    {'color': [209, 0, 151], 'isthing': 1, 'id': 28, 'name': 'patio'},
    {'color': [188, 208, 182], 'isthing': 1, 'id': 29, 'name': 'portico'},
    {'color': [0, 220, 176], 'isthing': 1, 'id': 30, 'name': 'powder'},
    {'color': [255, 99, 164], 'isthing': 1, 'id': 31, 'name': 'reach_closet'},
    {'color': [92, 0, 73], 'isthing': 1, 'id': 32, 'name': 'reading_nook'},
    {'color': [133, 129, 255], 'isthing': 1, 'id': 33, 'name': 'rear_porch'}, 
    {'color': [78, 180, 255], 'isthing': 1, 'id': 34, 'name': 'solarium'}, 
    {'color': [0, 228, 0], 'isthing': 1, 'id': 35, 'name': 'stairs_editor'}, 
    {'color': [174, 255, 243], 'isthing': 1, 'id': 36, 'name': 'util_hall'},
    {'color': [45, 89, 255], 'isthing': 1, 'id': 37, 'name': 'walk'},
    {'color': [134, 134, 103], 'isthing': 1, 'id': 38, 'name': 'water_closet'},
    {'color': [145, 148, 174], 'isthing': 1, 'id': 39, 'name': 'workshop'},
]
def load_annotations(annotations_path):
    # Load the annotations file in COCO format
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def generate_segmentation_masks(annotations, image_shape, image_info):
    # Initialize an empty array to store the segmentation masks
    segmentation_masks = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Iterate through the annotations to generate masks for each category
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_info['id']:
            category_id = annotation['category_id']
            category_color = next(cat['color'] for cat in HIGHARC_CATEGORIES if cat['id'] == category_id)
            segmentation_mask = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            segmentations = annotation['segmentation']
            # Convert segmentation to polygon format if necessary
            # (depending on the format of segmentation in your COCO annotations)
            # Then fill the polygon to generate the mask
            for segmentation in segmentations:
                pts = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(segmentation_mask, [pts], category_color)
            # Add the mask to the overall segmentation masks
            segmentation_masks = np.maximum(segmentation_masks, segmentation_mask)

    return segmentation_masks

def save_panoptic_segmentation(segmentation_masks, output_path):
    # Save the panoptic segmentation mask to a PNG file
    cv2.imwrite(output_path, segmentation_masks)

for key_path in ["valid", "test", "train"]:
    base_url = "../../dataset/seg_object_detection/auto_translate_v4-3"
    # base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"
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
