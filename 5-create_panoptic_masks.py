import os
import cv2
import numpy as np
import json
import shutil
from panopticapi.utils import id2rgb


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
            segmentation_mask = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
            segmentations = annotation['segmentation']

            for segmentation in segmentations:
                pts = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                color = id2rgb(annotation['id'])
                cv2.fillPoly(segmentation_mask, [pts], color)
            # Add the mask to the overall segmentation masks
            segmentation_masks = np.maximum(segmentation_masks, segmentation_mask)

    # if (segmentation_masks[:, :, 2] > 0).any():
    #     print("amin")
    return segmentation_masks

def save_panoptic_segmentation(segmentation_masks, output_path):
    cv2.imwrite(output_path, segmentation_masks)
 
    
dataset_type = "expriment_three"
key_paths = []
base_url = ""

if dataset_type == "construction":
    key_paths = ["valid", "test", "train"]
    # base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"
    base_url = "../../dataset/seg_object_detection/auto_translate_v4-3"
    
elif dataset_type == 'pulte_unlabel':
    key_paths = ['floorplans']
    base_url = "../../dataset/data_pulte/pulte"

elif dataset_type == 'pulte_lable_81':
    key_paths = ["valid", "train"]
    base_url = "../../dataset/experiment_two"
    
elif dataset_type == 'pseudo':
    key_paths = ["train"]
    base_url = "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo"
    
elif dataset_type == 'expriment_three':
    key_paths = ["test", "train"]
    base_url = "../../dataset/expriment_three_1/"
    

for key_path in key_paths:   
    print(f"key is {key_path}")
    print(f"base_url is {base_url}")
    image_dir = os.path.join(base_url, '{}'.format(key_path))
    print(f"image_dir is {image_dir}")
    # Path to the annotations file
    annotations_path = '_annotations.coco.json'
    # annotations_path = '_annotation_pulte_maskdino_augmented_file.json'

    # Output directory for panoptic segmentation masks
    output_dir = os.path.join(base_url, "panoptic_masks")
    # output_dir = os.path.join(base_url, "panoptic_masks_maskdino_augmented")

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
        output_path = os.path.join(output_dir, filename[:-4] + ".png")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
            search_directories = [
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/allisonramsey",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/brohn",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/centurycommunities",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/harrisdoyle",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/homeplans",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/lennar",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/lgi",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/nvhomes",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/pulte",
                "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999/yourarborhome"
            ]
            target_directory, filename = os.path.split(image_path)
            
            for directory in search_directories:
                image_path_search = os.path.join(directory, filename)
                image = cv2.imread(image_path_search)
                if image is not None:
                    # Copy the image to the target directory
                    shutil.copy(image_path_search, image_path)
                    print(f"Image copied from {image_path_search} to {image_path}")
                    break
                else:
                    print(f"Image not found in {image_path_search}")
                    print(image_path)
                    continue

        # Generate segmentation masks
        segmentation_masks = generate_segmentation_masks(annotations, image.shape, image_info)

        segmentation_masks = cv2.cvtColor(segmentation_masks, cv2.COLOR_BGR2RGB)

        # Save the panoptic segmentation mask
        save_panoptic_segmentation(segmentation_masks, output_path)
