# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import json
import os
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
from PIL import Image
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"

coco_json = {
    "info": {
        "description": "Panoptic segmentation dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "Your Name",
        "date_created": "2024-06-07"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"supercategory": "person", "isthing": 1, "id": 1, "name": "person"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 2, "name": "bicycle"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 3, "name": "car"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 4, "name": "motorcycle"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 5, "name": "airplane"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 6, "name": "bus"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 7, "name": "train"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 8, "name": "truck"}, 
                   {"supercategory": "vehicle", "isthing": 1, "id": 9, "name": "boat"}, 
                   {"supercategory": "outdoor", "isthing": 1, "id": 10, "name": "traffic light"}, 
                   {"supercategory": "outdoor", "isthing": 1, "id": 11, "name": "fire hydrant"}, 
                   {"supercategory": "outdoor", "isthing": 1, "id": 13, "name": "stop sign"}, 
                   {"supercategory": "outdoor", "isthing": 1, "id": 14, "name": "parking meter"}, 
                   {"supercategory": "outdoor", "isthing": 1, "id": 15, "name": "bench"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 16, "name": "bird"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 17, "name": "cat"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 18, "name": "dog"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 19, "name": "horse"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 20, "name": "sheep"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 21, "name": "cow"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 22, "name": "elephant"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 23, "name": "bear"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 24, "name": "zebra"}, 
                   {"supercategory": "animal", "isthing": 1, "id": 25, "name": "giraffe"}, 
                   {"supercategory": "accessory", "isthing": 1, "id": 27, "name": "backpack"},
                   {"supercategory": "accessory", "isthing": 1, "id": 28, "name": "umbrella"}, 
                   {"supercategory": "accessory", "isthing": 1, "id": 31, "name": "handbag"}, 
                   {"supercategory": "accessory", "isthing": 1, "id": 32, "name": "tie"}, 
                   {"supercategory": "accessory", "isthing": 1, "id": 33, "name": "suitcase"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 34, "name": "frisbee"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 35, "name": "skis"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 36, "name": "snowboard"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 37, "name": "sports ball"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 38, "name": "kite"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 39, "name": "baseball bat"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 40, "name": "baseball glove"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 41, "name": "skateboard"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 42, "name": "surfboard"}, 
                   {"supercategory": "sports", "isthing": 1, "id": 43, "name": "tennis racket"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 44, "name": "bottle"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 46, "name": "wine glass"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 47, "name": "cup"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 48, "name": "fork"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 49, "name": "knife"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 50, "name": "spoon"}, 
                   {"supercategory": "kitchen", "isthing": 1, "id": 51, "name": "bowl"}, 
                   {"supercategory": "food", "isthing": 1, "id": 52, "name": "banana"}, 
                   {"supercategory": "food", "isthing": 1, "id": 53, "name": "apple"}, 
                   {"supercategory": "food", "isthing": 1, "id": 54, "name": "sandwich"}, 
                   {"supercategory": "food", "isthing": 1, "id": 55, "name": "orange"}, 
                   {"supercategory": "food", "isthing": 1, "id": 56, "name": "broccoli"}, 
                   {"supercategory": "food", "isthing": 1, "id": 57, "name": "carrot"}, 
                   {"supercategory": "food", "isthing": 1, "id": 58, "name": "hot dog"}, 
                   {"supercategory": "food", "isthing": 1, "id": 59, "name": "pizza"}, 
                   {"supercategory": "food", "isthing": 1, "id": 60, "name": "donut"}, 
                   {"supercategory": "food", "isthing": 1, "id": 61, "name": "cake"}, 
                   {"supercategory": "furniture", "isthing": 1, "id": 62, "name": "chair"}, 
                   {"supercategory": "furniture", "isthing": 1, "id": 63, "name": "couch"}, 
                   {"supercategory": "furniture", "isthing": 1, "id": 64, "name": "potted plant"}, 
                   {"supercategory": "furniture", "isthing": 1, "id": 65, "name": "bed"}, 
                   {"supercategory": "furniture", "isthing": 1, "id": 67, "name": "dining table"}, 
                   {"supercategory": "furniture", "isthing": 1, "id": 70, "name": "toilet"}, 
                   {"supercategory": "electronic", "isthing": 1, "id": 72, "name": "tv"}, 
                   {"supercategory": "electronic", "isthing": 1, "id": 73, "name": "laptop"}, 
                   {"supercategory": "electronic", "isthing": 1, "id": 74, "name": "mouse"},
                   {"supercategory": "electronic", "isthing": 1, "id": 75, "name": "remote"}, 
                   {"supercategory": "electronic", "isthing": 1, "id": 76, "name": "keyboard"},
                   {"supercategory": "electronic", "isthing": 1, "id": 77, "name": "cell phone"}, 
                   {"supercategory": "appliance", "isthing": 1, "id": 78, "name": "microwave"}, 
                   {"supercategory": "appliance", "isthing": 1, "id": 79, "name": "oven"}, 
                   {"supercategory": "appliance", "isthing": 1, "id": 80, "name": "toaster"}, 
                   {"supercategory": "appliance", "isthing": 1, "id": 81, "name": "sink"}, 
                   {"supercategory": "appliance", "isthing": 1, "id": 82, "name": "refrigerator"}, 
                   {"supercategory": "indoor", "isthing": 1, "id": 84, "name": "book"}, 
                   {"supercategory": "indoor", "isthing": 1, "id": 85, "name": "clock"},
                   {"supercategory": "indoor", "isthing": 1, "id": 86, "name": "vase"}, 
                   {"supercategory": "indoor", "isthing": 1, "id": 87, "name": "scissors"}, 
                   {"supercategory": "indoor", "isthing": 1, "id": 88, "name": "teddy bear"}, 
                   {"supercategory": "indoor", "isthing": 1, "id": 89, "name": "hair drier"}, 
                   {"supercategory": "indoor", "isthing": 1, "id": 90, "name": "toothbrush"}, 
                   {"supercategory": "textile", "isthing": 0, "id": 92, "name": "banner"}, 
                   {"supercategory": "textile", "isthing": 0, "id": 93, "name": "blanket"}, 
                   {"supercategory": "building", "isthing": 0, "id": 95, "name": "bridge"}, 
                   {"supercategory": "raw-material", "isthing": 0, "id": 100, "name": "cardboard"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 107, "name": "counter"}, 
                   {"supercategory": "textile", "isthing": 0, "id": 109, "name": "curtain"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 112, "name": "door-stuff"}, 
                   {"supercategory": "floor", "isthing": 0, "id": 118, "name": "floor-wood"}, 
                   {"supercategory": "plant", "isthing": 0, "id": 119, "name": "flower"}, 
                   {"supercategory": "food-stuff", "isthing": 0, "id": 122, "name": "fruit"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 125, "name": "gravel"}, 
                   {"supercategory": "building", "isthing": 0, "id": 128, "name": "house"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 130, "name": "light"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 133, "name": "mirror-stuff"}, 
                   {"supercategory": "structural", "isthing": 0, "id": 138, "name": "net"}, 
                   {"supercategory": "textile", "isthing": 0, "id": 141, "name": "pillow"},
                   {"supercategory": "ground", "isthing": 0, "id": 144, "name": "platform"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 145, "name": "playingfield"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 147, "name": "railroad"}, 
                   {"supercategory": "water", "isthing": 0, "id": 148, "name": "river"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 149, "name": "road"}, 
                   {"supercategory": "building", "isthing": 0, "id": 151, "name": "roof"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 154, "name": "sand"}, 
                   {"supercategory": "water", "isthing": 0, "id": 155, "name": "sea"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 156, "name": "shelf"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 159, "name": "snow"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 161, "name": "stairs"}, 
                   {"supercategory": "building", "isthing": 0, "id": 166, "name": "tent"}, 
                   {"supercategory": "textile", "isthing": 0, "id": 168, "name": "towel"}, 
                   {"supercategory": "wall", "isthing": 0, "id": 171, "name": "wall-brick"}, 
                   {"supercategory": "wall", "isthing": 0, "id": 175, "name": "wall-stone"}, 
                   {"supercategory": "wall", "isthing": 0, "id": 176, "name": "wall-tile"}, 
                   {"supercategory": "wall", "isthing": 0, "id": 177, "name": "wall-wood"}, 
                   {"supercategory": "water", "isthing": 0, "id": 178, "name": "water-other"},
                   {"supercategory": "window", "isthing": 0, "id": 180, "name": "window-blind"},
                   {"supercategory": "window", "isthing": 0, "id": 181, "name": "window-other"},
                   {"supercategory": "plant", "isthing": 0, "id": 184, "name": "tree-merged"}, 
                   {"supercategory": "structural", "isthing": 0, "id": 185, "name": "fence-merged"},
                   {"supercategory": "ceiling", "isthing": 0, "id": 186, "name": "ceiling-merged"}, 
                   {"supercategory": "sky", "isthing": 0, "id": 187, "name": "sky-other-merged"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 188, "name": "cabinet-merged"}, 
                   {"supercategory": "furniture-stuff", "isthing": 0, "id": 189, "name": "table-merged"},
                   {"supercategory": "floor", "isthing": 0, "id": 190, "name": "floor-other-merged"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 191, "name": "pavement-merged"}, 
                   {"supercategory": "solid", "isthing": 0, "id": 192, "name": "mountain-merged"}, 
                   {"supercategory": "plant", "isthing": 0, "id": 193, "name": "grass-merged"}, 
                   {"supercategory": "ground", "isthing": 0, "id": 194, "name": "dirt-merged"}, 
                   {"supercategory": "raw-material", "isthing": 0, "id": 195, "name": "paper-merged"},
                   {"supercategory": "food-stuff", "isthing": 0, "id": 196, "name": "food-other-merged"}, 
                   {"supercategory": "building", "isthing": 0, "id": 197, "name": "building-other-merged"}, 
                   {"supercategory": "solid", "isthing": 0, "id": 198, "name": "rock-merged"}, 
                   {"supercategory": "wall", "isthing": 0, "id": 199, "name": "wall-other-merged"},
                   {"supercategory": "textile", "isthing": 0, "id": 200, "name": "rug-merged"}
                   ]
}

json_file = "/fs/scratch/PAS0536/amin/MultiGen-20M/images/aesthetics_6_plus_0_annotation_file.json"

# Function to check if an image has been used before
def check_image_usage(image_id, used_images):
    # Implement your logic to check if the image has been used
    # Example: check in a database, log file, or other data structure
    # For simplicity, let's assume we're checking existence in a set
    return image_id in used_images


used_images = set()

if os.path.exists(json_file):
        
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
else:
    data = coco_json

# Assuming 'images' is a key in your JSON structure containing image data
images = data.get('images', [])

for image in images:
    image_id = image.get('id')
    file_name = image.get('file_name')
    
    # Check if image has been used before
    if check_image_usage(image_id):
        print(f"Image '{file_name}' with ID {image_id} has been used before.")
    else:
        print(f"Image '{file_name}' with ID {image_id} has not been used before.")
    
    # Track this image as used (example: add to a set)
    used_images.add(image_id)


def read_image(path, format="BGR"):
    img = Image.open(path)
    if format == "BGR":
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    annotation_id = 0
    if args.input:
        print(args.input[0])
        input_path = os.path.expanduser(args.input[0])
       
        if os.path.isdir(input_path):
            args.input = glob.glob(os.path.join(input_path, "*"))
        else:
            args.input = glob.glob(input_path)
        assert args.input, "The input path(s) was not found"
        idx = len(used_images)
        for path in tqdm.tqdm(args.input):
            if path[-4:] == "json":
                continue
            img = read_image(path, format="BGR")
            height, width = img.shape[:2]
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            panoptic_seg, segments_info = predictions['panoptic_seg']
            panoptic_seg = panoptic_seg.cpu().detach().numpy()
            
            image_info = {
                "file_name": path,
                "height": height,
                "width": width,
                "id": idx
            }
            idx += 1
            coco_json['images'].append(image_info)
            annotations_info = []
            unique_ids = np.unique(panoptic_seg)
            for segment_id in unique_ids:
                mask = (panoptic_seg == segment_id).astype(np.uint8)
                iscrowd = 0  # or 1, based on your context
                area = int(np.sum(mask))
                if area <= 2500:
                    continue

                # Bounding box
                y_indices, x_indices = np.where(mask)
                bbox = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices) - np.min(x_indices)), int(np.max(y_indices) - np.min(y_indices))]

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    segmentation.append(contour)
                    
                annotation_info = {
                    "id": annotation_id,
                    'image_id': image_info['id'],
                    "category_id": int(segment_id),
                    "area": area,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "iscrowd": iscrowd
                }
                annotations_info.append(annotation_info)
                annotation_id += 1
            coco_json['annotations'].extend(annotations_info)

        with open(json_file, 'w') as f:
            json.dump(coco_json, f, indent=4)