#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from panopticapi.utils import rgb2id
from PIL import Image
import random



# HIGHARC_CATEGORIES = [
#     {'color': [250, 141, 255], 'isthing': 1, 'id': 0, 'name': 'architectural-plans-kBh5'},
#     {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'bath'},
#     {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bed_closet'}, 
#     {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'bed_pass'},
#     {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'bedroom'}, 
#     {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'chase'},
#     {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'closet'},
#     {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'dining'}, 
#     {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'entry'},
#     {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'fireplace'},
#     {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': 'flex'},
#     {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': 'foyer'},
#     {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'front_porch'},
#     {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'garage'},     
#     {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'general'},
#     {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'hall'}, 
#     {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'hall_cased_opening'}, 
#     {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'kitchen'}, 
#     {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'laundry'},     
#     {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'living'},
#     {'color': [102, 102, 156], 'isthing': 1, 'id': 20, 'name': 'master_bed'},
#     {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'master_closet'}, 
#     {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'master_hall'}, 
#     {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'master_vestibule'}, 
#     {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'mech'}, 
#     {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'mudroom'}, 
#     {'color': [255, 179, 240], 'isthing': 1, 'id': 26, 'name': 'office'},
#     {'color': [0, 125, 92], 'isthing': 1, 'id': 27, 'name': 'pantry'}, 
#     {'color': [209, 0, 151], 'isthing': 1, 'id': 28, 'name': 'patio'},
#     {'color': [188, 208, 182], 'isthing': 1, 'id': 29, 'name': 'portico'},
#     {'color': [0, 220, 176], 'isthing': 1, 'id': 30, 'name': 'powder'},
#     {'color': [255, 99, 164], 'isthing': 1, 'id': 31, 'name': 'reach_closet'},
#     {'color': [92, 0, 73], 'isthing': 1, 'id': 32, 'name': 'reading_nook'},
#     {'color': [133, 129, 255], 'isthing': 1, 'id': 33, 'name': 'rear_porch'}, 
#     {'color': [78, 180, 255], 'isthing': 1, 'id': 34, 'name': 'solarium'}, 
#     {'color': [0, 228, 0], 'isthing': 1, 'id': 35, 'name': 'stairs_editor'}, 
#     {'color': [174, 255, 243], 'isthing': 1, 'id': 36, 'name': 'util_hall'},
#     {'color': [45, 89, 255], 'isthing': 1, 'id': 37, 'name': 'walk'},
#     {'color': [134, 134, 103], 'isthing': 1, 'id': 38, 'name': 'water_closet'},
#     {'color': [145, 148, 174], 'isthing': 1, 'id': 39, 'name': 'workshop'},
# ]


# Function to generate a random color
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

# categories = [
#     {"id": 0, "name": "architectural-plans-fXWf", "supercategory": "none"},
#     {"id": 1, "name": "BALCONY", "supercategory": "architectural-plans-fXWf"},
#     {"id": 2, "name": "BASEMENT", "supercategory": "architectural-plans-fXWf"},
#     {"id": 3, "name": "BASEMENT FINISHED", "supercategory": "architectural-plans-fXWf"},
#     {"id": 4, "name": "BASEMENT UNFINISHED", "supercategory": "architectural-plans-fXWf"},
#     {"id": 5, "name": "BATHFULL", "supercategory": "architectural-plans-fXWf"},
#     {"id": 6, "name": "BEDROOM", "supercategory": "architectural-plans-fXWf"},
#     {"id": 7, "name": "BEDROOM_CLOSET", "supercategory": "architectural-plans-fXWf"},
#     {"id": 8, "name": "CAFE", "supercategory": "architectural-plans-fXWf"},
#     {"id": 9, "name": "CLOSET", "supercategory": "architectural-plans-fXWf"},
#     {"id": 10, "name": "COVERED LANAI", "supercategory": "architectural-plans-fXWf"},
#     {"id": 11, "name": "CRAWL SPACE", "supercategory": "architectural-plans-fXWf"},
#     {"id": 12, "name": "DECK", "supercategory": "architectural-plans-fXWf"},
#     {"id": 13, "name": "DEN", "supercategory": "architectural-plans-fXWf"},
#     {"id": 14, "name": "DINING", "supercategory": "architectural-plans-fXWf"},
#     {"id": 15, "name": "ENTRY", "supercategory": "architectural-plans-fXWf"},
#     {"id": 16, "name": "FLEX", "supercategory": "architectural-plans-fXWf"},
#     {"id": 17, "name": "FOYER", "supercategory": "architectural-plans-fXWf"},
#     {"id": 18, "name": "FRONT_PORCH", "supercategory": "architectural-plans-fXWf"},
#     {"id": 19, "name": "GAME ROOM", "supercategory": "architectural-plans-fXWf"},
#     {"id": 20, "name": "GARAGE", "supercategory": "architectural-plans-fXWf"},
#     {"id": 21, "name": "GENERAL", "supercategory": "architectural-plans-fXWf"},
#     {"id": 22, "name": "HALL", "supercategory": "architectural-plans-fXWf"},
#     {"id": 23, "name": "HVAC", "supercategory": "architectural-plans-fXWf"},
#     {"id": 24, "name": "KITCHEN", "supercategory": "architectural-plans-fXWf"},
#     {"id": 25, "name": "LAUNDRY", "supercategory": "architectural-plans-fXWf"},
#     {"id": 26, "name": "LIBRARY", "supercategory": "architectural-plans-fXWf"},
#     {"id": 27, "name": "LIVING", "supercategory": "architectural-plans-fXWf"},
#     {"id": 28, "name": "LOFT", "supercategory": "architectural-plans-fXWf"},
#     {"id": 29, "name": "MASTER_BED", "supercategory": "architectural-plans-fXWf"},
#     {"id": 30, "name": "MECH", "supercategory": "architectural-plans-fXWf"},
#     {"id": 31, "name": "MUDROOM", "supercategory": "architectural-plans-fXWf"},
#     {"id": 32, "name": "NOOK", "supercategory": "architectural-plans-fXWf"},
#     {"id": 33, "name": "OFFICE", "supercategory": "architectural-plans-fXWf"},
#     {"id": 34, "name": "OPEN TO BELOW", "supercategory": "architectural-plans-fXWf"},
#     {"id": 35, "name": "OWNER SUITE", "supercategory": "architectural-plans-fXWf"},
#     {"id": 36, "name": "PANTRY", "supercategory": "architectural-plans-fXWf"},
#     {"id": 37, "name": "PATIO", "supercategory": "architectural-plans-fXWf"},
#     {"id": 38, "name": "POWDER", "supercategory": "architectural-plans-fXWf"},
#     {"id": 39, "name": "PR", "supercategory": "architectural-plans-fXWf"},
#     {"id": 40, "name": "RECREATION ROOM", "supercategory": "architectural-plans-fXWf"},
#     {"id": 41, "name": "RISER", "supercategory": "architectural-plans-fXWf"},
#     {"id": 42, "name": "SHOWER", "supercategory": "architectural-plans-fXWf"},
#     {"id": 43, "name": "STAIRS", "supercategory": "architectural-plans-fXWf"},
#     {"id": 44, "name": "STORAGE", "supercategory": "architectural-plans-fXWf"},
#     {"id": 45, "name": "STUDY", "supercategory": "architectural-plans-fXWf"},
#     {"id": 46, "name": "TOILET", "supercategory": "architectural-plans-fXWf"},
#     {"id": 47, "name": "TUB", "supercategory": "architectural-plans-fXWf"},
#     {"id": 48, "name": "WALK_IN_CLOSET", "supercategory": "architectural-plans-fXWf"},
#     {"id": 49, "name": "WASH", "supercategory": "architectural-plans-fXWf"},
#     {"id": 50, "name": "WATER_CLOSET", "supercategory": "architectural-plans-fXWf"},
#     {"id": 51, "name": "fle", "supercategory": "architectural-plans-fXWf"},
#     {"id": 52, "name": "mechanical", "supercategory": "architectural-plans-fXWf"},
#     {"id": 53, "name": "ppc", "supercategory": "architectural-plans-fXWf"},
#     {"id": 54, "name": "unf", "supercategory": "architectural-plans-fXWf"}
# ]

categories = [
    {"id": 0, "name": "architectural-plans-fXWf", "supercategory": "none"},
    {"id": 1, "name": "BALCONY", "supercategory": "architectural-plans-fXWf"},
    {"id": 2, "name": "BASEMENT", "supercategory": "architectural-plans-fXWf"},
    {"id": 3, "name": "BASEMENT FINISHED", "supercategory": "architectural-plans-fXWf"},
    {"id": 4, "name": "BASEMENT UNFINISHED", "supercategory": "architectural-plans-fXWf"},
    {"id": 5, "name": "BATHFULL", "supercategory": "architectural-plans-fXWf"},
    {"id": 6, "name": "BEDROOM", "supercategory": "architectural-plans-fXWf"},
    {"id": 7, "name": "BEDROOM_CLOSET", "supercategory": "architectural-plans-fXWf"},
    {"id": 8, "name": "CAFE", "supercategory": "architectural-plans-fXWf"},
    {"id": 9, "name": "CLOSET", "supercategory": "architectural-plans-fXWf"},
    {"id": 10, "name": "COVERED LANAI", "supercategory": "architectural-plans-fXWf"},
    {"id": 11, "name": "CRAWL SPACE", "supercategory": "architectural-plans-fXWf"},
    {"id": 12, "name": "DECK", "supercategory": "architectural-plans-fXWf"},
    {"id": 13, "name": "DEN", "supercategory": "architectural-plans-fXWf"},
    {"id": 14, "name": "DINING", "supercategory": "architectural-plans-fXWf"},
    {"id": 15, "name": "ENTRY", "supercategory": "architectural-plans-fXWf"},
    {"id": 16, "name": "FLEX", "supercategory": "architectural-plans-fXWf"},
    {"id": 17, "name": "FOYER", "supercategory": "architectural-plans-fXWf"},
    {"id": 18, "name": "FRONT_PORCH", "supercategory": "architectural-plans-fXWf"},
    {"id": 19, "name": "GAME ROOM", "supercategory": "architectural-plans-fXWf"},
    {"id": 20, "name": "GARAGE", "supercategory": "architectural-plans-fXWf"},
    {"id": 21, "name": "GENERAL", "supercategory": "architectural-plans-fXWf"},
    {"id": 22, "name": "HALL", "supercategory": "architectural-plans-fXWf"},
    {"id": 23, "name": "HVAC", "supercategory": "architectural-plans-fXWf"},
    {"id": 24, "name": "KITCHEN", "supercategory": "architectural-plans-fXWf"},
    {"id": 25, "name": "LAUNDRY", "supercategory": "architectural-plans-fXWf"},
    {"id": 26, "name": "LIBRARY", "supercategory": "architectural-plans-fXWf"},
    {"id": 27, "name": "LIVING", "supercategory": "architectural-plans-fXWf"},
    {"id": 28, "name": "LOFT", "supercategory": "architectural-plans-fXWf"},
    {"id": 29, "name": "MASTER_BED", "supercategory": "architectural-plans-fXWf"},
    {"id": 30, "name": "MECH", "supercategory": "architectural-plans-fXWf"},
    {"id": 31, "name": "MUDROOM", "supercategory": "architectural-plans-fXWf"},
    {"id": 32, "name": "NOOK", "supercategory": "architectural-plans-fXWf"},
    {"id": 33, "name": "OFFICE", "supercategory": "architectural-plans-fXWf"},
    {"id": 34, "name": "OPEN TO BELOW", "supercategory": "architectural-plans-fXWf"},
    {"id": 35, "name": "OWNER SUITE", "supercategory": "architectural-plans-fXWf"},
    {"id": 36, "name": "PANTRY", "supercategory": "architectural-plans-fXWf"},
    {"id": 37, "name": "PATIO", "supercategory": "architectural-plans-fXWf"},
    {"id": 38, "name": "POWDER", "supercategory": "architectural-plans-fXWf"},
    {"id": 39, "name": "PR", "supercategory": "architectural-plans-fXWf"},
    {"id": 40, "name": "RISER", "supercategory": "architectural-plans-fXWf"},
    {"id": 41, "name": "SHOWER", "supercategory": "architectural-plans-fXWf"},
    {"id": 42, "name": "STAIRS", "supercategory": "architectural-plans-fXWf"},
    {"id": 43, "name": "STORAGE", "supercategory": "architectural-plans-fXWf"},
    {"id": 44, "name": "STUDY", "supercategory": "architectural-plans-fXWf"},
    {"id": 45, "name": "WALK_IN_CLOSET", "supercategory": "architectural-plans-fXWf"},
    {"id": 46, "name": "WATER_CLOSET", "supercategory": "architectural-plans-fXWf"},
    {"id": 47, "name": "mechanical", "supercategory": "architectural-plans-fXWf"},
    {"id": 48, "name": "ppc", "supercategory": "architectural-plans-fXWf"},
]



HIGHARC_CATEGORIES = [
    {'color': generate_random_color(), 'isthing': 1, 'id': cat['id'], 'name': cat['name']} for cat in categories
]

print(HIGHARC_CATEGORIES)



def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.

    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0
    id_map[0] = 255

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_type = "pulte_lable_81"
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
        base_url = "../../dataset/experiment_one"
    

    for key_path in key_paths:   
        separate_coco_semantic_from_panoptic(
            os.path.join(base_url, "{}/_panoptic_annotations.coco.json".format(key_path)),
            # os.path.join(base_url, "{}/_panoptic_annotation_pulte_maskdino_augmented_file.json".format(key_path)),
            os.path.join(base_url, "panoptic_masks/{}".format(key_path)),
            # os.path.join(base_url, "panoptic_masks_maskdino_augmented/{}".format(key_path)),
            os.path.join(base_url, "panoptic_stuff/{}".format(key_path)),
            # os.path.join(base_url, "panoptic_stuff_maskdino_augmented/{}".format(key_path)),
            HIGHARC_CATEGORIES,
        )
