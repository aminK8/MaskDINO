#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


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

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 60
    for i, k in enumerate(categories):
        id_map[k] = i
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name.replace(".jpg", ".png"))
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



# mscoco_category2name = {
#     0: "none",
#     1: "bath",
#     2: "bed_closet",
#     3: "bed_pass",
#     4: "bedroom",
#     5: "chase",
#     6: "closet",
#     7: "dining",
#     8: "entry",
#     9: "fireplace",
#     10: "flex",
#     11: "foyer",
#     12: "front_porch",
#     13: "garage",
#     14: "general",
#     15: "hall",
#     16: "hall_cased_opening",
#     17: "kitchen",
#     18: "laundry",
#     19: "living",
#     20: "master_bed",
#     21: "master_closet",
#     22: "master_hall",
#     23: "master_vestibule",
#     24: "mech",
#     25: "mudroom",
#     26: "office",
#     27: "pantry",
#     28: "patio",
#     29: "portico",
#     30: "powder",
#     31: "reach_closet",
#     32: "reading_nook",
#     33: "rear_porch",
#     34: "solarium",
#     35: "stairs_editor",
#     36: "util_hall",
#     37: "walk",
#     38: "water_closet",
#     39: "workshop"
# }

# mscoco_category2name = {
#     0: "architectural-plans-fXWf",
#     1: "BALCONY",
#     2: "BASEMENT",
#     3: "BASEMENT FINISHED",
#     4: "BASEMENT UNFINISHED",
#     5: "BATHFULL",
#     6: "BEDROOM",
#     7: "BEDROOM_CLOSET",
#     8: "CAFE",
#     9: "CLOSET",
#     10: "COVERED LANAI",
#     11: "CRAWL SPACE",
#     12: "DECK",
#     13: "DEN",
#     14: "DINING",
#     15: "ENTRY",
#     16: "FLEX",
#     17: "FOYER",
#     18: "FRONT_PORCH",
#     19: "GAME ROOM",
#     20: "GARAGE",
#     21: "GENERAL",
#     22: "HALL",
#     23: "HVAC",
#     24: "KITCHEN",
#     25: "LAUNDRY",
#     26: "LIBRARY",
#     27: "LIVING",
#     28: "LOFT",
#     29: "MASTER_BED",
#     30: "MECH",
#     31: "MUDROOM",
#     32: "NOOK",
#     33: "OFFICE",
#     34: "OPEN TO BELOW",
#     35: "OWNER SUITE",
#     36: "PANTRY",
#     37: "PATIO",
#     38: "POWDER",
#     39: "PR",
#     40: "RECREATION ROOM",
#     41: "RISER",
#     42: "SHOWER",
#     43: "STAIRS",
#     44: "STORAGE",
#     45: "STUDY",
#     46: "TOILET",
#     47: "TUB",
#     48: "WALK_IN_CLOSET",
#     49: "WASH",
#     50: "WATER_CLOSET",
#     51: "fle",
#     52: "mechanical",
#     53: "ppc",
#     54: "unf"
# }

mscoco_category2name = {
    0: "architectural-plans-fXWf",
    1: "BALCONY",
    2: "BASEMENT",
    3: "BASEMENT FINISHED",
    4: "BASEMENT UNFINISHED",
    5: "BATHFULL",
    6: "BEDROOM",
    7: "BEDROOM_CLOSET",
    8: "CAFE",
    9: "CLOSET",
    10: "COVERED LANAI",
    11: "CRAWL SPACE",
    12: "DECK",
    13: "DEN",
    14: "DINING",
    15: "ENTRY",
    16: "FLEX",
    17: "FOYER",
    18: "FRONT_PORCH",
    19: "GAME ROOM",
    20: "GARAGE",
    21: "GENERAL",
    22: "HALL",
    23: "HVAC",
    24: "KITCHEN",
    25: "LAUNDRY",
    26: "LIBRARY",
    27: "LIVING",
    28: "LOFT",
    29: "MASTER_BED",
    30: "MECH",
    31: "MUDROOM",
    32: "NOOK",
    33: "OFFICE",
    34: "OPEN TO BELOW",
    35: "OWNER SUITE",
    36: "PANTRY",
    37: "PATIO",
    38: "POWDER",
    39: "PR",
    40: "RISER",
    41: "SHOWER",
    42: "STAIRS",
    43: "STORAGE",
    44: "STUDY",
    45: "WALK_IN_CLOSET",
    46: "WATER_CLOSET",
    47: "mechanical",
    48: "ppc"
}



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
        base_url = "../../dataset/BrochurePlanLabeling.v5i.coco-segmentation"
    

    for key_path in key_paths: 
        separate_coco_semantic_from_panoptic(
            os.path.join(base_url, "{}/_panoptic_annotations.coco.json".format(key_path)),
            # os.path.join(base_url, "{}/_panoptic_annotation_pulte_maskdino_augmented_file.json".format(key_path)),
            os.path.join(base_url, "panoptic_masks/{}".format(key_path)),
            # os.path.join(base_url, "panoptic_masks_maskdino_augmented/{}".format(key_path)),
            os.path.join(base_url, "panoptic_semseg_{}".format(key_path)),
            # os.path.join(base_url, "panoptic_semseg_maskdino_augmented_{}".format(key_path)),
            mscoco_category2name,
        )