# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager
from maskdino.data.higharc_categories import HIGHARC_CATEGORIES


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "train_higharch_brochure_panoptic": (
        "train",
        "panoptic_masks/train",
        "train/_panoptic_annotations.coco.json",
        "panoptic_semseg_train",
    ),
    "val_higharch_brochure_panoptic": (
        "valid",
        "panoptic_masks/valid",
        "valid/_panoptic_annotations.coco.json",
        "panoptic_semseg_valid",
    ),
}


def get_metadata():
    meta = {}
    thing_classes = [k["name"] for k in HIGHARC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in HIGHARC_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in HIGHARC_CATEGORIES]
    stuff_colors = [k["color"] for k in HIGHARC_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors


    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(HIGHARC_CATEGORIES):
        if cat["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    
    return meta


def load_coco_panoptic_json(json_file, image_dir, panoptic_root, semseg_dir, meta):
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(panoptic_root, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "id": image_id,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_coco_panoptic_annos_sem_seg(name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    # print(f"metadata is {metadata}")
    MetadataCatalog.get(panoptic_name).set(
       **metadata,
       panoptic_json=panoptic_json,
    )

    semantic_name = name + "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        json_file=instances_json,
        panoptic_json=panoptic_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )
    

def register_all_coco_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        register_coco_panoptic_annos_sem_seg(
            name=prefix,
            metadata=get_metadata(),
            image_root=os.path.join(root, image_root),
            panoptic_root=os.path.join(root, panoptic_root),
            panoptic_json=os.path.join(root, panoptic_json),
            sem_seg_root=os.path.join(root, semantic_root),
            instances_json=os.path.join(root, image_root, "_annotations.coco.json"),
        )


_root = "/home/ubuntu/dataset/experiment_one"
register_all_coco_panoptic_annos_sem_seg(_root)
