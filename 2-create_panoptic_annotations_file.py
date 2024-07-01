import json
import os

def convert_coco_to_panoptic(coco_json):
    res = dict()
    if 'info' in coco_json:
        res['info'] = coco_json['info']
    else:
        res['info'] = []
        
    if 'licenses' in coco_json:
        res['licenses'] = coco_json['licenses']
    else:
        res['licenses'] = []
    res['categories'] = coco_json['categories']
    res['images'] = coco_json['images']

    panoptic_annotations = []
    for image_info in coco_json['images']:
        panoptic_annotation = {
            'id': image_info['id'],
            "file_name": image_info['file_name'],
            "image_id": image_info['id'],
            "segments_info": []
        }
        for annotation in coco_json['annotations']:
            if annotation['image_id'] == image_info['id']:
                segment_info = {
                    "id": annotation['id'],
                    "category_id": annotation['category_id'],
                    "iscrowd": annotation['iscrowd'],
                    "bbox": annotation['bbox'],
                    "segmentation": annotation['segmentation'],
                    "area": annotation['area']
                }
                panoptic_annotation["segments_info"].append(segment_info)
        panoptic_annotations.append(panoptic_annotation)

    res['annotations'] = panoptic_annotations
    return res



dataset_type = "pulte_lable_81"
key_paths = []
base_url = ""


if dataset_type == "construction":
    key_paths = ["valid", "test", "train"]
    # base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"
    base_url = "../../dataset/seg_object_detection/auto_translate_v4-3/{}/"
    
elif dataset_type == 'pulte_unlabel':
    key_paths = ['floorplans']
    base_url = "../../dataset/data_pulte/pulte/{}/"

elif dataset_type == 'pulte_lable_81':
    key_paths = ["valid", "train"]
    base_url = "../../dataset/BrochurePlanLabeling.v5i.coco-segmentation/{}"

for t in key_paths:    
    anno_file = os.path.join(base_url, "_annotations.coco.json").format(t)
    print(anno_file)
    # Load the JSON file
    with open(anno_file, 'r') as f:
        coco_json = json.load(f)
        
    panoptic_json = convert_coco_to_panoptic(coco_json)

    # Save to a JSON file
    out_file = os.path.join(base_url, "_panoptic_annotations.coco.json").format(t)
    with open(out_file, 'w') as f:
        json.dump(panoptic_json, f, indent=2)