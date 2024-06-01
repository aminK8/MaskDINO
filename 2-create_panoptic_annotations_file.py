import json
import os

def convert_coco_to_panoptic(coco_json):
    res = dict()
    res['info'] = coco_json['info']
    res['licenses'] = coco_json['licenses']
    res['categories'] = coco_json['categories'].append({
            "id": 40,
            "name": "none",
            "supercategory": "none",
            "isthing": 0
        })
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


for t in ["test", "valid", "train"]:
    # base_path = '/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation/{}/'
    base_path = "../../dataset/seg_object_detection/auto_translate_v4-3/{}/"
    
    
    anno_file = os.path.join(base_path, "_annotations.coco.json").format(t)
    print(anno_file)
    # Load the JSON file
    with open(anno_file, 'r') as f:
        coco_json = json.load(f)
        
        
    panoptic_json = convert_coco_to_panoptic(coco_json)

    # Save to a JSON file
    out_file = os.path.join(base_path, "_panoptic_annotations.coco.json").format(t)
    with open(out_file, 'w') as f:
        json.dump(panoptic_json, f, indent=2)