import json
import os

def add_isthing_to_categories(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    if 'categories' in data:
        for category in data['categories']:
            if category['id'] != len(data['categories']) -1:
                category['isthing'] = 1
            else:
                category['isthing'] = 0
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

def process_json_files_in_directory(json_file):
    add_isthing_to_categories(json_file)
    print(f"Added 'isthing' to categories in {json_file}")

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
    process_json_files_in_directory(os.path.join(base_url, key_path, '_annotations.coco.json'))
