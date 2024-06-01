import json
import os

def add_isthing_to_categories(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    if 'categories' in data:
        for category in data['categories']:
            if category['id'] != 0:
                category['isthing'] = 1
            else:
                category['isthing'] = 0
                
        categories = data['categories']
        categories.append({
                "id": 40,
                "name": "none",
                "supercategory": "none",
                "isthing": 0
            })
        data['categories'] = categories
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

def process_json_files_in_directory(json_file):
    add_isthing_to_categories(json_file)
    print(f"Added 'isthing' to categories in {json_file}")

# Change 'directory_path' to the directory containing your JSON files
for key_path in ["valid", "test", "train"]:
    base_url = "../../dataset/seg_object_detection/auto_translate_v4-3"
    # base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"
    # Directory containing images
    process_json_files_in_directory(os.path.join(base_url, key_path, '_annotations.coco.json'))
