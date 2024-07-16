import json
import glob
import shutil
import os


# Function to merge COCO json files
def merge_coco_json_files(base_url, datasetnames, output_dir, image_base_dir):
    merged_data = {
        "info": {
            "year": "2024",
            "version": "1",
            "description": "Merged dataset",
            "contributor": "",
            "url": "",
            "date_created": ""
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    image_id_offset = 0
    annotation_id_offset = 0

    for datasetname in datasetnames:
        url = os.path.join(base_url, datasetname, '_annotation_file.json')
        image_base_url = image_base_dir.format(datasetname, datasetname)
        print(url)
        with open(url, 'r') as f:
            data = json.load(f)
            merged_data["categories"] = data["categories"]
            
            for image in data["images"]:
                new_image = image.copy()
                new_image["id"] += image_id_offset
                # Copy image to new directory
                src_image_path = os.path.join(image_base_url, image["file_name"])
                dest_image_path = os.path.join(output_dir, image["file_name"])
                # /home/ubuntu/dataset/unlabeled/data_yourarborhome/yourarborhome/floorplans/76293600-c496-4cff-96f4-065147d77088_11_.png
                shutil.copy(src_image_path, dest_image_path)
                merged_data["images"].append(new_image)
            
            for annotation in data["annotations"]:
                new_annotation = annotation.copy()
                new_annotation["id"] += annotation_id_offset
                new_annotation["image_id"] += image_id_offset
                merged_data["annotations"].append(new_annotation)
        
        # Update the offsets for the next file
        image_id_offset = merged_data["images"][-1]["id"] + 1
        annotation_id_offset = merged_data["annotations"][-1]["id"] + 1
        
        print(f"image_id_offset is {image_id_offset}")

    return merged_data

# List of JSON files to merge

base_url = '/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/0029999'
datasetnames = ['yourarborhome',
                'yourarborhome',
                'brohn',
                'centurycommunities',
                'harrisdoyle',
                'homeplans',
                'lennar',
                'lgi',
                'nvhomes',
                'nvhomes',
                'pulte'] 

output_dir = '/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/merged/'
image_base_dir = "/home/ubuntu/dataset/unlabeled/data_{}/{}/floorplans"

# Merge the files
merged_data = merge_coco_json_files(base_url, datasetnames, output_dir, image_base_dir)

# Save the merged data to a new JSON file
with open("/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/merged_coco.json", "w") as f:
    json.dump(merged_data, f)
