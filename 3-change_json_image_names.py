import json




dataset_type = "expriment_three"
base_url = ""

if dataset_type == "construction":
    files_name = ["valid", "test", "train"]
    # base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"
    base_url = "../../dataset/seg_object_detection/auto_translate_v4-3/{}/_annotations.coco.json"
    
elif dataset_type == 'pulte_unlabel':
    files_name = ['floorplans']
    base_url = "../../dataset/data_pulte/pulte/{}/_annotations.coco.json"

elif dataset_type == 'pulte_lable_81':
    files_name = ["valid", "train"]
    base_url = "../../dataset/BrochurePlanLabeling.v5i.coco-segmentation/{}/_annotations.coco.json"
    
elif dataset_type == 'pseudo':
    files_name = ["train"]
    base_url = "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/{}/_annotations.coco.json"
    
elif dataset_type == 'expriment_three':
    files_name = ["test", "train"]
    base_url = "../../dataset/expriment_three_1/{}/_annotations.coco.json"
    

for file_name in files_name:
    path = base_url.format(file_name)
    with open(path, 'r') as file:
        data = json.load(file)

    # Change file names from .jpg to .png in the "images" section
    for image in data.get('images', []):
        if image.get('file_name'):
            image['file_name'] = image['file_name'][:-4] + ".png"
            
    # Write the updated content to a new JSON file
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"File names have been updated and saved to {path}")
            
            
if dataset_type == "construction":
    # base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"
    base_url = "../../dataset/seg_object_detection/auto_translate_v4-3/{}/_panoptic_annotations.coco.json"
    
elif dataset_type == 'pulte_unlabel':
    files_name = ['floorplans']
    base_url = "../../dataset/data_pulte/pulte/{}/_panoptic_annotations.coco.json"

elif dataset_type == 'pulte_lable_81':
    files_name = ["valid", "train"]
    base_url = "../../dataset/experiment_two/{}/_panoptic_annotations.coco.json"
  
elif dataset_type == 'pseudo':
    files_name = ["train"]
    base_url = "/home/ubuntu/code/MaskDINO/output_experiment_two/output/pseudo/{}/_panoptic_annotations.coco.json"

elif dataset_type == 'expriment_three':
    files_name = ["test", "train"]
    base_url = "../../dataset/expriment_three_1/{}/_panoptic_annotations.coco.json"    

for file_name in files_name:
    path = base_url.format(file_name)
    with open(path, 'r') as file:
        data = json.load(file)

    # Update file names from .jpg to .png in the "images" section
    for image in data.get('images', []):
        if image.get('file_name'):
            image['file_name'] = image['file_name'].replace('.jpg', '.png')
    
    # Update file names from .jpg to .png in the "annotations" section
    for annotation in data.get('annotations', []):
        if annotation.get('file_name'):
            annotation['file_name'] = annotation['file_name'].replace('.jpg', '.png')

    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"File names have been updated and saved to {path}")