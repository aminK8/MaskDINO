import os
import roboflow
import random
import json
from PIL import Image
import io


rf = roboflow.Roboflow(api_key="YJrno5yIfEo3z5kTMsJm")
project = rf.workspace("higharc-s9bm5").project("auto_translate_v4")

def predict_masks(project, version, data_image_path, images):
    model = project.version(version).model
    results = []
    for sample_name in images:
        image_path = os.path.join(data_image_path, sample_name)
        print(f"Processing {image_path}")
        result = predict_masks_for_image(model, image_path, confidence=25)
        # result = model.predict(image_path, confidence=25).json()
        results.append(result)
    return results

def predict_masks_for_image(model, image_path, confidence=25):
    # Open the image
    image = Image.open(image_path)
    
    # Convert the image to RGB if it is in 'P' mode
    if image.mode == 'P':
        image = image.convert('RGB')
        image.save(image_path, format="PNG")

    
    # Continue with your prediction logic
    result = model.predict(image_path, confidence=confidence).json()
    return result

def pick_random_samples(data_image_path, data_label_path=None, size=10):
    all_training_samples = [f for f in os.listdir(data_image_path) if os.path.isfile(os.path.join(data_image_path, f)) and f.endswith('.png')]
    
    if size is None:
        size = len(all_training_samples)
    training_subset_images = random.sample(all_training_samples, size)
    training_subset_labels = None
    if data_label_path != None:
        training_subset_labels = [os.path.join(data_label_path, sample_name.replace('.jpg', '.txt')) for sample_name in training_subset_images]
    return training_subset_images, training_subset_labels


training_data_main_path = '/Users/amin/Desktop/higharc/Datasets/unlabeled/data_pulte/pulte/'
data_image_path = os.path.join(training_data_main_path, 'floorplans')
size=None
# data_label_path = os.path.join(training_data_main_path, 'labels')
training_subset_images, training_subset_labels = pick_random_samples(data_image_path = data_image_path,data_label_path=None, size=size)

print("data is picked")
results = predict_masks(project=project, 
                    version= 2, # change the version to whatever version of data you use
                    data_image_path = data_image_path, 
                    images = training_subset_images)


coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}



HIGHARC_CATEGORIES = [
    {'color': [250, 141, 255], 'isthing': 1, 'id': 0, 'name': 'architectural-plans-kBh5'},
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'bath'},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bed_closet'}, 
    {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'bed_pass'},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'bedroom'}, 
    {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'chase'},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'closet'},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'dining'}, 
    {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'entry'},
    {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'fireplace'},
    {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': 'flex'},
    {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': 'foyer'},
    {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'front_porch'},
    {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'garage'},     
    {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'general'},
    {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'hall'}, 
    {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'hall_cased_opening'}, 
    {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'kitchen'}, 
    {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'laundry'},     
    {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'living'},
    {'color': [102, 102, 156], 'isthing': 1, 'id': 20, 'name': 'master_bed'},
    {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'master_closet'}, 
    {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'master_hall'}, 
    {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'master_vestibule'}, 
    {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'mech'}, 
    {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'mudroom'}, 
    {'color': [255, 179, 240], 'isthing': 1, 'id': 26, 'name': 'office'},
    {'color': [0, 125, 92], 'isthing': 1, 'id': 27, 'name': 'pantry'}, 
    {'color': [209, 0, 151], 'isthing': 1, 'id': 28, 'name': 'patio'},
    {'color': [188, 208, 182], 'isthing': 1, 'id': 29, 'name': 'portico'},
    {'color': [0, 220, 176], 'isthing': 1, 'id': 30, 'name': 'powder'},
    {'color': [255, 99, 164], 'isthing': 1, 'id': 31, 'name': 'reach_closet'},
    {'color': [92, 0, 73], 'isthing': 1, 'id': 32, 'name': 'reading_nook'},
    {'color': [133, 129, 255], 'isthing': 1, 'id': 33, 'name': 'rear_porch'}, 
    {'color': [78, 180, 255], 'isthing': 1, 'id': 34, 'name': 'solarium'}, 
    {'color': [0, 228, 0], 'isthing': 1, 'id': 35, 'name': 'stairs_editor'}, 
    {'color': [174, 255, 243], 'isthing': 1, 'id': 36, 'name': 'util_hall'},
    {'color': [45, 89, 255], 'isthing': 1, 'id': 37, 'name': 'walk'},
    {'color': [134, 134, 103], 'isthing': 1, 'id': 38, 'name': 'water_closet'},
    {'color': [145, 148, 174], 'isthing': 0, 'id': 39, 'name': 'workshop'},
]

coco_format['categories'] = HIGHARC_CATEGORIES

# Populate the images and annotations
annotation_id = 1
for image_id, result in enumerate(results, start=1):
    image_paths = set()
    for prediction in result['predictions']:
        image_path = prediction['image_path']
        if image_path not in image_paths:
            image_info = {
                "id": image_id,
                "width": int(result['image']['width']),
                "height": int(result['image']['height']),
                "file_name": image_path
            }
            coco_format['images'].append(image_info)
            image_paths.add(image_path)

        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        category_id = prediction['class_id']
                
        segmentation = []
        for point in prediction['points']:
            segmentation.extend([point['x'], point['y']])

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0,
            "segmentation": segmentation,
            "confidence": prediction['confidence'],
            "detection_id": prediction['detection_id']
        }
        coco_format['annotations'].append(annotation)
        annotation_id += 1
        
output_path = os.path.join(training_data_main_path, '_annotations.coco.json')
with open(output_path, 'w') as f:
    json.dump(coco_format, f, indent=4)

print("Conversion to COCO format complete.")