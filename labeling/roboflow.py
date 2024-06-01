import os
import roboflow
from roboflow import Roboflow
from random import random


roboflow.login()

rf = Roboflow(api_key="YJrno5yIfEo3z5kTMsJm")
project = rf.workspace("higharc-s9bm5").project("auto_translate_v4")

def predict_masks(project, version, data_image_path, images):
    model = project.version(version).model
    results = []
    for sample_name in images:
        image_path = os.path.join(data_image_path, sample_name)
        print(f"Processing {image_path}")
        result = model.predict(image_path, confidence=25).json()
        results.append(result)
    return results

def pick_random_samples(data_image_path, data_label_path, size=10):
    all_training_samples = [f for f in os.listdir(data_image_path) if os.path.isfile(os.path.join(data_image_path, f)) and f.endswith('.jpg')]
    training_subset_images = random.sample(all_training_samples, size)
    training_subset_labels = [os.path.join(data_label_path, sample_name.replace('.jpg', '.txt')) for sample_name in training_subset_images]
    return training_subset_images, training_subset_labels


training_data_main_path = 'auto_translate_v4-1/train'
data_image_path = os.path.join(training_data_main_path, 'images')
data_label_path = os.path.join(training_data_main_path, 'labels')
training_subset_images, training_subset_labels = pick_random_samples(data_image_path = data_image_path,data_label_path=data_label_path, size=size)

results = predict_masks(project=project, 
                    version= 1, # change the version to whatever version of data you use
                    data_image_path = data_image_path, 
                    images = training_subset_images)