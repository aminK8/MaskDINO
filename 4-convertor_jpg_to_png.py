import os
from PIL import Image

def convert_images(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for file in files:
        # Check if it's a JPG file
        if file.endswith(".jpg"):
            # Open the image
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path)
            
            # Convert to PNG
            png_path = img_path[:-4] + ".png"
            img.save(png_path, "PNG")
            
            print(f"Converted {file} to PNG")

# List of folders containing images
folders = ["panoptic_semseg_valid", "panoptic_semseg_train", "panoptic_semseg_test"]  # Add your folder names here

base_url = "../../dataset/seg_object_detection/auto_translate_v4-3"
# base_url = "/Users/amin/Desktop/higharc/Datasets/Laleled-2024-05-29/auto_translate_v4.v3i.coco-segmentation"

# Iterate through each folder and convert images
for folder in folders:
    folder_path = os.path.join(base_url, folder)
    if os.path.isdir(folder_path):
        convert_images(folder_path)
    else:
        print(f"{folder} is not a valid directory.")