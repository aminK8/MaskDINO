from PIL import Image
import cv2
import numpy as np
import os


def read_image(path):
    img = Image.open(path)
    
    # If the image doesn't have an alpha channel, add one
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Check if the image has an alpha channel (transparency)
    print(img.mode)
    if img.mode == "RGBA":
        # Create a white background image
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        
        # Paste the original image onto the white background
        white_bg.paste(img, (0, 0), img)
        
        img = white_bg
    
    img = img.convert("RGB")
    
    img.save(path)
    
    
def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png')):
            read_image(file_path)
            print(f"Processed {filename}")

# Example usage
folder_path = '/home/ubuntu/dataset/expriment_three_1/test'
process_images_in_folder(folder_path)

