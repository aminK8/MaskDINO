import os
from PIL import Image, ImageDraw, ImageFont

def combine_images(base_url, dirs, output_path):
    # Full paths to the directories
    full_image_paths = [os.path.join(base_url, dir_path) for dir_path in dirs]

    # List of images in each directory
    images_lists = [sorted(os.listdir(path)) for path in full_image_paths]

    # Ensure there is a corresponding image in each directory
    common_images = set(images_lists[0]).intersection(*images_lists[1:])
    
    for image_name in common_images:
        # Load images from all directories
        imgs = [Image.open(os.path.join(path, image_name)) for path in full_image_paths]

        # Resize images to the same height
        heights = [img.height for img in imgs]
        min_height = min(heights)
        imgs = [img.resize((img.width, min_height)) for img in imgs]
        
        # Create a new image with width = sum of all image widths and height = max height of all images
        combined_width = sum(img.width for img in imgs)
        combined_height = min_height + 50  # Add space for the labels
        combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        
        # Paste the images into the new image
        current_x = 0
        for i, img in enumerate(imgs):
            combined_image.paste(img, (current_x, 50))
            current_x += img.width
        
        # Draw the labels
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.load_default()
        current_x = 0
        for i, dir_path in enumerate(dirs):
            draw.text((current_x + imgs[i].width // 2, 10), dir_path, fill="black", font_size=40)
            current_x += imgs[i].width
        
        # Save the combined image
        combined_image.save(os.path.join(output_path, image_name))

# Define paths
base_url = "/Users/amin/Desktop/higharc/MaskDINO/demo"  # Change this to your actual base URL
dirs = ["pulte_data_with_ssl", "pulte_data_without_ssl", "YOLO", "Pulte"]
output_path = "/Users/amin/Desktop/higharc/MaskDINO/demo/combined_images_pulte1"

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Combine images
combine_images(base_url, dirs, output_path)

print("Images combined and saved in:", output_path)
