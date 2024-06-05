import os
from PIL import Image, ImageDraw, ImageFont

def combine_images(base_url, image_with_ssl_path, image_without_ssl_path, output_path):
    # Full paths to the directories
    full_image_with_ssl_path = os.path.join(base_url, image_with_ssl_path)
    full_image_without_ssl_path = os.path.join(base_url, image_without_ssl_path)

    # List of images in each directory
    images_with_ssl = sorted(os.listdir(full_image_with_ssl_path))
    images_without_ssl = sorted(os.listdir(full_image_without_ssl_path))

    # Ensure there is a corresponding image in each directory
    common_images = set(images_with_ssl).intersection(images_without_ssl)
    
    for image_name in common_images:
        # Load images
        img1 = Image.open(os.path.join(full_image_with_ssl_path, image_name))
        img2 = Image.open(os.path.join(full_image_without_ssl_path, image_name))
        
        # Resize images to the same height (if needed)
        img1 = img1.resize((img1.width, img2.height))
        img2 = img2.resize((img2.width, img1.height))
        
        # Create a new image with width = img1.width + img2.width and height = max height of both images
        combined_width = img1.width + img2.width
        combined_height = max(img1.height, img2.height) + 50  # Add space for the labels
        combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        
        # Paste the two images into the new image
        combined_image.paste(img1, (0, 50))
        combined_image.paste(img2, (img1.width, 50))
        
        # Draw the labels
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.load_default()
        draw.text((img1.width // 2, 10), "image_with_ssl", fill="black", font=font)
        draw.text((img1.width + img2.width // 2, 10), "image_without_ssl", fill="black", font=font)
        
        # Save the combined image
        combined_image.save(os.path.join(output_path, image_name))

# Define paths
base_url = "/Users/amin/Desktop/higharc/MaskDINO/demo"  # Change this to your actual base URL
image_with_ssl_path = "pulte_data_with_ssl"
image_without_ssl_path = "pulte_data_without_ssl"
output_path = "/Users/amin/Desktop/higharc/MaskDINO/demo/combined_images_pulte"

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Combine images
combine_images(base_url, image_with_ssl_path, image_without_ssl_path, output_path)

print("Images combined and saved in:", output_path)
