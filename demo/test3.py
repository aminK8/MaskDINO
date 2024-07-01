import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('/Users/amin/Desktop/higharc/magicplan/puzzle_fussion_magic_plan/magic_plan_dataset_puzzle_fusion/0_4.npz')

print("Keys in the .npz file:")
for key in data.keys():
    print(key)
# Assuming the image is stored under a key, e.g., 'image'
image = data['room_array']

# Display the image using matplotlib
plt.imshow(image)
plt.axis('off')  # Optional: to hide axes
plt.show()