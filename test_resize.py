import cv2
import os
import time

# Input and output directories
input_folder = './dataset/Dragonfruit'
output_folder = './dataset/Processed_Dragonfruit'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
start = time.time()
# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filter image files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Read the image
        image = cv2.imread(input_path)
        
        if image is not None:  # Ensure the image was read successfully
            # Perform some operation on the image (e.g., resizing)
            processed_image = cv2.resize(image, (512, 512))
            
            # Save the processed image to the output folder
            cv2.imwrite(output_path, processed_image)
        else:
            print(f"Warning: Could not read {input_path}")

end = time.time()
print(end-start)