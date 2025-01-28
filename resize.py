import cv2
import os
import time

# Input and output directories
input_folders = ['./dataset/Dragonfruit','./dataset/Egg','./dataset/Eggplant','./dataset/unknown_drew','./dataset/unknown_manbir','./dataset/unknown_yh']
output_folder = ['./processed_data/Processed_Dragonfruit','./processed_data/Processed_Egg','./processed_data/Processed_Eggplant','./processed_data/Processed_Unknown_Drew','./processed_data/Processed_Unknown_Manbir','./processed_data/Processed_Unknown_YH']

# Create output directory if it doesn't exist
for folder in output_folder:
    os.makedirs(folder, exist_ok=True)


def crop_center(image, crop_size):
    """
    Crops the center of an image to the specified size.
    
    Args:
        image: Input image as a NumPy array.
        crop_size: Tuple (height, width) for the crop size.
    
    Returns:
        Cropped image as a NumPy array.
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    
    # Check if image is smaller than the cropped dimesions
    if crop_h > h or crop_w > w:
        print(crop_h, h , crop_w, w)
        raise ValueError("Crop size exceeds image dimensions")
    
    # Calculate the center crop coordinates
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]

start = time.time()

# Loop through all folders and files in the input folders
for folder in input_folders:
    folder_start = time.time()
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png', '.jpeg','.JPG')):  # Filter image files
            input_path = os.path.join(folder, filename)
            idx = input_folders.index(folder)
            output_path = os.path.join(output_folder[idx], filename)
            
            # Read the image
            image = cv2.imread(input_path)
            
            if image is not None:  # Ensure the image was read successfully
                h, w = image.shape[:2]
                # rotate the image if wrong orientation
                if w > h:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                h, w = image.shape[:2]
                # Shrink Image before cropping
                if w == 960:
                    resized_image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA) #original res 960x1280 -> 384x512
                elif w == 1200:
                    resized_image = cv2.resize(image, None, fx=0.32, fy=0.32, interpolation=cv2.INTER_AREA) #original res 1200x1600 -> 384x512
                elif w == 3024:
                    resized_image = cv2.resize(image, None, fx=0.127, fy=0.127, interpolation=cv2.INTER_AREA) #original res 3024x4032 -> 384x512
                elif w == 1080:
                    resized_image = cv2.resize(image, None, fx=0.38, fy=0.38, interpolation=cv2.INTER_AREA) #original res 1080x2400 -> 410x912 [to be cropped]
                elif w == 540:
                    resized_image = cv2.resize(image, None, fx=0.72, fy=0.72, interpolation=cv2.INTER_AREA) #original res 540x720 -> 386x515 [to be cropped]
                else:
                    resized_image = image
                # crop to 384x512 resolution 
                try:
                    processed_image = crop_center(resized_image, (384, 512))
                
                    # Save the processed image to the output folder
                    cv2.imwrite(output_path, processed_image)
                except ValueError as e:
                    print(f"Skipping {filename}: {e}")
            else:
                print(f"Warning: Could not read {folder}")
    folder_end = time.time()
    print(f"Done processing: {folder} | Time elapsed: {folder_end - folder_start:.2f} seconds")

end = time.time()
print(f"Total Processing time: {end - start:.2f} seconds")
