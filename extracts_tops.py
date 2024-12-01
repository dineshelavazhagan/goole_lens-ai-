import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define HSV color ranges for tops (white and light whitish grey)
# Adjust these ranges based on your actual mask colors
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

lower_light_grey = np.array([0, 0, 180])
upper_light_grey = np.array([180, 30, 220])

def extract_tops(image_path, mask_path, top_region_proportion=0.5):
    """
    Extracts top clothing from the image based on the segmentation mask.
    Areas outside the extracted tops are set to white.

    Parameters:
    - image_path: Path to the original image.
    - mask_path: Path to the segmentation mask.
    - top_region_proportion: Proportion of the image height to consider from the top.

    Returns:
    - tops_image: Image containing only the top clothing areas with a white background.
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image file {image_path}.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the segmentation mask
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Error: Unable to read mask file {mask_path}.")
        return None
    mask_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Create masks for white and light whitish grey colors
    mask_white = cv2.inRange(mask_hsv, lower_white, upper_white)
    mask_light_grey = cv2.inRange(mask_hsv, lower_light_grey, upper_light_grey)

    # Combine the masks to cover both white and light whitish grey regions
    mask_tops = cv2.bitwise_or(mask_white, mask_light_grey)

    # Apply spatial constraints
    height, width = mask_tops.shape
    top_region_height = int(height * top_region_proportion)
    spatial_mask = np.zeros_like(mask_tops)
    spatial_mask[0:top_region_height, :] = 255  # Set upper region to 255

    # Combine color mask with spatial mask
    final_mask = cv2.bitwise_and(mask_tops, spatial_mask)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Create a white background image
    white_background = np.ones_like(image, dtype=np.uint8) * 255  # White background

    # Extract the top clothing areas from the original image
    tops = cv2.bitwise_and(image, image, mask=final_mask)

    # Invert the mask to get the background areas
    mask_inv = cv2.bitwise_not(final_mask)

    # Extract the background from the white image using the inverted mask
    background = cv2.bitwise_and(white_background, white_background, mask=mask_inv)

    # Combine the tops with the white background
    tops_image = cv2.add(tops, background)

    return tops_image

def visualize_top_extraction(image_path, mask_path, tops_image):
    """
    Visualizes the original image, segmentation mask, and extracted tops.

    Parameters:
    - image_path: Path to the original image.
    - mask_path: Path to the segmentation mask.
    - tops_image: Extracted tops image.
    """
    # Load images for visualization
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Unable to read image file {image_path} for visualization.")
        return
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Error: Unable to read mask file {mask_path} for visualization.")
        return
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title('Segmentation Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(tops_image)
    plt.title('Extracted Tops')
    plt.axis('off')

    plt.show()

def main():
    # Define directories
    images_dir = 'images/'        # Directory containing original images
    masks_dir = 'segm/'           # Directory containing segmentation masks
    output_dir = 'tops/'          # Directory to save extracted top clothing images

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # List all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

    print(f"Total images found: {len(image_files)}")

    # Process each image-mask pair
    for image_file in tqdm(image_files, desc='Processing Images'):
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)  # Assuming mask has the same filename
        mask_path = mask_path.replace(".jpg", "_segm.png")
        # Check if mask exists
        # print("mask", mask_path)
        if not os.path.exists(mask_path):
            # print(f"Warning: Mask for image {mask_path} not found. Skipping.")
            continue

        # Extract tops
        tops = extract_tops(image_path, mask_path, top_region_proportion=0.5)  # Adjust proportion if needed

        if tops is not None:
            # Convert RGB back to BGR for saving with OpenCV
            tops_bgr = cv2.cvtColor(tops, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, tops_bgr)
        else:
            print(f"Warning: Top extraction failed for image {image_file}.")

    print("Top extraction completed.")

    # Visualize a few samples
    sample_files = image_files[:5]  # Visualize first 5 images; adjust as needed

    for sample_file in sample_files:
        sample_image_path = os.path.join(images_dir, sample_file)
        sample_mask_path = os.path.join(masks_dir, sample_file)
        sample_output_path = os.path.join(output_dir, sample_file)

        # Load the extracted tops image
        tops = cv2.imread(sample_output_path)
        if tops is not None:
            tops = cv2.cvtColor(tops, cv2.COLOR_BGR2RGB)
            visualize_top_extraction(sample_image_path, sample_mask_path, tops)
        else:
            print(f"Error: Unable to load extracted tops for {sample_file}.")

if __name__ == "__main__":
    main()
