import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define HSV color ranges for clothing (top and bottom)
# Adjust these ranges based on your actual mask colors
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

lower_grey = np.array([0, 0, 50])
upper_grey = np.array([180, 50, 200])

def extract_clothing(image_path, mask_path):
    """
    Extracts clothing from the image based on the segmentation mask.

    Parameters:
    - image_path: Path to the original image.
    - mask_path: Path to the segmentation mask.

    Returns:
    - clothing_image: Image containing only the clothing areas. Background is black.
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
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Create masks for white and grey colors
    mask_white = cv2.inRange(mask, lower_white, upper_white)
    mask_grey = cv2.inRange(mask, lower_grey, upper_grey)

    # Combine the masks
    mask_clothes = cv2.bitwise_or(mask_white, mask_grey)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask_clothes = cv2.morphologyEx(mask_clothes, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clothes = cv2.morphologyEx(mask_clothes, cv2.MORPH_DILATE, kernel, iterations=1)

    # Apply the mask to the original image
    clothing_image = cv2.bitwise_and(image, image, mask=mask_clothes)

    return clothing_image

def visualize_clothing_extraction(image_path, mask_path, clothing_image):
    """
    Visualizes the original image, segmentation mask, and extracted clothing.

    Parameters:
    - image_path: Path to the original image.
    - mask_path: Path to the segmentation mask.
    - clothing_image: Extracted clothing image.
    """
    # Load images for visualization
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)
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
    plt.imshow(clothing_image)
    plt.title('Extracted Clothing')
    plt.axis('off')

    plt.show()

def main():
    # Define directories
    images_dir = 'check/'        # Directory containing original images
    masks_dir = 'segm/'           # Directory containing segmentation masks
    output_dir = 'clothes/'       # Directory to save extracted clothing images

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # List all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

    print(f"Total images found: {len(image_files)}")

    # Process each image-mask pair
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)  # Assuming mask has the same filename
        mask_path = mask_path.replace(".jpg","_segm.png")
        print("================================",image_path, mask_path)

        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: Mask for image {image_file} not found. Skipping.")
            continue

        # Extract clothing
        clothing = extract_clothing(image_path, mask_path)

        if clothing is not None:
            # Convert RGB back to BGR for saving with OpenCV
            clothing_bgr = cv2.cvtColor(clothing, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, clothing_bgr)
        else:
            print(f"Warning: Clothing extraction failed for image {image_file}.")

    print("Clothing extraction completed.")

    # Visualize a few samples
    sample_files = image_files[:5]  # Visualize first 5 images
    for sample_file in sample_files:
        sample_image_path = os.path.join(images_dir, sample_file)
        sample_mask_path = os.path.join(masks_dir, sample_file)
        sample_output_path = os.path.join(output_dir, sample_file)

        # Load the extracted clothing image
        print(sample_output_path)
        clothing = cv2.imread(sample_output_path)
        if clothing is not None:
            clothing = cv2.cvtColor(clothing, cv2.COLOR_BGR2RGB)
            visualize_clothing_extraction(sample_image_path, sample_mask_path, clothing)
        else:
            print(f"Error: Unable to load extracted clothing for {sample_file}.")

if __name__ == "__main__":
    main()
