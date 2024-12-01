import os

import shutil
image_dir = "tops/"
output_dir = "labels_sep/"
os.makedirs(output_dir, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(image_extensions)]

for each_image in image_paths:
    print(each_image.split("-")[0],each_image.split("-")[1])
    save_dir = output_dir+each_image.split("-")[0]+"/"+each_image.split("-")[1] 
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(each_image, save_dir)
    # break
