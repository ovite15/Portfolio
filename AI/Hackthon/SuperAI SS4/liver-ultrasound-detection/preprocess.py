import os
import random
import shutil
import cv2
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.exposure import match_histograms
from skimage.color import rgb2gray
from skimage.transform import resize
import albumentations as A
import matplotlib.pyplot as plt

# Load the mapping file
mapping = pl.read_excel("./mapping.xlsx")

# Filter the data according to the specified conditions
machine = mapping.filter(pl.col("Source") == "machine").filter(pl.col("Type").is_in(["machine_positive", "machine_negative"]))
mobile = mapping.filter(pl.col("Source") == "mobile").filter(pl.col("Type").is_in(["mobile_positive", "mobile_negative"]))

def read_image(file_name):
    path_img = f"./process_data/images/data/{file_name}.jpg"
    if os.path.exists(path_img):
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return None

# Define the output directories for images and labels
output_dir = './augmented_data2'
images_dir = os.path.join(output_dir, 'images')
labels_dir = os.path.join(output_dir, 'labels')

# Create necessary directories
os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

# Define the proportion of data to use for validation
val_ratio = 0.34

# List to store the image file paths for splitting
image_files = []

# Set random seed for transformation reproducibility
transform_seed = 99

# Transformation definition
transform = A.Compose(
    [
        A.CLAHE(clip_limit=3, p=1),
        A.HistogramMatching(reference_images=[], p=1, read_fn=lambda x: np.array(x)),
        A.PixelDistributionAdaptation(reference_images=[], p=1, read_fn=lambda x: np.array(x)),
        A.Sharpen(p=1),
        A.HueSaturationValue(p=1)
    ],
)

data_m = machine["Image File"].to_list()
data_m = list(map(lambda x: x.split('.')[0], data_m))

# List of values to choose from for smobile
smobile_values = [3643, 1182, 23829]

# Counters for annotation files
copied_annotations = 0
created_empty_annotations = 0

for i in data_m:
    img = read_image(i)
    if img is None:
        continue

    # Randomly select a value from smobile_values
    smobile_id = str(random.choice(smobile_values))
    
    # Filter the mobile dataframe to get the image with the selected smobile_id
    reference = read_image(smobile_id)
    if reference is None:
        continue
    
    # Set the seed for the transformation
    np.random.seed(transform_seed)
    random.seed(transform_seed)
    
    # Update reference images for transformations
    transform.transforms[1].reference_images = [reference]
    transform.transforms[2].reference_images = [reference]
    
    transformed = transform(image=img)["image"]
    
    # Reset the random seed to allow random operations again
    np.random.seed(None)
    random.seed(None)
    
    # Determine output path and name
    file_name = f"transformed_{i}"
    output_image_path = os.path.join(output_dir, f"{file_name}.jpg")
    
    # Save the transformed image
    cv2.imwrite(output_image_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
    
    # Check for the existence of the original annotation file
    original_annotation_path = f"./process_data/images/data/{i}.txt"
    
    output_annotation_path = os.path.join(output_dir, f"{file_name}.txt")
    if os.path.exists(original_annotation_path):
        shutil.copy2(original_annotation_path, output_annotation_path)
        copied_annotations += 1
    else:
        with open(output_annotation_path, 'w') as f:
            f.write("")  # Write empty annotation file
        created_empty_annotations += 1
    
    # Store the image file path
    image_files.append(f"{file_name}.jpg")

# Shuffle and split the data into training and validation sets
random.shuffle(image_files)
split_index = int(len(image_files) * (1 - val_ratio))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Function to move files to the train/val directories
def move_files(files, image_target_dir, label_target_dir):
    for file in files:
        image_src = os.path.join(output_dir, file)
        annotation_src = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.txt")
        image_dst = os.path.join(image_target_dir, file)
        annotation_dst = os.path.join(label_target_dir, f"{os.path.splitext(file)[0]}.txt")
        
        os.makedirs(os.path.dirname(image_dst), exist_ok=True)
        os.makedirs(os.path.dirname(annotation_dst), exist_ok=True)
        
        shutil.move(image_src, image_dst)
        shutil.move(annotation_src, annotation_dst)

# Move files to respective directories
move_files(train_files, os.path.join(images_dir, 'train'), os.path.join(labels_dir, 'train'))
move_files(val_files, os.path.join(images_dir, 'val'), os.path.join(labels_dir, 'val'))

print("Data augmentation completed, images and annotations saved, and data split into training and validation sets.")
print(f"Copied {copied_annotations} annotation files.")
print(f"Created {created_empty_annotations} empty annotation files.")

# Define the source folders and the destination folder for final organization
def copy_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.relpath(root, src_dir)
            dest_file_dir = os.path.join(dest_dir, rel_path)
            os.makedirs(dest_file_dir, exist_ok=True)
            dest_file = os.path.join(dest_file_dir, file)
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

def main():
    # Define source and destination directories
    augmented_data_images_train = './augmented_data2/images/train'
    augmented_data_images_val = './augmented_data2/images/val'
    augmented_data_labels_train = './augmented_data2/labels/train'
    augmented_data_labels_val = './augmented_data2/labels/val'

    new_data_images_train = './new_data1/images/train'
    new_data_images_val = './new_data1/images/val'
    new_data_labels_train = './new_data1/labels/train'
    new_data_labels_val = './new_data1/labels/val'

    # Copy files from augmented_data to new_data
    copy_files(augmented_data_images_train, new_data_images_train)
    copy_files(augmented_data_images_val, new_data_images_val)
    copy_files(augmented_data_labels_train, new_data_labels_train)
    copy_files(augmented_data_labels_val, new_data_labels_val)

if __name__ == '__main__':
    main()
