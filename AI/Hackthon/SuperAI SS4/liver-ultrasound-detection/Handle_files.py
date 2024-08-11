import os
import shutil
from tqdm import tqdm

def create_yolo_structure(base_dir):
    # Define the new structure
    yolo_structure = {
        'dataset/images/train': [],
        'dataset/images/val': [],
        'dataset/images/test': [],
        'dataset/labels/train': [],
        'dataset/labels/val': []
    }
    
    # Create directories if they don't exist
    for path in yolo_structure.keys():
        full_path = os.path.join(base_dir, path)
        os.makedirs(full_path, exist_ok=True)

def copy_files(src_dir, dst_dir, file_extension):
    for root, _, files in tqdm(os.walk(src_dir)):
        for file in tqdm(files):
            if file.endswith(file_extension):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)

def main():
    base_dir = '/home/worawit.tepsan/Project_AI/Image/liver_detection'  # Adjust this path if your dataset is in a different location
    
    create_yolo_structure(base_dir)
    
    # Copy training images and labels
    copy_files(os.path.join(base_dir, 'train/train/images'), os.path.join(base_dir, 'dataset/images/train'), '.jpg')
    copy_files(os.path.join(base_dir, 'train/train/annotations'), os.path.join(base_dir, 'dataset/labels/train'), '.txt')
    
    # Copy validation images and labels
    copy_files(os.path.join(base_dir, 'val/val/images'), os.path.join(base_dir, 'dataset/images/val'), '.jpg')
    copy_files(os.path.join(base_dir, 'val/val/annotations'), os.path.join(base_dir, 'dataset/labels/val'), '.txt')
    
    # Copy test images (no labels for test)
    copy_files(os.path.join(base_dir, 'test/test/images'), os.path.join(base_dir, 'dataset/images/test'), '.jpg')
    
    print("Dataset has been reorganized for YOLOv8 format.")

if __name__ == '__main__':
    main()
