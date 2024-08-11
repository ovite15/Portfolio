import torch
import pandas as pd
from ultralytics import YOLOv10
import os

print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Load the model
model = YOLOv10('/project/lt900118-ai24o6/IT/training/jupyter/runs/detect/train33/weights/best.pt')

# Run inference on the specified source with confidence threshold
results = model.predict(
    source='/project/lt900118-ai24o6/IT/training/jupyter/dataset/images/test',
    device=[0,1,2,3],
    save=True,
    conf=0.6,
    project='results',
    stream=True
)

def convert_to_pascal_voc(box):
    """
    Converts a bounding box from (x_center, y_center, width, height) format to
    Pascal VOC (xmin, ymin, xmax, ymax) format.
    """
    x_center, y_center, width, height = box
    xmin = x_center - (width / 2)
    ymin = y_center - (height / 2)
    xmax = x_center + (width / 2)
    ymax = y_center + (height / 2)
    return [xmin, ymin, xmax, ymax]

# Prepare data for the submission file
submission_data = []

for r in results:
    if hasattr(r, 'boxes'):
        boxes = r.boxes.xywh  # Assuming boxes are in (x_center, y_center, width, height) format
        labels = r.boxes.cls  # Class labels

        # Convert each box to Pascal VOC format
        pascal_voc_boxes = [convert_to_pascal_voc(box) for box in boxes.cpu().numpy()]
        
        # Prepare the annotation and label data
        image_file = os.path.basename(r.path)  # Get the filename with extension
        image_number = os.path.splitext(image_file)[0]  # Remove the extension to get the number
        
        # Collect all annotations and labels for the current image
        all_annotations = [str(bbox) for bbox in pascal_voc_boxes]
        all_labels = [int(label.item()) for label in labels]
        
        # Append to submission data
        submission_data.append([image_number, f'[{",".join(all_annotations)}]', f'[{",".join(map(str, all_labels))}]'])
    else:
        print("No bounding boxes found in the result.")

# Create a DataFrame for the submission
submission_df = pd.DataFrame(submission_data, columns=["Image File", "Annotation", "Label"])
submission_df['Image File'] = submission_df['Image File'].astype('int64')

sample = pd.read_csv("/project/lt900118-ai24o6/IT/training/jupyter/sample_submission.csv")
merged_df = pd.merge(sample, submission_df, on='Image File')

sub = merged_df.copy()
sub['Annotation_x'][3:] = sub['Annotation_y'][3:]
sub['Label_x'][3:] = sub['Label_y'][3:]
sub = sub.drop(columns=['Annotation_y','Label_y'])
sub.rename(columns={'Annotation_x': 'Annotation', 'Label_x': 'Label'}, inplace=True)


# Save the DataFrame to a CSV file
sub.to_csv('giveme2.0.csv', index=False)
