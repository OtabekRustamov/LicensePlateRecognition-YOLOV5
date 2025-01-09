import os
import torch
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Set environment for GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Define classes
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'License Plate']


def draw_boxes(image, boxes, labels, classes):
    """
    Draw bounding boxes with class labels displayed below the box.
    """
    for b, l in zip(boxes, labels):
        class_id = int(l)
        class_name = classes[class_id]
        x_min, y_min, x_max, y_max = list(map(int, b))

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        # Place label below the bounding box
        label = class_name
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y_min = y_max + ret[1] + baseline  # Adjust position below the bounding box
        cv2.rectangle(image, (x_min, y_max), (x_min + ret[0], label_y_min), (0, 255, 0), -1)
        cv2.putText(image, label, (x_min, y_max + ret[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def test_single_image(image_path, weights, img_size=640, conf_thres=0.25, iou_thres=0.45, output_dir="outputs"):
    """
    Test a single image using the YOLO model.
    Args:
        image_path (str): Path to the input image.
        weights (str): Path to the YOLO model weights.
        img_size (int): Image size for inference.
        conf_thres (float): Confidence threshold for predictions.
        iou_thres (float): IoU threshold for NMS.
        output_dir (str): Directory to save the output image.
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(weights, map_location=device)['ema'].float().eval()
    if device.type != 'cpu':
        model.half()

    # Load and preprocess image
    image = cv2.imread(image_path)  # BGR format
    assert image is not None, f"Image not found at {image_path}"
    original_image = image.copy()
    h0, w0 = image.shape[:2]

    # Resize and pad image
    image, ratio, pad = letterbox(image, img_size, scaleup=False)
    image = image[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and reshape
    image = np.ascontiguousarray(image)

    # Convert to tensor
    image_tensor = torch.from_numpy(image).to(device)
    image_tensor = image_tensor.half() if device.type != 'cpu' else image_tensor.float()
    image_tensor /= 255.0  # Normalize to [0, 1]
    if image_tensor.ndimension() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Perform inference
    pred = model(image_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]

    # Process predictions
    pred_boxes, pred_classes = [], []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(image_tensor.shape[2:], pred[:, :4], original_image.shape).round()
        for *xyxy, conf, cls in pred:
            pred_boxes.append([int(x) for x in xyxy])
            pred_classes.append(int(cls))

    # Debugging: Print results
    print(f"Detected classes: {[CLASSES[c] for c in pred_classes]}")

    # Draw and save results
    draw_boxes(original_image, pred_boxes, pred_classes, CLASSES)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, original_image)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    test_single_image(
        image_path="plate.PNG",  # Replace with the path to your image
        weights="best.pt",         # Replace with the path to your model weights
        img_size=640,                      # Image size for YOLO
        conf_thres=0.25,                   # Confidence threshold
        iou_thres=0.45,                    # IoU threshold
        output_dir="outputs"               # Output directory
    )
