import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def compute_metrics(true_labels, predictions):
    """
    Computes sensitivity, specificity, and accuracy based on true labels and predictions.

    Args:
        true_labels (np.ndarray): Ground truth binary labels.
        predictions (np.ndarray): Binary predictions.

    Returns:
        dict: Dictionary containing sensitivity, specificity, and accuracy.
    """
    # Flatten arrays for metric computation
    true_labels = true_labels.flatten()
    predictions = predictions.flatten()

    # Check for the special case where both arrays are all zeros
    if np.all(true_labels == 0) and np.all(predictions == 0):
        return {
            'sensitivity': 0,
            'specificity': 1,
            'accuracy': 1
        }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()

    # Calculate metrics
    sensitivity = recall_score(true_labels, predictions, zero_division=0)  # Same as TP / (TP + FN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(true_labels, predictions)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

def find_matching_test_files(predicted_bounds, test_frames_dir):
    """
    Finds all test files whose extents overlap with the predicted file bounds.

    Args:
        predicted_bounds (tuple): Bounding box of the predicted file (minx, miny, maxx, maxy).
        test_frames_dir (str): Directory containing test frames.

    Returns:
        list: List of paths to matching test files.
    """
    matching_files = []
    for test_file in os.listdir(test_frames_dir):
        if not test_file.endswith(".tif"):
            continue

        test_path = os.path.join(test_frames_dir, test_file)
        with rasterio.open(test_path) as test_ds:
            test_bounds = test_ds.bounds
            if box(*predicted_bounds).intersects(box(*test_bounds)):
                matching_files.append(test_path)

    return matching_files

# Paths
test_frames_dir = r"N:\isipd\projects\p_planetdw\data\dw_detection\aerial\training_data\20240816-0921_MACS_reprocessing_1024p\real"
predicted_images_dir = r"N:\isipd\projects\p_planetdw\data\dw_detection\aerial\results\macs_final_1\rasters"
output_metrics_path = r"N:\isipd\projects\p_planetdw\data\dw_detection\aerial\results\evaluation_metrics.txt"

# Create output directory if needed
os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)

# Variables to accumulate metrics
total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

# Iterate through predicted files
for predicted_file in os.listdir(predicted_images_dir):
    if not predicted_file.endswith(".tif"):
        continue

    predicted_path = os.path.join(predicted_images_dir, predicted_file)

    with rasterio.open(predicted_path) as pred_ds:
        predicted_bounds = pred_ds.bounds

        # Find all matching test files
        matching_test_files = find_matching_test_files(predicted_bounds, test_frames_dir)

        for test_path in matching_test_files:
            with rasterio.open(test_path) as test_ds:
                test_data = test_ds.read(5)  # Band 5 contains the labels
                test_bounds = test_ds.bounds

                # Calculate the intersection window
                intersection_window = from_bounds(
                    max(test_bounds[0], predicted_bounds[0]),
                    max(test_bounds[1], predicted_bounds[1]),
                    min(test_bounds[2], predicted_bounds[2]),
                    min(test_bounds[3], predicted_bounds[3]),
                    pred_ds.transform
                )

                # Read the predicted data in the intersection area
                pred_data = pred_ds.read(1, window=intersection_window)

                # Align test data to the same window
                test_window = from_bounds(
                    max(test_bounds[0], predicted_bounds[0]),
                    max(test_bounds[1], predicted_bounds[1]),
                    min(test_bounds[2], predicted_bounds[2]),
                    min(test_bounds[3], predicted_bounds[3]),
                    test_ds.transform
                )
                test_data_clipped = test_ds.read(5, window=test_window)

                # Binarize predictions and test labels (assuming threshold of 0.5)
                pred_binary = (pred_data > 0.5).astype(int)
                test_binary = (test_data_clipped > 0.5).astype(int)

                # Compute confusion matrix components
                metrics = compute_metrics(test_binary, pred_binary)

                # Update totals only if metrics are valid
                if metrics['specificity'] > 0 or metrics['sensitivity'] > 0 or metrics['accuracy'] > 0:
                    tn, fp, fn, tp = confusion_matrix(test_binary.flatten(), pred_binary.flatten(), labels=[0, 1]).ravel()
                    total_tp += tp
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn

# Compute overall metrics
sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0

# Save metrics to file
with open(output_metrics_path, "w") as f:
    f.write("Overall Metrics\n")
    f.write(f"Sensitivity: {sensitivity*100:.4f}%\n")
    f.write(f"Specificity: {specificity*100:.4f}%\n")
    f.write(f"Accuracy: {accuracy*100:.4f}%\n")

print(f"Overall metrics saved to {output_metrics_path}")
