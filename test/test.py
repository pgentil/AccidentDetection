from inference.models.utils import get_roboflow_model
import cv2
import time
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model_name = "amazon-accident-detection-o3juo"
model_version = "3"
api_key = "ktSFVMakkE69oahKbqtv"
temp_vid = "./temp_video.mp4"

model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key=api_key
)

def reduce_fps(video_path: str, target_fps: int = 5, cut_video: bool = False, video_len : int = 10):
    #capturamos el original
    cap = cv2.VideoCapture(video_path)
    #agarramos los fps del video original
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    #caluclamos el intervalo de frames para reducir a target_fps
    frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1
    #preparamos el video de salida codec mp4v
    out = cv2.VideoWriter(temp_vid, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (int(cap.get(3)), int(cap.get(4))))
    if (cut_video):
        max_frames = target_fps * video_len
    frames = []
    i = 0
    effective_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret or (cut_video and effective_frames >= max_frames): # Si no hay más frames o hemos alcanzado el límite
            break
        if i % frame_interval == 0:
            #escribimos el frame en el video de salida
            out.write(frame)
            frames.append(frame)
            effective_frames += 1
        i += 1
    cap.release()
    out.release()
    return frames


def process_video(video_path: str | None = None, frames: list = None):
    i = 0
    if frames is not None:
        for frame in frames:
            results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
            print(results[0].predictions if results[0].predictions else "No predictions")
            i = i + 1
    else:
        if video_path is None:
            raise ValueError("Either video_path or frames must be provided")
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.infer(image=frame, confidence=0.7, iou_threshold=0.5)
            # print(results[0].predictions if results[0].predictions else "No predictions")
            i = i + 1
        cap.release()
    return i


def check_image_labels(images_dir: str, labels_dir: str):
    """
    Iterate through images, perform inference, and check if corresponding labels exist and are non-empty.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Get all image files (common image extensions) - using set to avoid duplicates
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    image_files = set()  # Use set to avoid duplicates
    for ext in image_extensions:
        image_files.update(images_path.glob(ext))
    
    image_files = list(image_files)  # Convert back to list
    image_files.sort()  # Sort for consistent ordering
    
    # print(f"Found {len(image_files)} unique images in {images_dir}")
    
    # # Debug: show first few files found
    # print("First 10 files found:")
    # for i, img_file in enumerate(image_files[:10]):
    #     print(f"  {i+1}: {img_file.name}")
    # if len(image_files) > 10:
    #     print(f"  ... and {len(image_files) - 10} more files")
    # print()
    
    results = {
        'total_images': len(image_files),
        'images_with_predictions': 0,
        'images_with_labels': 0,
        'images_with_both': 0,
        'images_with_neither': 0,
        'true_positives': 0,  # Model predicts AND label exists
        'false_positives': 0,  # Model predicts BUT no label
        'false_negatives': 0,  # Model doesn't predict BUT label exists
        'true_negatives': 0   # Model doesn't predict AND no label
    }
    
    processed_count = 0
    for image_file in image_files:
        processed_count += 1
        print(f"Processing ({processed_count}/{len(image_files)}): {image_file.name}")
        
        # Load and infer on image
        image = cv2.imread(str(image_file))
        if image is None:
            # print(f"Could not load image: {image_file}")
            continue
            
        inference_results = model.infer(image=image, confidence=0.70, iou_threshold=0.5)
        has_predictions = len(inference_results[0].predictions) > 0
        
        if has_predictions:
            results['images_with_predictions'] += 1
            # print(f"  Model predictions: {len(inference_results[0].predictions)}")
            # for pred in inference_results[0].predictions:
            #     print(f"    - {pred.class_name} (confidence: {pred.confidence:.2f})")
        # else:
        #     print("  No model predictions")
        
        # Check corresponding label file
        # Assuming label files have same name but .txt extension
        label_file = labels_path / (image_file.stem + '.txt')
        has_label = False
        
        if label_file.exists():
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    has_label = len(content) > 0
                    if has_label:
                        results['images_with_labels'] += 1
                        # print(f"  Label file exists and has content: {label_file.name}")
                        # print(f"    Content: {content}")
                    # else:
                    #     print(f"  Label file exists but is empty: {label_file.name}")
            except Exception as e:
                # print(f"  Error reading label file {label_file}: {e}")
                pass
        # else:
        #     print(f"  No corresponding label file found: {label_file.name}")
        
        # Update statistics
        if has_predictions and has_label:
            results['images_with_both'] += 1
            results['true_positives'] += 1
        elif has_predictions and not has_label:
            results['false_positives'] += 1
        elif not has_predictions and has_label:
            results['false_negatives'] += 1
        elif not has_predictions and not has_label:
            results['images_with_neither'] += 1
            results['true_negatives'] += 1
        
        # print("-" * 50)
    
    return results

def save_confusion_matrix(results, save_path="./confusion_matrix.png"):
    """
    Generate and save a confusion matrix visualization.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create confusion matrix data
        confusion_matrix = np.array([
            [results['true_positives'], results['false_negatives']],  # Actual Positive
            [results['false_positives'], results['true_negatives']]    # Actual Negative
        ])
        
        # Set up the plot
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        ax = sns.heatmap(confusion_matrix, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=['Predicted Positive', 'Predicted Negative'],
                        yticklabels=['Actual Positive', 'Actual Negative'],
                        cbar_kws={'label': 'Count'})
        
        # Add labels and title
        plt.title('Confusion Matrix - Accident Detection Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('Actual Labels', fontsize=12)
        
        # Add percentage annotations
        total = confusion_matrix.sum()
        for i in range(2):
            for j in range(2):
                percentage = (confusion_matrix[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        # Add metrics text box
        tp, fp, fn, tn = results['true_positives'], results['false_positives'], results['false_negatives'], results['true_negatives']
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}'
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Confusion matrix saved to: {save_path}")
        return True
        
    except ImportError as e:
        print(f"Warning: Could not create confusion matrix plot. Missing dependencies: {e}")
        print("To install required packages, run: pip install matplotlib seaborn")
        return False
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        return False


def save_basic_confusion_matrix(results, save_path="./confusion_matrix_basic.txt"):
    """
    Save a basic text-based confusion matrix if matplotlib is not available.
    """
    tp, fp, fn, tn = results['true_positives'], results['false_positives'], results['false_negatives'], results['true_negatives']
    
    confusion_text = f"""
CONFUSION MATRIX
================

                 PREDICTED
                Pos    Neg
ACTUAL   Pos   {tp:4d}   {fn:4d}
         Neg   {fp:4d}   {tn:4d}

METRICS:
--------
Total Images: {results['total_images']}
Accuracy:  {(tp + tn) / results['total_images']:.3f}
Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.3f}
Recall:    {tp / (tp + fn) if (tp + fn) > 0 else 0:.3f}

True Positives:  {tp} (Model detected accident, label exists)
False Positives: {fp} (Model detected accident, no label)
False Negatives: {fn} (Model missed accident, label exists)
True Negatives:  {tn} (Model no detection, no label)
"""
    
    with open(save_path, 'w') as f:
        f.write(confusion_text)
    
    print(f"Basic confusion matrix saved to: {save_path}")
    return True


def print_analysis_results(results):
    """Print analysis results in a formatted way."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Total images processed: {results['total_images']}")
    print(f"Images with model predictions: {results['images_with_predictions']}")
    print(f"Images with labels: {results['images_with_labels']}")
    print(f"Images with both predictions and labels: {results['images_with_both']}")
    print(f"Images with neither predictions nor labels: {results['images_with_neither']}")
    
    print("\nCONFUSION MATRIX:")
    print(f"True Positives (Model predicts + Label exists): {results['true_positives']}")
    print(f"False Positives (Model predicts + No label): {results['false_positives']}")
    print(f"False Negatives (No prediction + Label exists): {results['false_negatives']}")
    print(f"True Negatives (No prediction + No label): {results['true_negatives']}")
    
    # Calculate metrics if possible
    if results['true_positives'] + results['false_positives'] > 0:
        precision = results['true_positives'] / (results['true_positives'] + results['false_positives'])
        print(f"\nPrecision: {precision:.2f}")
    
    if results['true_positives'] + results['false_negatives'] > 0:
        recall = results['true_positives'] / (results['true_positives'] + results['false_negatives'])
        print(f"Recall: {recall:.2f}")
    
    if results['total_images'] > 0:
        accuracy = (results['true_positives'] + results['true_negatives']) / results['total_images']
        print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    # video_path = "zVzXEht1aME.mp4"  # Replace with your test video path
    # frames = reduce_fps(video_path, target_fps=5, cut_video=True, video_len=10)
    # print(f"Reduced to {len(frames)} frames")
    # start = time.time()
    # i = process_video(temp_vid, frames=frames)
    # end = time.time()
    # print(f"Processing time: {end - start} seconds")
    # # os.remove(temp_vid)
    # print(i)
    
    # Define paths to your images and labels directories
    images_directory = "./test/images"  # Update this path
    labels_directory = "./test/labels"  # Update this path
    
    print(f"Current working directory: {Path.cwd()}")
    print(f"Images directory: {Path(images_directory).absolute()}")
    print(f"Labels directory: {Path(labels_directory).absolute()}")
    
    # Check if directories exist
    if not Path(images_directory).exists():
        print(f"Images directory does not exist: {images_directory}")
        exit(1)
    if not Path(labels_directory).exists():
        print(f"Labels directory does not exist: {labels_directory}")
        exit(1)
    
    # Run the analysis
    start_time = time.time()
    analysis_results = check_image_labels(images_directory, labels_directory)
    end_time = time.time()
    
    # Print results
    print_analysis_results(analysis_results)
    
    # Generate and save confusion matrix
    confusion_matrix_path = "./confusion_matrix.png"
    success = save_confusion_matrix(analysis_results, confusion_matrix_path)
    
    # If matplotlib failed, save a basic text version
    if not success:
        save_basic_confusion_matrix(analysis_results, "./confusion_matrix.txt")
    
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
