import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from bpnn import BPNN
import os

def load_test_data():
    """Load dataset for evaluation"""
    import cv2
    
    dataset_path = 'dataset'
    img_size = (64, 64)
    
    if not os.path.exists(dataset_path):
        print("[X] ERROR: 'dataset' folder not found!")
        return None, None, None
    
    images = []
    labels = []
    label_names = []
    
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Loading {len(files)} images for evaluation...")
    
    for filename in files:
        try:
            parts = filename.rsplit('.', 1)[0].split('_')
            if len(parts) >= 3:
                name = parts[0]
                expression = parts[1]
                label = f"{name}_{expression}"
            else:
                label = parts[0] if parts else filename
            
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            img_vector = img.flatten()
            
            images.append(img_vector)
            labels.append(label)
            
            if label not in label_names:
                label_names.append(label)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if len(images) == 0:
        return None, None, None
    
    X = np.array(images)
    label_names.sort()
    label_map = {i: name for i, name in enumerate(label_names)}
    reverse_map = {name: i for i, name in label_map.items()}
    y_int = np.array([reverse_map[label] for label in labels])
    
    return X, y_int, label_map

def plot_confusion_matrix(y_true, y_pred, label_map, save_path='confusion_matrix.png'):
    """
    Generate and display confusion matrix heatmap
    
    Args:
        y_true: True labels (integers)
        y_pred: Predicted labels (integers)
        label_map: Dictionary mapping integers to class names
        save_path: Path to save the figure
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get class names
    class_names = [label_map[i] for i in range(len(label_map))]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Face Recognition BPNN', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Confusion matrix saved to: {save_path}")
    
    # Display
    plt.show()
    
    return cm

def calculate_metrics(y_true, y_pred, label_map):
    """
    Calculate detailed metrics per class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_map: Dictionary mapping integers to class names
    """
    from sklearn.metrics import classification_report, accuracy_score
    
    print("\n" + "=" * 70)
    print("DETAILED EVALUATION METRICS")
    print("=" * 70)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Per-class metrics
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS:")
    print("-" * 70)
    
    class_names = [label_map[i] for i in sorted(label_map.keys())]
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    return accuracy

def plot_accuracy_per_class(y_true, y_pred, label_map, save_path='accuracy_per_class.png'):
    """
    Plot accuracy for each class
    """
    class_names = [label_map[i] for i in sorted(label_map.keys())]
    accuracies = []
    
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            class_accuracy = (y_pred[mask] == i).sum() / mask.sum() * 100
            accuracies.append(class_accuracy)
        else:
            accuracies.append(0)
    
    # Create bar plot
    plt.figure(figsize=(14, 6))
    colors = ['green' if acc > 90 else 'orange' if acc > 70 else 'red' for acc in accuracies]
    bars = plt.bar(range(len(class_names)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Per-class accuracy chart saved to: {save_path}")
    plt.show()

def evaluate_model():
    """Main evaluation function"""
    
    print("=" * 70)
    print("MODEL EVALUATION WITH CONFUSION MATRIX")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists('face_model.pkl'):
        print("\n[X] ERROR: Model not found!")
        print("Please train the model first: python 1_train.py")
        return
    
    # Load model
    print("\n[Loading Model...]")
    model = BPNN(0, 0, 0)
    model.load_model('face_model.pkl')
    
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    print("[OK] Model loaded successfully!")
    
    # Load test data
    print("\n[Loading Dataset...]")
    X, y_true, _ = load_test_data()
    
    if X is None:
        print("[X] Failed to load dataset")
        return
    
    print(f"[OK] Loaded {len(X)} samples")
    print(f"[OK] Number of classes: {len(label_map)}")
    
    # Make predictions
    print("\n[Making Predictions...]")
    y_pred = model.predict(X)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_pred == y_true) * 100
    print(f"\n[RESULT] Overall Accuracy: {accuracy:.2f}%")
    
    # Calculate and display confusion matrix
    print("\n[Generating Confusion Matrix...]")
    cm = plot_confusion_matrix(y_true, y_pred, label_map)
    
    # Calculate detailed metrics
    calculate_metrics(y_true, y_pred, label_map)
    
    # Plot per-class accuracy
    print("\n[Generating Per-Class Accuracy Chart...]")
    plot_accuracy_per_class(y_true, y_pred, label_map)
    
    # Error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    errors = y_pred != y_true
    num_errors = errors.sum()
    
    print(f"\nTotal Errors: {num_errors} out of {len(y_true)} ({num_errors/len(y_true)*100:.2f}%)")
    
    if num_errors > 0:
        print("\nMost Common Misclassifications:")
        for true_label in range(len(label_map)):
            mask = (y_true == true_label) & errors
            if mask.sum() > 0:
                pred_labels = y_pred[mask]
                unique, counts = np.unique(pred_labels, return_counts=True)
                
                if len(unique) > 0:
                    most_common = unique[np.argmax(counts)]
                    count = counts[np.argmax(counts)]
                    
                    print(f"  {label_map[true_label]} misclassified as {label_map[most_common]}: {count} times")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - accuracy_per_class.png")

if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import sklearn
        import seaborn
        import matplotlib
    except ImportError as e:
        print("[X] ERROR: Missing required library!")
        print("\nPlease install:")
        print("  pip install scikit-learn seaborn matplotlib")
        print(f"\nMissing: {e}")
        exit(1)
    
    evaluate_model()