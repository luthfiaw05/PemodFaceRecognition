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
            # Parse filename
            # Format baru: Name_Index.jpg
            # Format lama: Name_Expression_Index.jpg
            # Logic: Ambil bagian pertama sebagai Nama
            parts = filename.rsplit('.', 1)[0].split('_')
            
            if len(parts) > 0:
                name = parts[0]
                label = name
            else:
                label = "Unknown"
            
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Preprocess image
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
    
    # Sort label names to ensure consistency with training
    label_names.sort()
    
    # Create mapping
    label_map = {i: name for i, name in enumerate(label_names)}
    reverse_map = {name: i for i, name in label_map.items()}
    
    # Convert string labels to integers
    y_int = np.array([reverse_map[label] for label in labels])
    
    return X, y_int, label_map

def plot_confusion_matrix(y_true, y_pred, label_map, save_path='confusion_matrix.png'):
    """
    Generate and display confusion matrix heatmap
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get class names
    class_names = [label_map[i] for i in range(len(label_map))]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Face Recognition (Identity Only)', fontsize=16, fontweight='bold')
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
    # We use a try-except block just in case some classes are missing in the test set
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)
    except Exception as e:
        print(f"Could not generate full report: {e}")
        print("Simple Accuracy per class:")
        for i, name in label_map.items():
            mask = y_true == i
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == y_true[mask]) * 100
                print(f"  {name}: {acc:.2f}%")
    
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
    plt.figure(figsize=(12, 6))
    colors = ['green' if acc > 90 else 'orange' if acc > 70 else 'red' for acc in accuracies]
    bars = plt.bar(range(len(class_names)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Person (Class)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Person Recognition Accuracy', fontsize=16, fontweight='bold')
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
    print("MODEL EVALUATION (IDENTITY ONLY)")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists('face_model.pkl'):
        print("\n[X] ERROR: Model not found!")
        print("Please train the model first: python 1_train.py")
        return
    
    # Load model
    print("\n[Loading Model...]")
    # Initialize dummy model
    model = BPNN(0, 0, 0)
    model.load_model('face_model.pkl')
    
    # Load saved label map from training to ensure consistency
    if os.path.exists('label_map.pkl'):
        with open('label_map.pkl', 'rb') as f:
            train_label_map = pickle.load(f)
    else:
        print("Warning: label_map.pkl not found. Evaluation might be mismatched if classes differ.")
        train_label_map = None
    
    print("[OK] Model loaded successfully!")
    
    # Load test data
    print("\n[Loading Dataset...]")
    X, y_true, test_label_map = load_test_data()
    
    if X is None:
        print("[X] Failed to load dataset")
        return
    
    # Consistency Check
    # Ensure the test data mapping matches the training mapping
    if train_label_map:
        # Check if classes match
        train_classes = sorted(list(train_label_map.values()))
        test_classes = sorted(list(test_label_map.values()))
        
        if train_classes != test_classes:
            print("\n[WARNING] Classes in dataset differ from training classes!")
            print(f"Training classes: {train_classes}")
            print(f"Dataset classes:  {test_classes}")
            
            # Remap y_true to match training indices
            # Create a map from Name -> Training Index
            name_to_train_idx = {name: idx for idx, name in train_label_map.items()}
            
            new_y_true = []
            valid_indices = []
            
            for i, local_idx in enumerate(y_true):
                name = test_label_map[local_idx]
                if name in name_to_train_idx:
                    new_y_true.append(name_to_train_idx[name])
                    valid_indices.append(i)
            
            X = X[valid_indices]
            y_true = np.array(new_y_true)
            label_map = train_label_map
            print(f"[INFO] Filtered dataset to match training classes. {len(X)} samples remaining.")
        else:
            label_map = test_label_map
    else:
        label_map = test_label_map

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
                    most_common_idx = np.argmax(counts)
                    most_common = unique[most_common_idx]
                    count = counts[most_common_idx]
                    
                    true_name = label_map.get(true_label, f"Unknown({true_label})")
                    pred_name = label_map.get(most_common, f"Unknown({most_common})")
                    
                    print(f"  '{true_name}' misclassified as '{pred_name}': {count} times")
    
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