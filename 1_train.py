import os
import cv2
import numpy as np
import pickle
from bpnn import BPNN

def load_dataset(dataset_path='dataset', img_size=(64, 64)):
    """
    Load images from dataset folder
    Expected naming: Name_Expression_Index.jpg
    
    Returns:
        X: Image data (flattened vectors)
        y: Labels (encoded as integers)
        label_map: Dictionary mapping label indices to names
    """
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: '{dataset_path}' folder not found!")
        return None, None, None
    
    images = []
    labels = []
    label_names = []
    
    # Read all image files
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(files) == 0:
        print(f"ERROR: No images found in '{dataset_path}' folder!")
        return None, None, None
    
    print(f"Found {len(files)} images in dataset")
    
    for filename in files:
        # Parse filename: Name_Expression_Index.jpg
        try:
            parts = filename.rsplit('.', 1)[0].split('_')
            if len(parts) >= 3:
                name = parts[0]  # Extract person's name
                expression = parts[1]  # Extract expression
                # Combine name and expression as label
                label = f"{name}_{expression}"
            else:
                name = parts[0] if parts else filename
                label = name
            
            # Read and preprocess image
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read {filename}")
                continue
            
            # Resize image
            img = cv2.resize(img, img_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Flatten image to vector
            img_vector = img.flatten()
            
            images.append(img_vector)
            labels.append(label)
            
            if label not in label_names:
                label_names.append(label)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if len(images) == 0:
        print("ERROR: No valid images were loaded!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(images)
    
    # Create label mapping
    label_names.sort()  # Sort for consistency
    label_map = {i: name for i, name in enumerate(label_names)}
    reverse_map = {name: i for i, name in label_map.items()}
    
    # Convert labels to integers
    y_int = np.array([reverse_map[label] for label in labels])
    
    # One-hot encode labels
    num_classes = len(label_names)
    y = np.zeros((len(y_int), num_classes))
    y[np.arange(len(y_int)), y_int] = 1
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Number of classes (Name_Expression): {num_classes}")
    print(f"Classes in dataset: {label_names}")
    print(f"Image size: {img_size}")
    print(f"Feature vector size: {X.shape[1]}")
    
    return X, y, label_map

def train_model():
    """Train the BPNN model on the dataset"""
    
    print("=" * 60)
    print("           FACE RECOGNITION TRAINING - BPNN")
    print("=" * 60)
    
    # Load dataset
    IMG_SIZE = (64, 64)
    X, y, label_map = load_dataset(img_size=IMG_SIZE)
    
    if X is None:
        print("\nTraining aborted due to dataset loading error.")
        return
    
    # Network architecture
    input_size = X.shape[1]  # Flattened image size
    hidden_size = 128        # Hidden layer neurons
    output_size = y.shape[1] # Number of classes
    
    print(f"\n{'─' * 60}")
    print("Neural Network Architecture:")
    print(f"  Input Layer:  {input_size} neurons")
    print(f"  Hidden Layer: {hidden_size} neurons")
    print(f"  Output Layer: {output_size} neurons")
    print(f"{'─' * 60}")
    
    # Create and train model
    model = BPNN(input_size, hidden_size, output_size, learning_rate=0.1)
    
    print("\nStarting training...")
    print("(This may take a few minutes depending on dataset size)\n")
    
    # Train the model
    model.train(X, y, epochs=500, batch_size=16, verbose=True)
    
    # Evaluate final accuracy
    final_accuracy = model.evaluate(X, y)
    print(f"\n{'=' * 60}")
    print(f"Final Training Accuracy: {final_accuracy:.2f}%")
    print(f"{'=' * 60}")
    
    # Save model
    model.save_model('face_model.pkl')
    
    # Save label mapping
    with open('label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    print("Label mapping saved to 'label_map.pkl'")
    
    # Save image size info
    with open('img_size.pkl', 'wb') as f:
        pickle.dump(IMG_SIZE, f)
    print(f"Image size info saved to 'img_size.pkl'")
    
    print("\n✓ Training completed successfully!")
    print("You can now run '2_recognize.py' for face recognition")

if __name__ == "__main__":
    train_model()