import numpy as np
import time
import matplotlib.pyplot as plt
from bpnn import BPNN
import os
import cv2

def load_dataset_bpnn(dataset_path='dataset', img_size=(64, 64)):
    """Load dataset for BPNN (flattened) - Identity Only"""
    
    images = []
    labels = []
    label_names = []
    
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in files:
        try:
            # Parse filename
            # Format: Name_Index.jpg or Name_Expression_Index.jpg
            # Logic: Take the first part as the name
            parts = filename.rsplit('.', 1)[0].split('_')
            
            if len(parts) > 0:
                name = parts[0]
                label = name
            else:
                continue

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
            continue
    
    X = np.array(images)
    label_names.sort()
    label_map = {i: name for i, name in enumerate(label_names)}
    reverse_map = {name: i for i, name in label_map.items()}
    y_int = np.array([reverse_map[label] for label in labels])
    
    num_classes = len(label_names)
    y = np.zeros((len(y_int), num_classes))
    y[np.arange(len(y_int)), y_int] = 1
    
    return X, y, label_map, y_int

def load_dataset_cnn(dataset_path='dataset', img_size=(64, 64)):
    """Load dataset for CNN (2D images) - Identity Only"""
    
    images = []
    labels = []
    label_names = []
    
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in files:
        try:
            # Parse filename
            # Format: Name_Index.jpg or Name_Expression_Index.jpg
            # Logic: Take the first part as the name
            parts = filename.rsplit('.', 1)[0].split('_')
            
            if len(parts) > 0:
                name = parts[0]
                label = name
            else:
                continue
            
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension for CNN
            
            images.append(img)
            labels.append(label)
            
            if label not in label_names:
                label_names.append(label)
        
        except Exception as e:
            continue
    
    X = np.array(images)
    label_names.sort()
    reverse_map = {name: i for i, name in enumerate(label_names)}
    y = np.array([reverse_map[label] for label in labels])
    
    return X, y, len(label_names)

def train_bpnn_model(X, y, epochs=500):
    """Train BPNN model"""
    print("\n[Training BPNN Model]")
    print("-" * 50)
    
    input_size = X.shape[1]
    hidden_size = 128
    output_size = y.shape[1]
    
    model = BPNN(input_size, hidden_size, output_size, learning_rate=0.1)
    
    start_time = time.time()
    # Reduced verbosity for comparison script
    model.train(X, y, epochs=epochs, batch_size=16, verbose=False)
    training_time = time.time() - start_time
    
    accuracy = model.evaluate(X, y)
    
    # Model size calculation
    model_size = (model.weights_ih.nbytes + model.weights_ho.nbytes + 
                  model.bias_h.nbytes + model.bias_o.nbytes) / (1024 * 1024)
    
    # Inference speed test
    inference_times = []
    test_batch = X[:10] if len(X) >= 10 else X
    for _ in range(100):
        start = time.time()
        _ = model.predict(test_batch)
        inference_times.append(time.time() - start)
    avg_inference_time = np.mean(inference_times) * 1000
    
    # Parameters count
    params = (input_size * hidden_size + hidden_size + 
              hidden_size * output_size + output_size)
    
    return {
        'model': model,
        'training_time': training_time,
        'accuracy': accuracy,
        'inference_time': avg_inference_time,
        'model_size': model_size,
        'parameters': params
    }

def create_cnn_model(input_shape, num_classes):
    """Create CNN model using Keras"""
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn_model(X, y, num_classes, epochs=50):
    """Train CNN model"""
    print("\n[Training CNN Model]")
    print("-" * 50)
    
    try:
        import tensorflow as tf
        # Suppress TensorFlow warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except:
        pass
    
    model = create_cnn_model(X.shape[1:], num_classes)
    
    print(f"CNN Architecture created for {num_classes} people/classes.")
    # model.summary() # Optional: Commented out to reduce clutter
    
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=16,
        verbose=0,
        validation_split=0.0
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    _, accuracy = model.evaluate(X, y, verbose=0)
    accuracy *= 100
    
    # Model size (approximate)
    model_size = sum([np.prod(w.shape) for w in model.get_weights()]) * 4 / (1024 * 1024)
    
    # Inference speed test
    inference_times = []
    test_batch = X[:10] if len(X) >= 10 else X
    for _ in range(100):
        start = time.time()
        _ = model.predict(test_batch, verbose=0)
        inference_times.append(time.time() - start)
    avg_inference_time = np.mean(inference_times) * 1000
    
    # Parameters count
    params = model.count_params()
    
    return {
        'model': model,
        'training_time': training_time,
        'accuracy': accuracy,
        'inference_time': avg_inference_time,
        'model_size': model_size,
        'parameters': params,
        'history': history
    }

def plot_comparison(bpnn_results, cnn_results):
    """Plot comprehensive comparison"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Deep Comparison: BPNN vs CNN for Face Recognition (Identity Only)', 
                 fontsize=18, fontweight='bold')
    
    models = ['BPNN\n(Custom)', 'CNN\n(Deep Learning)']
    colors = ['#3498db', '#e74c3c']
    
    # 1. Training Time
    ax1 = fig.add_subplot(gs[0, 0])
    times = [bpnn_results['training_time'], cnn_results['training_time']]
    bars1 = ax1.bar(models, times, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax1.set_ylabel('Time (seconds)', fontweight='bold', fontsize=10)
    ax1.set_title('Training Time', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    accuracies = [bpnn_results['accuracy'], cnn_results['accuracy']]
    bars2 = ax2.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax2.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=10)
    ax2.set_title('Model Accuracy', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=95, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Inference Time
    ax3 = fig.add_subplot(gs[0, 2])
    inf_times = [bpnn_results['inference_time'], cnn_results['inference_time']]
    bars3 = ax3.bar(models, inf_times, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax3.set_ylabel('Time (ms)', fontweight='bold', fontsize=10)
    ax3.set_title('Inference Time (10 samples)', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Model Size
    ax4 = fig.add_subplot(gs[1, 0])
    sizes = [bpnn_results['model_size'], cnn_results['model_size']]
    bars4 = ax4.bar(models, sizes, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax4.set_ylabel('Size (MB)', fontweight='bold', fontsize=10)
    ax4.set_title('Model Size', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 5. Parameters Count
    ax5 = fig.add_subplot(gs[1, 1])
    params = [bpnn_results['parameters'], cnn_results['parameters']]
    bars5 = ax5.bar(models, params, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax5.set_ylabel('Count', fontweight='bold', fontsize=10)
    ax5.set_title('Number of Parameters', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 6. Speed Ratio (BPNN as baseline)
    ax6 = fig.add_subplot(gs[1, 2])
    train_ratio = cnn_results['training_time'] / bpnn_results['training_time'] if bpnn_results['training_time'] > 0 else 0
    infer_ratio = cnn_results['inference_time'] / bpnn_results['inference_time'] if bpnn_results['inference_time'] > 0 else 0
    ratios = [train_ratio, infer_ratio]
    labels = ['Training', 'Inference']
    bars6 = ax6.bar(labels, ratios, color=['#9b59b6', '#f39c12'], alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Ratio (CNN/BPNN)', fontweight='bold', fontsize=10)
    ax6.set_title('Speed Ratio (>1 = BPNN faster)', fontweight='bold', fontsize=12)
    ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax6.grid(axis='y', alpha=0.3)
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 7. Comprehensive Comparison Table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = [
        ['Metric', 'BPNN', 'CNN', 'Winner'],
        ['Training Time', f'{bpnn_results["training_time"]:.2f}s', 
         f'{cnn_results["training_time"]:.2f}s',
         'BPNN' if bpnn_results["training_time"] < cnn_results["training_time"] else 'CNN'],
        ['Accuracy', f'{bpnn_results["accuracy"]:.2f}%', 
         f'{cnn_results["accuracy"]:.2f}%',
         'BPNN' if bpnn_results["accuracy"] > cnn_results["accuracy"] else 'CNN'],
        ['Inference Time', f'{bpnn_results["inference_time"]:.2f}ms', 
         f'{cnn_results["inference_time"]:.2f}ms',
         'BPNN' if bpnn_results["inference_time"] < cnn_results["inference_time"] else 'CNN'],
        ['Model Size', f'{bpnn_results["model_size"]:.2f}MB', 
         f'{cnn_results["model_size"]:.2f}MB',
         'BPNN' if bpnn_results["model_size"] < cnn_results["model_size"] else 'CNN'],
        ['Parameters', f'{bpnn_results["parameters"]:,}', 
         f'{cnn_results["parameters"]:,}',
         'BPNN' if bpnn_results["parameters"] < cnn_results["parameters"] else 'CNN'],
    ]
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color winner column
    for i in range(1, len(table_data)):
        if table_data[i][3] == 'BPNN':
            table[(i, 3)].set_facecolor('#d5f4e6')
        else:
            table[(i, 3)].set_facecolor('#fadbd8')
    
    plt.savefig('bpnn_vs_cnn_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Comparison chart saved to: bpnn_vs_cnn_comparison.png")
    plt.show()

def print_detailed_analysis(bpnn_results, cnn_results):
    """Print detailed analysis"""
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON ANALYSIS: BPNN vs CNN")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'BPNN':<25} {'CNN':<25} {'Ratio':<10}")
    print("-" * 80)
    
    t_ratio = cnn_results['training_time']/bpnn_results['training_time'] if bpnn_results['training_time'] > 0 else 0
    print(f"{'Training Time':<25} {bpnn_results['training_time']:>23.2f}s "
          f"{cnn_results['training_time']:>23.2f}s "
          f"{t_ratio:>8.2f}x")
    
    a_ratio = cnn_results['accuracy']/bpnn_results['accuracy'] if bpnn_results['accuracy'] > 0 else 0
    print(f"{'Accuracy':<25} {bpnn_results['accuracy']:>22.2f}% "
          f"{cnn_results['accuracy']:>22.2f}% "
          f"{a_ratio:>8.2f}x")
    
    i_ratio = cnn_results['inference_time']/bpnn_results['inference_time'] if bpnn_results['inference_time'] > 0 else 0
    print(f"{'Inference Time':<25} {bpnn_results['inference_time']:>22.2f}ms "
          f"{cnn_results['inference_time']:>22.2f}ms "
          f"{i_ratio:>8.2f}x")
    
    s_ratio = cnn_results['model_size']/bpnn_results['model_size'] if bpnn_results['model_size'] > 0 else 0
    print(f"{'Model Size':<25} {bpnn_results['model_size']:>22.2f}MB "
          f"{cnn_results['model_size']:>22.2f}MB "
          f"{s_ratio:>8.2f}x")
    
    p_ratio = cnn_results['parameters']/bpnn_results['parameters'] if bpnn_results['parameters'] > 0 else 0
    print(f"{'Parameters':<25} {bpnn_results['parameters']:>23,} "
          f"{cnn_results['parameters']:>23,} "
          f"{p_ratio:>8.2f}x")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("\n1. ACCURACY:")
    acc_diff = abs(bpnn_results['accuracy'] - cnn_results['accuracy'])
    if acc_diff < 2:
        print(f"   Both models achieve SIMILAR accuracy (difference: {acc_diff:.2f}%)")
    elif bpnn_results['accuracy'] > cnn_results['accuracy']:
        print(f"   BPNN is MORE ACCURATE by {acc_diff:.2f}%")
    else:
        print(f"   CNN is MORE ACCURATE by {acc_diff:.2f}%")
    
    print("\n2. TRAINING SPEED:")
    train_ratio = cnn_results['training_time'] / bpnn_results['training_time'] if bpnn_results['training_time'] > 0 else 0
    if train_ratio > 1:
        print(f"   BPNN trains {train_ratio:.1f}x FASTER than CNN")
        print(f"   CNN takes {train_ratio:.1f} times longer to train")
    else:
        print(f"   CNN trains {1/train_ratio:.1f}x FASTER than BPNN")
    
    print("\n3. INFERENCE SPEED:")
    infer_ratio = cnn_results['inference_time'] / bpnn_results['inference_time'] if bpnn_results['inference_time'] > 0 else 0
    if infer_ratio > 1:
        print(f"   BPNN is {infer_ratio:.1f}x FASTER at inference")
        print(f"   Better for REAL-TIME applications")
    else:
        print(f"   CNN is {1/infer_ratio:.1f}x FASTER at inference")
    
    print("\n4. MODEL COMPLEXITY:")
    param_ratio = cnn_results['parameters'] / bpnn_results['parameters'] if bpnn_results['parameters'] > 0 else 0
    print(f"   CNN has {param_ratio:.1f}x MORE parameters than BPNN")
    print(f"   BPNN: {bpnn_results['parameters']:,} parameters")
    print(f"   CNN: {cnn_results['parameters']:,} parameters")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    bpnn_score = 0
    cnn_score = 0
    
    if bpnn_results['training_time'] < cnn_results['training_time']:
        bpnn_score += 1
    else:
        cnn_score += 1
    
    if bpnn_results['accuracy'] > cnn_results['accuracy']:
        bpnn_score += 2
    else:
        cnn_score += 2
    
    if bpnn_results['inference_time'] < cnn_results['inference_time']:
        bpnn_score += 1
    else:
        cnn_score += 1
    
    if bpnn_results['model_size'] < cnn_results['model_size']:
        bpnn_score += 1
    else:
        cnn_score += 1
    
    print(f"\nOverall Score: BPNN = {bpnn_score}/5, CNN = {cnn_score}/5")
    
    print("\nUse BPNN when:")
    print("  - Fast training is required")
    print("  - Limited computational resources")
    print("  - Real-time inference is critical")
    print("  - Small to medium dataset (100-1000 samples)")
    print("  - Model interpretability is important")
    
    print("\nUse CNN when:")
    print("  - Maximum accuracy is critical")
    print("  - Large dataset available (1000+ samples)")
    print("  - Computational resources are abundant")
    print("  - Spatial features are important")
    print("  - Transfer learning is needed")

def main():
    """Main comparison function"""
    
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: BPNN vs CNN (Identity Only)")
    print("=" * 80)
    
    # Check dataset
    if not os.path.exists('dataset'):
        print("\n[X] ERROR: 'dataset' folder not found!")
        return
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"\n[OK] TensorFlow version: {tf.__version__}")
    except ImportError:
        print("\n[X] ERROR: TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        return
    
    # Load data for both models
    print("\n[Loading Dataset for BPNN...]")
    X_bpnn, y_bpnn, label_map, _ = load_dataset_bpnn()
    print(f"[OK] BPNN dataset loaded: {len(X_bpnn)} samples, {y_bpnn.shape[1]} classes")
    
    print("\n[Loading Dataset for CNN...]")
    X_cnn, y_cnn, num_classes = load_dataset_cnn()
    print(f"[OK] CNN dataset loaded: {len(X_cnn)} samples, {num_classes} classes")
    
    # Train BPNN
    print("\n" + "=" * 80)
    print("TRAINING PHASE")
    print("=" * 80)
    
    bpnn_results = train_bpnn_model(X_bpnn, y_bpnn, epochs=500)
    print(f"[OK] BPNN trained - Accuracy: {bpnn_results['accuracy']:.2f}%, "
          f"Time: {bpnn_results['training_time']:.2f}s")
    
    # Train CNN (fewer epochs for speed)
    cnn_results = train_cnn_model(X_cnn, y_cnn, num_classes, epochs=50)
    print(f"[OK] CNN trained - Accuracy: {cnn_results['accuracy']:.2f}%, "
          f"Time: {cnn_results['training_time']:.2f}s")
    
    # Print analysis
    print_detailed_analysis(bpnn_results, cnn_results)
    
    # Plot comparison
    print("\n[Generating Comparison Charts...]")
    plot_comparison(bpnn_results, cnn_results)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()