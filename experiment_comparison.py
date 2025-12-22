import os
import cv2
import numpy as np
import pickle
import time
from datetime import datetime
import pandas as pd

# Import BPNN class (harus ada bpnn.py di folder yang sama)
try:
    from bpnn import BPNN
except ImportError:
    print("[ERROR] File 'bpnn.py' tidak ditemukan!")
    print("Pastikan bpnn.py ada di folder yang sama dengan script ini.")
    exit(1)

# ============================================================================
# MODIFIED BPNN CLASS WITH DIFFERENT ACTIVATIONS
# ============================================================================

class BPNN_Flexible(BPNN):
    """Extended BPNN dengan support untuk multiple activation functions"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation='sigmoid'):
        super().__init__(input_size, hidden_size, output_size, learning_rate)
        self.activation_type = activation
    
    def activate_hidden(self, x):
        """Apply activation function berdasarkan type"""
        if self.activation_type == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_type == 'relu':
            return self.relu(x)
        elif self.activation_type == 'tanh':
            return self.tanh(x)
        elif self.activation_type == 'leaky_relu':
            return self.leaky_relu(x)
        else:
            return self.sigmoid(x)
    
    def activate_hidden_derivative(self, x):
        """Derivative untuk backpropagation"""
        if self.activation_type == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_type == 'relu':
            return self.relu_derivative(x)
        elif self.activation_type == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation_type == 'leaky_relu':
            return self.leaky_relu_derivative(x)
        else:
            return self.sigmoid_derivative(x)
    
    # Additional activation functions
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def forward(self, X):
        """Modified forward dengan flexible activation"""
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights_ih) + self.bias_h
        self.hidden_output = self.activate_hidden(self.hidden_input)
        
        # Output layer (softmax tetap sama)
        self.output_input = np.dot(self.hidden_output, self.weights_ho) + self.bias_o
        self.output = self.softmax(self.output_input)
        
        return self.output
    
    def backward(self, X, y, output):
        """Modified backward dengan flexible activation derivative"""
        m = X.shape[0]
        
        # Output layer error
        output_error = output - y
        output_delta = output_error
        
        # Hidden layer error
        hidden_error = np.dot(output_delta, self.weights_ho.T)
        hidden_delta = hidden_error * self.activate_hidden_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_ho -= self.learning_rate * np.dot(self.hidden_output.T, output_delta) / m
        self.bias_o -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        
        self.weights_ih -= self.learning_rate * np.dot(X.T, hidden_delta) / m
        self.bias_h -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dataset_with_size(dataset_path='dataset', img_size=(64, 64)):
    """
    Load dataset dengan ukuran image yang bisa disesuaikan.
    Hanya mengambil IDENTITAS (Nama) sebagai label.
    """
    
    if not os.path.exists(dataset_path):
        return None, None, None
    
    images = []
    labels = []
    label_names = []
    
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(files) == 0:
        return None, None, None
    
    for filename in files:
        try:
            # Parse filename
            # Format: Name_Index.jpg atau Name_Expression_Index.jpg
            # Logic: Ambil bagian pertama sebelum underscore sebagai Nama
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
            
            # Resize ke ukuran yang ditentukan
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            img_vector = img.flatten()
            
            images.append(img_vector)
            labels.append(label)
            
            if label not in label_names:
                label_names.append(label)
        
        except Exception as e:
            continue
    
    if len(images) == 0:
        return None, None, None
    
    X = np.array(images)
    label_names.sort()
    label_map = {i: name for i, name in enumerate(label_names)}
    reverse_map = {name: i for i, name in label_map.items()}
    y_int = np.array([reverse_map[label] for label in labels])
    
    num_classes = len(label_names)
    y = np.zeros((len(y_int), num_classes))
    y[np.arange(len(y_int)), y_int] = 1
    
    return X, y, label_map


# ============================================================================
# EXPERIMENT 1: IMAGE SIZE COMPARISON
# ============================================================================

def experiment_image_sizes():
    """
    Eksperimen perbandingan berbagai ukuran input image
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: PERBANDINGAN UKURAN INPUT IMAGE")
    print("="*80)
    
    # Ukuran yang akan ditest
    sizes = [
        (32, 32),
        (48, 48),
        (64, 64),
        (80, 80),
        (96, 96)
    ]
    
    results = []
    
    for img_size in sizes:
        width, height = img_size
        size_name = f"{width}x{height}"
        
        print(f"\n{'='*80}")
        print(f"Testing Image Size: {size_name}")
        print(f"{'='*80}")
        
        # Load dataset dengan ukuran ini
        print(f"\n[1/4] Loading dataset dengan ukuran {size_name}...")
        X, y, label_map = load_dataset_with_size(img_size=img_size)
        
        if X is None:
            print(f"[ERROR] Gagal load dataset!")
            continue
        
        print(f"[OK] Dataset loaded: {len(X)} samples, {len(label_map)} classes (People)")
        
        # Setup model
        input_size = X.shape[1]
        hidden_size = 128
        output_size = y.shape[1]
        
        print(f"\n[2/4] Membuat model...")
        print(f"      Input: {input_size} neurons")
        print(f"      Hidden: {hidden_size} neurons")
        print(f"      Output: {output_size} neurons")
        
        model = BPNN_Flexible(input_size, hidden_size, output_size, 
                             learning_rate=0.1, activation='sigmoid')
        
        # Training
        print(f"\n[3/4] Training model...")
        start_time = time.time()
        
        model.train(X, y, epochs=300, batch_size=16, verbose=False)
        
        training_time = time.time() - start_time
        
        # Evaluation
        print(f"\n[4/4] Evaluating model...")
        accuracy = model.evaluate(X, y)
        
        # Calculate model size
        total_params = (input_size * hidden_size + 
                       hidden_size * output_size + 
                       hidden_size + output_size)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Test inference speed
        test_sample = X[0:1]
        start = time.time()
        for _ in range(100):
            model.predict(test_sample)
        inference_time = ((time.time() - start) / 100) * 1000  # ms
        
        result = {
            'Size': size_name,
            'Input Neurons': input_size,
            'Total Parameters': total_params,
            'Model Size (MB)': round(model_size_mb, 2),
            'Training Time (s)': round(training_time, 2),
            'Accuracy (%)': round(accuracy, 2),
            'Inference (ms)': round(inference_time, 2)
        }
        
        results.append(result)
        
        print(f"\n[RESULTS]")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Training Time: {training_time:.2f} seconds")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print(f"  Inference Time: {inference_time:.2f} ms")
    
    # Display results table
    print("\n" + "="*80)
    print("HASIL PERBANDINGAN UKURAN IMAGE")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*80)
    print("ANALISIS")
    print("="*80)
    
    if len(results) > 0:
        best_accuracy = max(results, key=lambda x: x['Accuracy (%)'])
        fastest_training = min(results, key=lambda x: x['Training Time (s)'])
        smallest_model = min(results, key=lambda x: x['Model Size (MB)'])
        
        print(f"\nBest Accuracy: {best_accuracy['Size']} ({best_accuracy['Accuracy (%)']}%)")
        print(f"Fastest Training: {fastest_training['Size']} ({fastest_training['Training Time (s)']}s)")
        print(f"Smallest Model: {smallest_model['Size']} ({smallest_model['Model Size (MB)']} MB)")
        
        print("\n[KESIMPULAN]")
        print(f"Untuk balance optimal: 64x64 atau 80x80")
        print(f"Untuk speed priority: 32x32 atau 48x48")
        print(f"Untuk accuracy priority: 80x80 atau 96x96")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_sizes_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n[OK] Results saved to: {filename}")
    
    return results


# ============================================================================
# EXPERIMENT 2: ACTIVATION FUNCTION COMPARISON
# ============================================================================

def experiment_activation_functions():
    """
    Eksperimen perbandingan berbagai fungsi aktivasi
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: PERBANDINGAN FUNGSI AKTIVASI")
    print("="*80)
    
    # Fixed image size untuk fair comparison
    img_size = (64, 64)
    
    # Activation functions to test
    activations = ['sigmoid', 'relu', 'tanh', 'leaky_relu']
    
    results = []
    
    # Load dataset once
    print(f"\n[INFO] Loading dataset dengan ukuran {img_size[0]}x{img_size[1]}...")
    X, y, label_map = load_dataset_with_size(img_size=img_size)
    
    if X is None:
        print("[ERROR] Gagal load dataset!")
        return []
    
    print(f"[OK] Dataset loaded: {len(X)} samples, {len(label_map)} classes")
    
    input_size = X.shape[1]
    hidden_size = 128
    output_size = y.shape[1]
    
    for activation in activations:
        
        print(f"\n{'='*80}")
        print(f"Testing Activation Function: {activation.upper()}")
        print(f"{'='*80}")
        
        # Setup model
        print(f"\n[1/3] Membuat model dengan aktivasi {activation}...")
        model = BPNN_Flexible(input_size, hidden_size, output_size, 
                             learning_rate=0.1, activation=activation)
        
        # Training
        print(f"\n[2/3] Training model...")
        start_time = time.time()
        
        try:
            model.train(X, y, epochs=300, batch_size=16, verbose=False)
            training_time = time.time() - start_time
            training_success = True
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            training_time = 0
            training_success = False
            continue
        
        # Evaluation
        print(f"\n[3/3] Evaluating model...")
        accuracy = model.evaluate(X, y)
        
        # Test inference speed
        test_sample = X[0:1]
        start = time.time()
        for _ in range(100):
            model.predict(test_sample)
        inference_time = ((time.time() - start) / 100) * 1000  # ms
        
        result = {
            'Activation': activation.capitalize(),
            'Training Time (s)': round(training_time, 2),
            'Accuracy (%)': round(accuracy, 2),
            'Inference (ms)': round(inference_time, 2),
            'Status': 'Success' if training_success else 'Failed'
        }
        
        results.append(result)
        
        print(f"\n[RESULTS]")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Training Time: {training_time:.2f} seconds")
        print(f"  Inference Time: {inference_time:.2f} ms")
    
    # Display results table
    print("\n" + "="*80)
    print("HASIL PERBANDINGAN FUNGSI AKTIVASI")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*80)
    print("ANALISIS")
    print("="*80)
    
    if len(results) > 0:
        best_accuracy = max(results, key=lambda x: x['Accuracy (%)'])
        fastest_training = min(results, key=lambda x: x['Training Time (s)'])
        fastest_inference = min(results, key=lambda x: x['Inference (ms)'])
        
        print(f"\nBest Accuracy: {best_accuracy['Activation']} ({best_accuracy['Accuracy (%)']}%)")
        print(f"Fastest Training: {fastest_training['Activation']} ({fastest_training['Training Time (s)']}s)")
        print(f"Fastest Inference: {fastest_inference['Activation']} ({fastest_inference['Inference (ms)']} ms)")
        
        print("\n[KESIMPULAN]")
        print(f"ReLU biasanya memberikan balance terbaik antara speed dan accuracy")
        print(f"Sigmoid lebih stabil tapi lebih lambat (vanishing gradient)")
        print(f"Tanh lebih baik dari Sigmoid (zero-centered)")
        print(f"Leaky ReLU mencegah dying ReLU problem")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_activations_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n[OK] Results saved to: {filename}")
    
    return results


# ============================================================================
# EXPERIMENT 3: COMBINED COMPARISON
# ============================================================================

def experiment_combined():
    """
    Eksperimen kombinasi: Best image size × Best activation
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: KOMBINASI OPTIMAL")
    print("="*80)
    
    print("\nMenguji kombinasi ukuran dan aktivasi terbaik...")
    
    # Test combinations
    combinations = [
        ((64, 64), 'sigmoid'),  # Current default
        ((64, 64), 'relu'),     # Recommended
        ((80, 80), 'relu'),     # High accuracy
        ((48, 48), 'relu'),     # Fast
    ]
    
    results = []
    
    for img_size, activation in combinations:
        
        config_name = f"{img_size[0]}x{img_size[1]}_{activation}"
        
        print(f"\n{'='*80}")
        print(f"Testing: {config_name}")
        print(f"{'='*80}")
        
        # Load dataset
        X, y, label_map = load_dataset_with_size(img_size=img_size)
        
        if X is None:
            continue
        
        input_size = X.shape[1]
        hidden_size = 128
        output_size = y.shape[1]
        
        # Train
        model = BPNN_Flexible(input_size, hidden_size, output_size, 
                             learning_rate=0.1, activation=activation)
        
        start_time = time.time()
        model.train(X, y, epochs=300, batch_size=16, verbose=False)
        training_time = time.time() - start_time
        
        accuracy = model.evaluate(X, y)
        
        result = {
            'Configuration': config_name,
            'Image Size': f"{img_size[0]}x{img_size[1]}",
            'Activation': activation.capitalize(),
            'Accuracy (%)': round(accuracy, 2),
            'Training Time (s)': round(training_time, 2)
        }
        
        results.append(result)
        
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Training: {training_time:.2f}s")
    
    # Display results
    print("\n" + "="*80)
    print("HASIL KOMBINASI OPTIMAL")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    if len(results) > 0:
        best = max(results, key=lambda x: x['Accuracy (%)'])
        print(f"\n[REKOMENDASI]")
        print(f"Konfigurasi terbaik: {best['Configuration']}")
        print(f"  Accuracy: {best['Accuracy (%)']}%")
        print(f"  Training: {best['Training Time (s)']}s")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_combined_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n[OK] Results saved to: {filename}")
    
    return results


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu untuk menjalankan eksperimen"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "EKSPERIMEN PERBANDINGAN" + " "*34 + "║")
    print("║" + " "*15 + "Face Recognition BPNN System" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("\n[ERROR] Folder 'dataset' tidak ditemukan!")
        print("Pastikan Anda sudah mengumpulkan dataset terlebih dahulu.")
        print("Run: python 0_collect.py")
        return
    
    files = [f for f in os.listdir('dataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) == 0:
        print("\n[ERROR] Dataset kosong!")
        print("Collect images dulu dengan: python 0_collect.py")
        return
    
    print(f"\n[OK] Dataset ditemukan: {len(files)} images")
    
    while True:
        print("\n" + "="*80)
        print("PILIH EKSPERIMEN")
        print("="*80)
        print("1. Perbandingan Ukuran Input Image (32x32, 48x48, 64x64, 80x80, 96x96)")
        print("2. Perbandingan Fungsi Aktivasi (Sigmoid, ReLU, Tanh, Leaky ReLU)")
        print("3. Kombinasi Optimal (Size × Activation)")
        print("4. Run All Experiments")
        print("0. Exit")
        print("-" * 80)
        
        choice = input("\nPilihan (0-4): ").strip()
        
        if choice == '1':
            experiment_image_sizes()
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            experiment_activation_functions()
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            experiment_combined()
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            print("\n[INFO] Running all experiments...")
            print("Ini akan memakan waktu 10-30 menit tergantung dataset size.")
            confirm = input("Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                experiment_image_sizes()
                print("\n" + "="*80)
                experiment_activation_functions()
                print("\n" + "="*80)
                experiment_combined()
                print("\n[OK] All experiments completed!")
            
            input("\nPress Enter to continue...")
        
        elif choice == '0':
            print("\nGoodbye!")
            break
        
        else:
            print("\n[ERROR] Pilihan tidak valid!")


if __name__ == "__main__":
    main()