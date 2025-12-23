import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bpnn import BPNN

# ============================================================================
# EXTENDED CLASS UNTUK TRACKING HISTORY
# ============================================================================
class TrackingBPNN(BPNN):
    """
    Turunan dari class BPNN untuk menambahkan fitur tracking
    Loss dan Accuracy per epoch.
    """
    def train_with_tracking(self, X, y, epochs=1000, batch_size=16, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # List untuk menyimpan history
        loss_history = []
        accuracy_history = []
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # --- 1. Forward Pass Global (Hanya untuk hitung Metric/Laporan) ---
            # Kita hitung error dan akurasi rata-rata seluruh data sebelum update bobot
            
            # Forward pass manual (gunakan variabel lokal agar tidak ganggu batch training)
            h_input_global = np.dot(X, self.weights_ih) + self.bias_h
            h_output_global = self.sigmoid(h_input_global)
            o_input_global = np.dot(h_output_global, self.weights_ho) + self.bias_o
            output_global = self.softmax(o_input_global)
            
            # --- 2. Calculate Metrics ---
            
            # Calculate MSE Loss
            error = y - output_global
            mse_loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))
            loss_history.append(mse_loss)
            
            # Calculate Accuracy
            predictions = np.argmax(output_global, axis=1)
            targets = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == targets) * 100
            accuracy_history.append(accuracy)
            
            # --- 3. Backward Pass (Training / Weight Update per Batch) ---
            
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward batch
                # PENTING: Kita harus update self.hidden_output agar backward() bisa membacanya
                self.hidden_input = np.dot(X_batch, self.weights_ih) + self.bias_h
                self.hidden_output = self.sigmoid(self.hidden_input) # <--- PERBAIKAN DISINI
                
                output_input = np.dot(self.hidden_output, self.weights_ho) + self.bias_o
                output = self.softmax(output_input)
                
                # Backward batch
                # backward() di bpnn.py menggunakan self.hidden_output
                self.backward(X_batch, y_batch, output)
            
            # Print progress setiap 10% epoch atau epoch pertama/terakhir
            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {mse_loss:.6f} - Acc: {accuracy:.2f}%")
                
        return loss_history, accuracy_history

# ============================================================================
# DATA LOADING (Format Baru: Name_Index.jpg)
# ============================================================================
def load_dataset(dataset_path='dataset', img_size=(64, 64)):
    images = []
    labels = []
    label_names = []
    
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return None, None, None

    files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
    
    for filename in files:
        try:
            # Parse: Name_Index.jpg -> Ambil Name
            parts = filename.split('_')
            name = parts[0]
            
            img = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            images.append(img.flatten())
            labels.append(name)
            
            if name not in label_names:
                label_names.append(name)
        except:
            continue
            
    X = np.array(images)
    label_names.sort()
    label_map = {i: name for i, name in enumerate(label_names)}
    reverse_map = {name: i for i, name in label_map.items()}
    
    y_int = np.array([reverse_map[l] for l in labels])
    y = np.zeros((len(y_int), len(label_names)))
    y[np.arange(len(y_int)), y_int] = 1
    
    return X, y, label_names

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================
def plot_analysis(loss_hist, acc_hist, epochs):
    """
    Membuat grafik Loss dan Accuracy berdampingan
    """
    epochs_range = range(1, len(loss_hist) + 1)
    
    plt.figure(figsize=(15, 6))
    
    # 1. Grafik Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_hist, 'r-', linewidth=2, label='Total Loss (MSE)')
    plt.title('Model Loss per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 2. Grafik Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc_hist, 'b-', linewidth=2, label='Training Accuracy')
    plt.title('Model Accuracy per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105) # Batas 0-100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Target 90%')
    plt.legend()
    
    plt.tight_layout()
    
    # Save graph
    filename = 'training_analysis_graph.png'
    plt.savefig(filename, dpi=300)
    print(f"\n[OK] Grafik disimpan sebagai: {filename}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("ANALISIS AKURASI DAN LOSS BPNN")
    print("=" * 70)
    
    # 1. Konfigurasi
    try:
        max_epochs = int(input("Masukkan jumlah epoch (default 500): "))
    except ValueError:
        max_epochs = 500
        
    img_size = (64, 64)
    hidden_neurons = 128
    learning_rate = 0.1
    
    # 2. Load Data
    print("\n[1/4] Loading Dataset...")
    X, y, label_names = load_dataset(img_size=img_size)
    
    if X is None or len(X) == 0:
        print("[X] Dataset kosong atau tidak ditemukan.")
        return

    input_size = X.shape[1]
    output_size = y.shape[1]
    
    print(f"      Data: {len(X)} samples")
    print(f"      Classes: {len(label_names)} {label_names}")
    print(f"      Input: {input_size} -> Hidden: {hidden_neurons} -> Output: {output_size}")
    
    # 3. Inisialisasi Model Tracking
    print("\n[2/4] Initializing Tracking Model...")
    model = TrackingBPNN(input_size, hidden_neurons, output_size, learning_rate)
    
    # 4. Training dengan Tracking
    print("\n[3/4] Starting Training & Analysis...")
    loss_history, acc_history = model.train_with_tracking(X, y, epochs=max_epochs)
    
    # 5. Visualisasi
    print("\n[4/4] Generating Graphs...")
    
    final_loss = loss_history[-1]
    final_acc = acc_history[-1]
    
    print("-" * 50)
    print(f"FINAL RESULT (Epoch {max_epochs}):")
    print(f"  Total Loss : {final_loss:.6f}")
    print(f"  Accuracy   : {final_acc:.2f}%")
    print("-" * 50)
    
    plot_analysis(loss_history, acc_history, max_epochs)

if __name__ == "__main__":
    main()