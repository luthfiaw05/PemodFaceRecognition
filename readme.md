# Face Recognition Project using BPNN Algorithm (Vector Based)

A complete face recognition system using **Backpropagation Neural Network (BPNN)** with vector-based image processing.

## 📁 Project Structure

```
FaceRec_Project/
│
├── dataset/              # Folder to store training images
├── bpnn.py               # Neural Network Class (BPNN implementation)
├── 0_collect.py          # Collect face images from webcam
├── 1_train.py            # Train the model on dataset
├── 2_recognize.py        # Real-time face recognition (camera)
├── main.py               # ⭐ Main menu system (RECOMMENDED)
├── maincam.py            # Camera mode (fast loading, no training)
└── maininput.py          # Image file input (fast loading, no training)
```

## 🚀 Quick Start

### Prerequisites

Install required packages:

```bash
pip install numpy opencv-python
```

### Option A: Using Main Menu (Recommended)

Simply run the main menu:

```bash
python main.py
```

This provides an interactive menu with all options:
1. Collect Face Images
2. Train Model
3. Real-time Recognition (camera)
4. Recognize from Image File
5. Check System Status
6. Help & Information

### Option B: Manual Workflow

#### Step 1: Collect Face Images

Run the image collection script:

```bash
python 0_collect.py
```

- Enter person's name (e.g., "Fathoni")
- Enter expression (e.g., "Senyum", "Sedih", "Netral")
- Collect 20-50 images per person for best results
- Press **SPACE** to capture, **ESC** to cancel

**Image Naming Format:** `Name_Expression_Index.jpg`

Examples:
- `Fathoni_Senyum_1.jpg`
- `Alice_Netral_10.jpg`
- `Bob_Sedih_5.jpg`

#### Step 2: Train the Model

Train ONLY when you have new images or new people:

```bash
python 1_train.py
```

This creates model files that are reused for fast loading.

#### Step 3: Use Recognition (Fast - No Training!)

**For Camera Mode:**

```bash
python maincam.py
```

- Opens camera immediately
- No training delay
- Press **'q'** to quit

**For Image File Input:**

```bash
python maininput.py
```

- Enter image file path
- Get instant recognition results
- No training delay

## 🧠 How It Works

### 1. **Vector-Based Image Representation**
- Images are converted to grayscale
- Resized to 64x64 pixels
- Flattened into 4096-dimensional vectors
- Normalized to [0, 1] range

### 2. **Neural Network Architecture**

```
Input Layer:  4096 neurons (64x64 flattened image)
      ↓
Hidden Layer: 128 neurons (with sigmoid activation)
      ↓
Output Layer: N neurons (N = number of people)
              (with softmax activation)
```

### 3. **Training Process**
- **Algorithm:** Backpropagation with mini-batch gradient descent
- **Loss Function:** Cross-entropy loss
- **Optimization:** Stochastic gradient descent
- **Epochs:** 500 (default)
- **Batch Size:** 16
- **Learning Rate:** 0.1

### 4. **Recognition Process**
1. Detect faces using Haar Cascade
2. Extract and preprocess face region
3. Convert to feature vector
4. Feed to neural network
5. Get prediction with confidence score

## 📊 Features

### ✨ Real-time Recognition
- Live webcam feed
- Multiple face detection
- Confidence scoring
- Color-coded results:
  - 🟢 Green: High confidence (>70%)
  - 🟡 Yellow: Medium confidence (50-70%)
  - 🔴 Red: Low confidence (<50%)

### 📷 Image File Recognition
- Process any image file
- Batch processing support
- Top-3 predictions display
- Save annotated results

### 🎯 Training Features
- Automatic data loading
- Progress monitoring
- Accuracy evaluation
- Model persistence

## 📝 File Descriptions

### Core Files

| File | Description |
|------|-------------|
| `main.py` | ⭐ Interactive menu system (recommended) |
| `bpnn.py` | Complete BPNN implementation with training and prediction methods |
| `0_collect.py` | Interactive face image collection tool |
| `1_train.py` | Model training script (run once or when dataset changes) |
| `2_recognize.py` | Real-time camera recognition |
| `maincam.py` | Camera mode (fast loading, no training) |
| `maininput.py` | Recognition from image files (fast loading, no training) |

### Generated Files (after training)

| File | Description |
|------|-------------|
| `face_model.pkl` | Trained neural network weights |
| `label_map.pkl` | Mapping between class indices and names |
| `img_size.pkl` | Image size configuration |

## 🎓 Usage Examples

### Example 1: First Time Setup (Single Person)

1. **Collect images:**
   ```bash
   python main.py
   # Select Option 1
   # Enter name: "Fathoni"
   # Enter expression: "Senyum"
   # Collect 30 images
   ```

2. **Train model (ONE TIME):**
   ```bash
   python main.py
   # Select Option 2
   # Wait for training to complete
   ```

3. **Use recognition (FAST):**
   ```bash
   python main.py
   # Select Option 3 for camera
   # OR Option 4 for image file
   ```

### Example 2: Adding New People

1. **Collect new images:**
   - Add `Alice_Netral_1.jpg` ... `Alice_Netral_25.jpg`
   - Add `Bob_Senyum_1.jpg` ... `Bob_Senyum_25.jpg`

2. **Re-train model:**
   ```bash
   python 1_train.py
   ```
   (Only needed once after adding new people)

3. **Use recognition:**
   - Now runs FAST (no training needed)
   - Recognizes all three people

### Example 3: Different Expressions

Mix expressions for robust recognition:
- `Fathoni_Senyum_1.jpg` to `Fathoni_Senyum_10.jpg`
- `Fathoni_Sedih_11.jpg` to `Fathoni_Sedih_20.jpg`
- `Fathoni_Netral_21.jpg` to `Fathoni_Netral_30.jpg`

## ⚙️ Important Notes

### When to Train vs When to Recognize

**TRAIN (Run `1_train.py`) when:**
- First time setup
- Adding new people to dataset
- Adding more images of existing people
- Recognition accuracy is low

**RECOGNIZE (Run `maincam.py` or `maininput.py`) for:**
- Daily use (FAST - no training)
- After model is already trained
- Testing the system
- Demo purposes

### Training vs Recognition Speed

- **Training:** Takes 2-5 minutes (depends on dataset size)
- **Recognition:** Instant loading (uses saved model)

This separation allows you to:
- Change dataset without re-running recognition
- Use recognition multiple times without retraining
- Share trained models without sharing dataset

## ⚙️ Configuration Options

### Modify Training Parameters

Edit `1_train.py`:

```python
# Network architecture
input_size = X.shape[1]  # Automatic
hidden_size = 128        # Change this for more/fewer neurons
output_size = y.shape[1] # Automatic

# Training parameters
model.train(X, y, 
    epochs=500,      # Number of training iterations
    batch_size=16,   # Batch size
    verbose=True     # Show progress
)

# Learning rate
model = BPNN(input_size, hidden_size, output_size, 
             learning_rate=0.1)  # Adjust this
```

### Modify Image Size

Edit `1_train.py`:

```python
IMG_SIZE = (64, 64)  # Change to (32, 32) or (128, 128)
```

## 🔧 Troubleshooting

### "No images found in dataset"
- Make sure images are in the `dataset/` folder
- Check file extensions (.jpg, .jpeg, .png)
- Verify naming format: `Name_Expression_Index.jpg`

### "Cannot access camera"
- Check if camera is connected
- Close other applications using the camera
- Try a different camera index in `cv2.VideoCapture(0)` → change `0` to `1`

### Low Accuracy
- Collect more images per person (30-50 recommended)
- Use consistent lighting conditions
- Include different angles and expressions
- Increase hidden layer size or epochs

### "Model file not found"
- Run `1_train.py` or `maincam.py` first
- Make sure training completed successfully

## 📚 Technical Details

### BPNN Algorithm

**Forward Propagation:**
```
hidden = sigmoid(X · W_ih + b_h)
output = softmax(hidden · W_ho + b_o)
```

**Backward Propagation:**
```
output_error = output - y_true
hidden_error = output_error · W_ho^T
```

**Weight Update:**
```
W_ho = W_ho - lr · (hidden^T · output_error)
W_ih = W_ih - lr · (X^T · hidden_error)
```

### Activation Functions

- **Sigmoid:** Used in hidden layer
  - Formula: `σ(x) = 1 / (1 + e^(-x))`
  - Range: (0, 1)

- **Softmax:** Used in output layer
  - Formula: `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`
  - Outputs probability distribution

### Loss Function

- **Cross-Entropy Loss:**
  - Formula: `L = -Σ(y_true · log(y_pred))`
  - Measures difference between predicted and true distributions

## 🎯 Best Practices

1. **Data Collection:**
   - Collect 30-50 images per person
   - Use good lighting
   - Vary expressions and angles
   - Keep face centered

2. **Training:**
   - Use consistent image sizes
   - Monitor training accuracy
   - Retrain if adding new people

3. **Recognition:**
   - Ensure good lighting
   - Face camera directly
   - Maintain reasonable distance

## 📈 Performance Tips

- **Speed:** Reduce image size for faster processing
- **Accuracy:** Increase hidden neurons or collect more data
- **Memory:** Use smaller batch sizes if running out of memory

## 🤝 Contributing

Feel free to modify and improve the code:
- Add data augmentation
- Implement learning rate decay
- Add validation set
- Experiment with different architectures

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

## 👨‍💻 Author

Created as a demonstration of face recognition using Backpropagation Neural Networks with vector-based image processing.

---

**Happy Coding! 🚀**