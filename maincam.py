import os
import sys
import time

def run_pipeline():
    """
    Automated pipeline for face recognition:
    1. Train the model on dataset
    2. Launch camera for real-time recognition
    """
    
    print("=" * 60)
    print("     AUTOMATED FACE RECOGNITION SYSTEM - BPNN")
    print("=" * 60)
    
    # --- CHECK REQUIREMENTS ---
    print("\n[Checking Requirements...]")
    
    if not os.path.exists("bpnn.py"):
        print("[X] CRITICAL ERROR: 'bpnn.py' is missing!")
        print("Please ensure all required files are present.")
        return
    
    if not os.path.exists("dataset"):
        print("[X] ERROR: 'dataset' folder not found!")
        print("\nPlease create 'dataset' folder and add face images.")
        print("Image naming format: Name_Expression_Index.jpg")
        print("Example: Fathoni_Senyum_1.jpg")
        print("\nYou can run '0_collect.py' to collect images using webcam.")
        return
    
    # Check if dataset has images
    dataset_files = [f for f in os.listdir("dataset") if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(dataset_files) == 0:
        print("[X] ERROR: 'dataset' folder is empty!")
        print("\nPlease add face images to the dataset folder.")
        print("Image naming format: Name_Expression_Index.jpg")
        print("\nYou can run '0_collect.py' to collect images using webcam.")
        return
    
    print(f"[OK] Found {len(dataset_files)} images in dataset")
    print("[OK] All requirements met!")
    
    # --- STEP 1: TRAINING ---
    print("\n" + "=" * 60)
    print("STEP 1/2: Training Neural Network")
    print("=" * 60)
    print("\nThis will train the BPNN model on your dataset...")
    print("Please wait, this may take a few minutes.\n")
    
    time.sleep(2)
    
    # Run training script
    train_exit_code = os.system(f'"{sys.executable}" 1_train.py')
    
    if train_exit_code != 0:
        print("\n[X] CRITICAL ERROR: Training failed!")
        print("\nPossible reasons:")
        print("1. Corrupted or invalid images in dataset")
        print("2. Insufficient memory")
        print("3. Missing dependencies (numpy, opencv-python)")
        print("\nPlease check the error messages above for details.")
        return
    
    print("\n[OK] Training completed successfully!")
    time.sleep(2)
    
    # --- STEP 2: RECOGNITION ---
    print("\n" + "=" * 60)
    print("STEP 2/2: Launching Real-Time Face Recognition")
    print("=" * 60)
    print("\nStarting camera for face recognition...")
    print("Press 'q' inside the camera window to exit.\n")
    
    time.sleep(2)
    
    # Run recognition script
    os.system(f'"{sys.executable}" 2_recognize.py')
    
    # --- COMPLETION ---
    print("\n" + "=" * 60)
    print("           SESSION COMPLETED")
    print("=" * 60)
    print("\nThank you for using the Face Recognition System!")
    print("To run again, simply execute this script.")

def print_usage_info():
    """Print usage information"""
    print("\n" + "=" * 60)
    print("           HOW TO USE THIS SYSTEM")
    print("=" * 60)
    print("\n1. COLLECT IMAGES (Optional - if dataset is empty):")
    print("   Run: python 0_collect.py")
    print("   - Enter person's name and expression")
    print("   - Position face in camera and press SPACE to capture")
    print("   - Collect 20-50 images per person for best results")
    
    print("\n2. TRAIN & RECOGNIZE (Current script):")
    print("   Run: python maincam.py")
    print("   - Automatically trains model on dataset")
    print("   - Opens camera for real-time recognition")
    
    print("\n3. RECOGNIZE FROM IMAGE FILE:")
    print("   Run: python maininput.py")
    print("   - Enter path to image file for recognition")
    
    print("\n" + "=" * 60)
    print("IMAGE NAMING FORMAT:")
    print("  Name_Expression_Index.jpg")
    print("  Example: Fathoni_Senyum_1.jpg, Alice_Neutral_5.jpg")
    print("=" * 60)

if __name__ == "__main__":
    print("\n*** FACE RECOGNITION SYSTEM ***")
    print("    Using Backpropagation Neural Network (BPNN)")
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage_info()
    else:
        run_pipeline()