import os
import sys

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print main header"""
    print("=" * 70)
    print("          FACE RECOGNITION SYSTEM - BPNN")
    print("          Backpropagation Neural Network")
    print("=" * 70)

def check_model_status():
    """Check if model is trained"""
    if os.path.exists('face_model.pkl'):
        print("\n[Status] Model: TRAINED")
        
        # Check dataset
        if os.path.exists('dataset'):
            files = [f for f in os.listdir('dataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"[Status] Dataset: {len(files)} images")
        else:
            print("[Status] Dataset: NOT FOUND")
            
        # Check labels
        if os.path.exists('label_map.pkl'):
            import pickle
            with open('label_map.pkl', 'rb') as f:
                label_map = pickle.load(f)
            print(f"[Status] People: {list(label_map.values())}")
    else:
        print("\n[Status] Model: NOT TRAINED")
        print("[Info] You need to train the model first (Option 2)")

def print_menu():
    """Print main menu"""
    print("\n" + "-" * 70)
    print("MAIN MENU:")
    print("-" * 70)
    print("  1. Collect Face Images (from webcam)")
    print("  2. Train Model (required once or when dataset changes)")
    print("  3. Real-time Recognition (camera)")
    print("  4. Recognize from Image File")
    print("  5. Check System Status")
    print("  6. Help & Information")
    print("  0. Exit")
    print("-" * 70)

def collect_images():
    """Run image collection script"""
    clear_screen()
    print_header()
    print("\n>>> COLLECTING FACE IMAGES <<<\n")
    
    if not os.path.exists('0_collect.py'):
        print("[X] ERROR: '0_collect.py' not found!")
        input("\nPress Enter to continue...")
        return
    
    os.system(f'"{sys.executable}" 0_collect.py')
    input("\nPress Enter to continue...")

def train_model():
    """Run training script"""
    clear_screen()
    print_header()
    print("\n>>> TRAINING MODEL <<<\n")
    
    if not os.path.exists('1_train.py'):
        print("[X] ERROR: '1_train.py' not found!")
        input("\nPress Enter to continue...")
        return
    
    if not os.path.exists('dataset'):
        print("[X] ERROR: 'dataset' folder not found!")
        print("\nPlease collect images first (Option 1)")
        input("\nPress Enter to continue...")
        return
    
    files = [f for f in os.listdir('dataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) == 0:
        print("[X] ERROR: No images in 'dataset' folder!")
        print("\nPlease collect images first (Option 1)")
        input("\nPress Enter to continue...")
        return
    
    print(f"[OK] Found {len(files)} images in dataset")
    print("\nStarting training...\n")
    
    exit_code = os.system(f'"{sys.executable}" 1_train.py')
    
    if exit_code == 0:
        print("\n[OK] Training completed successfully!")
    else:
        print("\n[X] Training failed. Check errors above.")
    
    input("\nPress Enter to continue...")

def camera_recognition():
    """Run camera recognition"""
    clear_screen()
    print_header()
    print("\n>>> REAL-TIME CAMERA RECOGNITION <<<\n")
    
    if not os.path.exists('face_model.pkl'):
        print("[X] ERROR: Model not trained!")
        print("\nPlease train the model first (Option 2)")
        input("\nPress Enter to continue...")
        return
    
    print("[OK] Model found. Launching camera...\n")
    os.system(f'"{sys.executable}" maincam.py')
    input("\nPress Enter to continue...")

def image_recognition():
    """Run image file recognition"""
    clear_screen()
    print_header()
    print("\n>>> IMAGE FILE RECOGNITION <<<\n")
    
    if not os.path.exists('face_model.pkl'):
        print("[X] ERROR: Model not trained!")
        print("\nPlease train the model first (Option 2)")
        input("\nPress Enter to continue...")
        return
    
    print("[OK] Model found. Starting recognition...\n")
    os.system(f'"{sys.executable}" maininput.py')
    input("\nPress Enter to continue...")

def show_status():
    """Show system status"""
    clear_screen()
    print_header()
    print("\n>>> SYSTEM STATUS <<<")
    
    print("\n" + "-" * 70)
    print("FILE STATUS:")
    print("-" * 70)
    
    required_files = ['bpnn.py', '0_collect.py', '1_train.py', 
                      '2_recognize.py', 'maincam.py', 'maininput.py']
    
    for file in required_files:
        status = "[OK]" if os.path.exists(file) else "[X] MISSING"
        print(f"  {status} {file}")
    
    print("\n" + "-" * 70)
    print("MODEL STATUS:")
    print("-" * 70)
    
    model_files = ['face_model.pkl', 'label_map.pkl', 'img_size.pkl']
    for file in model_files:
        status = "[OK]" if os.path.exists(file) else "[X] NOT FOUND"
        print(f"  {status} {file}")
    
    print("\n" + "-" * 70)
    print("DATASET STATUS:")
    print("-" * 70)
    
    if os.path.exists('dataset'):
        files = [f for f in os.listdir('dataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  [OK] Dataset folder exists")
        print(f"  [OK] {len(files)} images found")
        
        if len(files) > 0:
            # Count images per person
            people = {}
            for file in files:
                try:
                    name = file.split('_')[0]
                    people[name] = people.get(name, 0) + 1
                except:
                    pass
            
            print(f"\n  People in dataset:")
            for name, count in people.items():
                print(f"    - {name}: {count} images")
    else:
        print("  [X] Dataset folder not found")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")
    print("-" * 70)
    
    if not os.path.exists('face_model.pkl'):
        print("  ! Train the model (Option 2)")
    if not os.path.exists('dataset'):
        print("  ! Create dataset folder and collect images (Option 1)")
    
    if os.path.exists('dataset'):
        files = [f for f in os.listdir('dataset') if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) == 0:
            print("  ! Collect face images (Option 1)")
        elif len(files) < 20:
            print("  ! Collect more images for better accuracy (20-50 per person)")
    
    input("\nPress Enter to continue...")

def show_help():
    """Show help information"""
    clear_screen()
    print_header()
    print("\n>>> HELP & INFORMATION <<<")
    
    print("\n" + "-" * 70)
    print("WORKFLOW:")
    print("-" * 70)
    print("""
  FIRST TIME SETUP:
  
    1. Collect Images (Option 1)
       - Use your webcam to capture face images
       - Collect 20-50 images per person
       - Images saved as: Name_Expression_Index.jpg
       - Example: Fathoni_Senyum_1.jpg
    
    2. Train Model (Option 2)
       - Trains the neural network on your images
       - Takes a few minutes depending on dataset size
       - Creates model files (face_model.pkl, etc.)
       - Only needed ONCE or when you add new people
    
    3. Use Recognition (Option 3 or 4)
       - Option 3: Real-time camera recognition
       - Option 4: Recognize from image files
       - Fast loading (no training needed)
    
  ADDING NEW PEOPLE:
  
    1. Collect new images (Option 1)
    2. Re-train model (Option 2)
    3. Start using recognition again
""")
    
    print("\n" + "-" * 70)
    print("IMAGE NAMING FORMAT:")
    print("-" * 70)
    print("""
  Format: Name_Expression_Index.jpg
  
  Examples:
    - Fathoni_Senyum_1.jpg    (Fathoni, smiling, image 1)
    - Alice_Netral_10.jpg     (Alice, neutral, image 10)
    - Bob_Sedih_5.jpg         (Bob, sad, image 5)
  
  Note: The expression field is for organization only.
        The model recognizes people, not expressions.
""")
    
    print("\n" + "-" * 70)
    print("TIPS FOR BEST RESULTS:")
    print("-" * 70)
    print("""
  - Collect 30-50 images per person
  - Use good, consistent lighting
  - Include different angles (slightly left/right)
  - Include different expressions
  - Keep face centered and visible
  - Avoid glasses glare if possible
  - Use similar lighting for training and recognition
""")
    
    print("\n" + "-" * 70)
    print("TECHNICAL INFO:")
    print("-" * 70)
    print("""
  Algorithm: Backpropagation Neural Network (BPNN)
  Architecture: Input (4096) -> Hidden (128) -> Output (N people)
  Image Size: 64x64 pixels (grayscale)
  Feature Vector: 4096 dimensions (flattened image)
  Training: 500 epochs, batch size 16
  Face Detection: Haar Cascade Classifier
""")
    
    input("\nPress Enter to continue...")

def main():
    """Main program loop"""
    
    while True:
        clear_screen()
        print_header()
        check_model_status()
        print_menu()
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            collect_images()
        elif choice == '2':
            train_model()
        elif choice == '3':
            camera_recognition()
        elif choice == '4':
            image_recognition()
        elif choice == '5':
            show_status()
        elif choice == '6':
            show_help()
        elif choice == '0':
            clear_screen()
            print("\nThank you for using Face Recognition System!")
            print("Goodbye!\n")
            break
        else:
            print("\n[X] Invalid choice! Please enter 0-6.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Warning] Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")
        input("Press Enter to exit...")