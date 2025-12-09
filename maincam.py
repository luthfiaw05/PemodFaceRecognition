import os
import sys

def run_camera_recognition():
    """
    Launch camera for real-time face recognition
    (Assumes model is already trained)
    """
    
    print("=" * 60)
    print("     REAL-TIME FACE RECOGNITION - CAMERA MODE")
    print("=" * 60)
    
    # --- CHECK REQUIREMENTS ---
    print("\n[Checking Requirements...]")
    
    if not os.path.exists("bpnn.py"):
        print("[X] CRITICAL ERROR: 'bpnn.py' is missing!")
        print("Please ensure all required files are present.")
        return
    
    if not os.path.exists("2_recognize.py"):
        print("[X] CRITICAL ERROR: '2_recognize.py' is missing!")
        return
    
    # Check if model exists
    if not os.path.exists("face_model.pkl"):
        print("[X] ERROR: Model not found!")
        print("\nYou need to train the model first.")
        print("\nOptions:")
        print("  1. Run 'python 1_train.py' to train the model")
        print("  2. Make sure you have images in the 'dataset' folder")
        print("\nImage naming format: Name_Expression_Index.jpg")
        print("Example: Fathoni_Senyum_1.jpg")
        return
    
    print("[OK] Model found!")
    print("[OK] All requirements met!")
    
    # --- LAUNCH CAMERA ---
    print("\n" + "=" * 60)
    print("           LAUNCHING CAMERA")
    print("=" * 60)
    print("\nStarting camera for face recognition...")
    print("Press 'q' inside the camera window to exit.\n")
    
    # Run recognition script
    os.system(f'"{sys.executable}" 2_recognize.py')
    
    # --- COMPLETION ---
    print("\n" + "=" * 60)
    print("           SESSION ENDED")
    print("=" * 60)

def print_usage_info():
    """Print usage information"""
    print("\n" + "=" * 60)
    print("           HOW TO USE THIS SYSTEM")
    print("=" * 60)
    print("\n[WORKFLOW]")
    print("\n1. COLLECT IMAGES (First time setup):")
    print("   Run: python 0_collect.py")
    print("   - Collect 20-50 images per person")
    print("   - Images saved to 'dataset' folder")
    
    print("\n2. TRAIN MODEL (Required once, or when dataset changes):")
    print("   Run: python 1_train.py")
    print("   - Trains the neural network")
    print("   - Creates model files")
    print("   - Only needed when you add new people or images")
    
    print("\n3. CAMERA RECOGNITION (This script):")
    print("   Run: python maincam.py")
    print("   - Opens camera for real-time recognition")
    print("   - Fast loading (no training)")
    
    print("\n4. IMAGE FILE RECOGNITION:")
    print("   Run: python maininput.py")
    print("   - Recognize faces from image files")
    print("   - Fast loading (no training)")
    
    print("\n" + "=" * 60)
    print("[WHEN TO RETRAIN]")
    print("=" * 60)
    print("  - When you add new people to dataset")
    print("  - When you add more images of existing people")
    print("  - When recognition accuracy is low")
    print("\n  Simply run: python 1_train.py")
    
    print("\n" + "=" * 60)
    print("IMAGE NAMING FORMAT:")
    print("  Name_Expression_Index.jpg")
    print("  Example: Fathoni_Senyum_1.jpg")
    print("=" * 60)

if __name__ == "__main__":
    print("\n*** FACE RECOGNITION SYSTEM - CAMERA MODE ***")
    print("    Fast Loading (Model-based Recognition)")
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage_info()
    else:
        run_camera_recognition()