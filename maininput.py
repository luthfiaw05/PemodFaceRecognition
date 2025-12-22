import cv2
import numpy as np
import pickle
import os
from bpnn import BPNN

# KONFIGURASI THRESHOLD
# Jika confidence di bawah angka ini, dianggap "Unidentified"
CONFIDENCE_THRESHOLD = 60.0 

def load_model_and_labels():
    """Load trained model and label mapping"""
    
    if not os.path.exists('face_model.pkl'):
        print("[X] ERROR: Model file 'face_model.pkl' not found!")
        print("\nYou need to train the model first.")
        print("\nRun: python 1_train.py")
        print("\nMake sure you have images in the 'dataset' folder.")
        return None, None, None
    
    # Load model
    model = BPNN(0, 0, 0)  # Dummy initialization
    model.load_model('face_model.pkl')
    
    # Load label mapping
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    # Load image size
    with open('img_size.pkl', 'rb') as f:
        img_size = pickle.load(f)
    
    print("[OK] Model loaded successfully!")
    print(f"[OK] Recognized people: {list(label_map.values())}")
    
    return model, label_map, img_size

def preprocess_face(face_img, img_size):
    """Preprocess face image for prediction"""
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img
    
    # Resize to match training size
    face_resized = cv2.resize(face_gray, img_size)
    
    # Normalize
    face_normalized = face_resized.astype(np.float32) / 255.0
    
    # Flatten to vector
    face_vector = face_normalized.flatten()
    
    return face_vector

def recognize_from_image(image_path, model, label_map, img_size):
    """Recognize faces in a given image"""
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"[X] ERROR: Image file '{image_path}' not found!")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"[X] ERROR: Could not read image '{image_path}'")
        print("Supported formats: .jpg, .jpeg, .png, .bmp")
        return
    
    print(f"\n[OK] Image loaded: {image_path}")
    print(f"  Size: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    
    print(f"\n[Detected] {len(faces)} face(s) in the image")
    
    if len(faces) == 0:
        print("\n[Warning] No faces detected in the image!")
        cv2.imshow('No Faces Detected', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    print("\n" + "=" * 60)
    print("RECOGNITION RESULTS:")
    print("=" * 60)
    
    # Process each detected face
    for idx, (x, y, w, h) in enumerate(faces, 1):
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_vector = preprocess_face(face_img, img_size)
        face_vector = face_vector.reshape(1, -1)
        
        # Predict
        prediction_proba = model.predict_proba(face_vector)
        predicted_class = np.argmax(prediction_proba)
        confidence = prediction_proba[0][predicted_class] * 100
        
        # --- LOGIC UNIDENTIFIED / UNKNOWN ---
        is_unknown = False
        
        if confidence < CONFIDENCE_THRESHOLD:
            is_unknown = True
            predicted_name = "Unidentified"
            display_label = "Unidentified"
            color = (100, 100, 100) # Abu-abu gelap
            text_color = (255, 255, 255)
        else:
            predicted_name = label_map[predicted_class]
            display_label = predicted_name
            
            # Color coding for known faces
            if confidence > 80:
                color = (0, 255, 0)      # Hijau (Sangat Yakin)
                text_color = (0, 0, 0)
            elif confidence > 65:
                color = (0, 255, 255)    # Kuning (Yakin)
                text_color = (0, 0, 0)
            else:
                color = (0, 0, 255)      # Merah (Kurang Yakin tapi masih diatas threshold)
                text_color = (255, 255, 255)

        # Print results to terminal
        print(f"\nFace #{idx}:")
        print(f"  Result: {predicted_name}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Tampilkan Top 3 hanya jika dikenal atau confidence lumayan
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        print(f"  Top 3 Candidates:")
        for rank, class_idx in enumerate(top_3_indices, 1):
            label = label_map[class_idx]
            conf = prediction_proba[0][class_idx] * 100
            print(f"     {rank}. {label}: {conf:.2f}%")
        
        if is_unknown:
            print(f"  [!] Threshold Check: FAILED (< {CONFIDENCE_THRESHOLD}%)")
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Display name and confidence
        label_text = f"{display_label} ({confidence:.1f}%)"
        
        # Background for text
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw filled rectangle for text background
        cv2.rectangle(frame, (x, y-35), (x + text_w + 10, y), color, -1)
        
        # Draw text
        cv2.putText(frame, label_text, (x+5, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        # Add face number
        cv2.putText(frame, f"#{idx}", (x, y+h+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    print("\n" + "=" * 60)
    
    # Display result
    print("\n[Display] Showing result... (Press any key to close)")
    
    # Resize for display if too large
    max_width = 1200
    max_height = 800
    h, w = frame.shape[:2]
    
    if w > max_width or h > max_height:
        scale = min(max_width/w, max_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    cv2.imshow('Face Recognition Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ask to save result
    save = input("\n[Save] Save result image? (y/n): ").strip().lower()
    if save == 'y':
        output_name = "result_" + os.path.basename(image_path)
        cv2.imwrite(output_name, frame)
        print(f"[OK] Result saved as: {output_name}")

def main():
    """Main function for image-based recognition"""
    
    print("=" * 60)
    print("   FACE RECOGNITION FROM IMAGE FILE - INPUT MODE")
    print("=" * 60)
    
    # Load model
    print("\n[Loading Model...]")
    model, label_map, img_size = load_model_and_labels()
    
    if model is None:
        print("\n[Info] Train the model first:")
        print("  Run: python 1_train.py")
        print("\nMake sure you have images in 'dataset' folder.")
        return
    
    print("\n" + "-" * 60)
    print(f"[INFO] Confidence Threshold: {CONFIDENCE_THRESHOLD}%")
    print("Faces with confidence below this value will be 'Unidentified'")
    print("-" * 60)
    
    # Get image path from user
    while True:
        print("\nEnter image path (or 'q' to quit):")
        image_path = input(">>> ").strip().strip('"').strip("'")
        
        if image_path.lower() == 'q':
            print("\nGoodbye!")
            break
        
        if image_path == "":
            continue
            
        recognize_from_image(image_path, model, label_map, img_size)
        
        print("\n" + "-" * 60)
        
        another = input("\nProcess another image? (y/n): ").strip().lower()
        if another != 'y':
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Warning] Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")