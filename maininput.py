import cv2
import numpy as np
import pickle
import os
from bpnn import BPNN

def load_model_and_labels():
    """Load trained model and label mapping"""
    
    if not os.path.exists('face_model.pkl'):
        print("[X] ERROR: Model file 'face_model.pkl' not found!")
        print("Please run '1_train.py' first to train the model.")
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
    
    print("[OK] Model and labels loaded successfully!")
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
        print("Tips:")
        print("  - Make sure the face is clearly visible")
        print("  - Face should be well-lit and frontal")
        print("  - Try a different image")
        
        # Display image anyway
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
        
        # Get predicted name
        predicted_name = label_map[predicted_class]
        
        # Print results
        print(f"\nFace #{idx}:")
        print(f"  Predicted: {predicted_name}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        print(f"  Top 3 predictions:")
        for rank, class_idx in enumerate(top_3_indices, 1):
            name = label_map[class_idx]
            conf = prediction_proba[0][class_idx] * 100
            print(f"     {rank}. {name}: {conf:.2f}%")
        
        # Choose color based on confidence
        if confidence > 70:
            color = (0, 255, 0)  # Green - High confidence
            conf_text = "HIGH"
        elif confidence > 50:
            color = (0, 255, 255)  # Yellow - Medium confidence
            conf_text = "MEDIUM"
        else:
            color = (0, 0, 255)  # Red - Low confidence
            conf_text = "LOW"
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Display name and confidence
        label = f"{predicted_name} ({confidence:.1f}%)"
        
        # Background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x, y-40), (x + text_size[0] + 10, y), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x+5, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add face number
        cv2.putText(frame, f"Face #{idx}", (x+5, y+h+25),
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
    print("   FACE RECOGNITION FROM IMAGE FILE - BPNN")
    print("=" * 60)
    
    # Load model
    model, label_map, img_size = load_model_and_labels()
    
    if model is None:
        print("\n[Warning] Please train the model first by running:")
        print("   python 1_train.py")
        print("   OR")
        print("   python maincam.py")
        return
    
    print("\n" + "-" * 60)
    
    # Get image path from user
    while True:
        print("\nEnter image path (or 'q' to quit):")
        image_path = input(">>> ").strip()
        
        if image_path.lower() == 'q':
            print("\nGoodbye!")
            break
        
        # Remove quotes if present
        image_path = image_path.strip('"\'')
        
        # Recognize faces in image
        recognize_from_image(image_path, model, label_map, img_size)
        
        print("\n" + "-" * 60)
        
        # Ask if user wants to process another image
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