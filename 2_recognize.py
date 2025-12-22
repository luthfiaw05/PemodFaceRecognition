import cv2
import numpy as np
import pickle
import os
from bpnn import BPNN
import time  # Ditambahkan untuk perhitungan FPS yang lebih akurat jika diperlukan

def load_model_and_labels():
    """Load trained model and label mapping"""
    
    if not os.path.exists('face_model.pkl'):
        print("ERROR: Model file 'face_model.pkl' not found!")
        print("Please run '1_train.py' first to train the model.")
        return None, None, None
    
    # Load model
    # Initialize with dummy values, actual structure is loaded from file
    model = BPNN(input_size=1, hidden_size=1, output_size=1) 
    model.load_model('face_model.pkl')
    
    # Load label mapping
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    # Load image size
    with open('img_size.pkl', 'rb') as f:
        img_size = pickle.load(f)
    
    print("Model and labels loaded successfully!")
    print(f"Recognized people: {list(label_map.values())}")
    
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

def recognize_faces():
    """Real-time face recognition using webcam"""
    
    print("=" * 60)
    print("         REAL-TIME FACE RECOGNITION - BPNN")
    print("=" * 60)
    
    # Load model
    model, label_map, img_size = load_model_and_labels()
    
    if model is None:
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        return
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("\n[OK] Camera started successfully!")
    print("Press 'q' to quit\n")
    
    # For FPS calculation
    fps_counter = 0
    fps_start = cv2.getTickCount()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        # Process each detected face
        for (x, y, w, h) in faces:
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
            # HAPUS LOGIKA EKSPRESI DISINI
            # Karena label_map sekarang hanya berisi "Nama", kita langsung pakai saja.
            predicted_name = label_map[predicted_class]
            
            display_label = predicted_name
            
            # Choose color based on confidence
            if confidence > 80:
                color = (0, 255, 0)      # Green - High confidence
            elif confidence > 60:
                color = (0, 255, 255)    # Yellow - Medium confidence
            else:
                color = (0, 0, 255)      # Red - Low confidence
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display name and confidence
            label_text = f"{display_label} ({confidence:.1f}%)"
            
            # Background for text
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-35), (x + text_size[0] + 10, y), color, -1)
            
            # Draw text
            # Gunakan warna hitam (0,0,0) untuk teks jika background kuning/hijau agar kontras,
            # atau putih (255,255,255) jika background merah/biru. 
            # Disini kita pakai hitam/putih simple logic atau statis.
            text_color = (0, 0, 0) if confidence > 60 else (255, 255, 255)
            
            cv2.putText(frame, label_text, (x+5, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 15: # Update setiap 15 frame agar tidak terlalu flicker
            fps_end = cv2.getTickCount()
            time_diff = (fps_end - fps_start) / cv2.getTickFrequency()
            fps = fps_counter / time_diff
            fps_counter = 0
            fps_start = cv2.getTickCount()
        
        # Display info
        info_text = f"Faces: {len(faces)} | Press 'q' to quit"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Recognition', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera stopped. Goodbye!")

if __name__ == "__main__":
    recognize_faces()