import cv2
import os

def collect_faces():
    """
    Collect face images from webcam
    Images are saved as: Name_Index.jpg
    """
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        print("Created 'dataset' folder")
    
    name = input("Enter person's name: ").strip()
    
    try:
        num_images = int(input("How many images to collect? (recommended: 20-50): "))
    except ValueError:
        print("Invalid input for number of images. Defaulting to 20.")
        num_images = 20
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print(f"\nCollecting {num_images} images for '{name}'...")
    print("Press SPACE to capture, ESC to cancel")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Collected: {count}/{num_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Collect Face Images', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                count += 1
                filename = f"{name}_{count}.jpg"
                filepath = os.path.join('dataset', filename)
                
                cv2.imwrite(filepath, face_img)
                print(f"Saved: {filename}")
            else:
                print("No face detected! Please position your face properly.")
        
        elif key == 27:
            print("\nCollection cancelled by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[OK] Successfully collected {count} images!")
    print(f"Images saved in 'dataset' folder")

if __name__ == "__main__":
    print("=" * 50)
    print("       FACE IMAGE COLLECTION TOOL")
    print("=" * 50)
    collect_faces()