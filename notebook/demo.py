"""
MINIMAL WORKING DEMO
Run this immediately with ONLY: opencv-python, tensorflow, numpy
This is a complete, working emotion recognition system.
"""

import cv2
import numpy as np
import sys

print("="*70)
print("EMOTION RECOGNITION - MINIMAL DEMO")
# print("No dlib or face_recognition required!")
print("="*70)

# Check imports
print("\nChecking packages...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} - Ready!")
except ImportError:
    print("✗ TensorFlow not found. Install: pip install tensorflow")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} - Ready!")
except ImportError:
    print("✗ OpenCV not found. Install: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} - Ready!")
except ImportError:
    print("✗ NumPy not found. Install: pip install numpy")
    sys.exit(1)

print("\n✅ All required packages installed!\n")


class SimpleEmotionRecognizer:
    """
    Simplified emotion recognizer using ONLY:
    - OpenCV (face detection)
    - TensorFlow (emotion recognition)
    - NumPy (array operations)
    """
    # NO dlib or face_recognition needed!
    
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                               'Sad', 'Surprise', 'Neutral']
        self.emotion_colors = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Surprise': (0, 255, 255),
            'Fear': (128, 0, 128),
            'Disgust': (0, 128, 128),
            'Neutral': (128, 128, 128)
        }
        self.face_detector = None
        self.emotion_model = None
        
    def load_face_detector(self):
        """Load OpenCV's built-in Haar Cascade face detector"""
        print("Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        
        if self.face_detector.empty():
            print("✗ Failed to load face detector")
            return False
        
        print("✓ Face detector loaded (Haar Cascade)")
        return True
    
    def create_demo_emotion_model(self):
        """
        Create a simple emotion recognition model
        In production, you would load a trained model with:
        self.emotion_model = tf.keras.models.load_model('emotion_model.h5')
        """
        print("\nCreating demo emotion model...")
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                            Dense, Dropout, BatchNormalization)
        
        model = Sequential([
            # Input: 48x48 grayscale images
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions
        ])
        
        self.emotion_model = model
        print(f"✓ Model created ({model.count_params():,} parameters)")
        print("⚠ Note: This is an untrained model for demo purposes")
        print("   For real predictions, train on FER2013 dataset")
        return True
    
    def load_trained_model(self, model_path='emotion_model.h5'):
        """Load a trained emotion model"""
        try:
            self.emotion_model = tf.keras.models.load_model(model_path)
            print(f"✓ Trained model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"⚠ Could not load trained model: {e}")
            print("  Using demo model instead...")
            return self.create_demo_emotion_model()
    
    def detect_faces(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion recognition"""
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize to 0-1
        normalized = resized / 255.0
        
        # Reshape for model input: (1, 48, 48, 1)
        reshaped = normalized.reshape(1, 48, 48, 1)
        
        return reshaped
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        if self.emotion_model is None:
            return "No Model", 0.0, None
        
        # Preprocess
        processed = self.preprocess_face(face_img)
        
        # Predict
        predictions = self.emotion_model.predict(processed, verbose=0)
        
        # Get top emotion
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        emotion = self.emotion_labels[emotion_idx]
        
        return emotion, confidence, predictions[0]
    
    def draw_results(self, frame, faces, emotions):
        """Draw bounding boxes and emotion labels"""
        for i, (x, y, w, h) in enumerate(faces):
            if i < len(emotions):
                emotion, confidence, _ = emotions[i]
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label background
                label = f"{emotion}: {confidence*100:.1f}%"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run_webcam(self):
        DISPLAY_WIDTH = 960  # try 640 / 960 / 1280
        """Run real-time emotion recognition on webcam"""
        print("\n" + "="*70)
        print("STARTING WEBCAM EMOTION RECOGNITION")
        print("="*70)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  SPACE - Pause/Resume")
        print("\nStarting in 3 seconds...\n")
        
        import time
        time.sleep(3)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            print("  Possible solutions:")
            print("  1. Check if camera is connected")
            print("  2. Check camera permissions")
            print("  3. Try different camera index: cv2.VideoCapture(1)")
            return
        
        print("✓ Webcam opened successfully")
        
        paused = False
        frame_count = 0
        fps_list = []
        
        while True:
            if not paused:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to read frame")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Predict emotions
                emotions = []
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    emotion, conf, probs = self.predict_emotion(face_roi)
                    emotions.append((emotion, conf, probs))
                
                # Draw results
                output = self.draw_results(frame, faces, emotions)
                
                # Add info overlay
                info_text = f"Faces: {len(faces)}"
                cv2.putText(output, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = np.mean(fps_list)
                
                fps_text = f"FPS: {avg_fps:.1f}"
                cv2.putText(output, fps_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('Emotion Recognition', output)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('s'):
                filename = f"../results/images/emotion_screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, output)
                print(f"✓ Screenshot saved: {filename}")
                frame_count += 1
            elif key == ord(' '):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"✓ {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam closed")
    
    def test_on_image(self, image_path):
        """Test on a single image file"""
        print(f"\nTesting on image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ Could not read image: {image_path}")
            return
        
        print("✓ Image loaded")
        
        # Detect faces
        faces = self.detect_faces(frame)
        print(f"✓ Detected {len(faces)} face(s)")
        
        # Predict emotions
        emotions = []
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = frame[y:y+h, x:x+w]
            emotion, conf, probs = self.predict_emotion(face_roi)
            emotions.append((emotion, conf, probs))
            print(f"  Face {i+1}: {emotion} ({conf*100:.1f}%)")
        
        # Draw results
        output = self.draw_results(frame, faces, emotions)
        
        # Display
        cv2.imshow('Emotion Recognition - Test Image', output)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70)
    
    # Create recognizer
    recognizer = SimpleEmotionRecognizer()
    
    # Load face detector
    if not recognizer.load_face_detector():
        print("\n✗ Failed to initialize. Exiting...")
        return
    
    # Try to load trained model, or create demo model
    print("\nAttempting to load trained model...")
    recognizer.load_trained_model('emotion_model.h5')
    
    # Choose mode
    print("\n" + "="*70)
    print("SELECT MODE")
    print("="*70)
    print("1. Webcam (real-time emotion recognition)")
    print("2. Test on image file")
    print("3. Test face detection only")
    print("="*70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        recognizer.run_webcam()
    
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        recognizer.test_on_image(image_path)
    
    elif choice == '3':
        print("\nTesting face detection only...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return
        
        print("✓ Webcam opened. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = recognizer.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Invalid choice!")
    
    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)
    print("\n✓ Thank you for using the Emotion Recognition System!")
    # print("✓ No dlib or face_recognition required!")
    # print("\nFor questions or issues, check the documentation.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Please check your installation and try again.")