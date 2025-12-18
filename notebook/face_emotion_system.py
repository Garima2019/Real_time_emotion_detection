"""
face_emotion_system.py
Production-Ready Face Emotion Recognition System

Features:
- Face detection using OpenCV Haar Cascade
- Emotion recognition using TensorFlow CNN
- Real-time video processing
- Image and video file support
- Model training capabilities
- Data export functionality
- Performance optimization
- Error handling and logging

Version: 1.1 - Windows Console Fixed
License: MIT
"""

import cv2
import numpy as np
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import io

# Fix Windows console encoding BEFORE any logging
if sys.platform == 'win32':
    # Enable ANSI escape sequences on Windows 10+
    os.system('')
    # Force UTF-8 for console output
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, 
        Dropout, BatchNormalization, Activation
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Install with: pip install tensorflow")
    sys.exit(1)

# Create logs directory if it doesn't exist
os.makedirs('../logs', exist_ok=True)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/emotion_recognition.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Symbol constants for cross-platform compatibility
USE_ASCII_SYMBOLS = sys.platform == 'win32'  # Use ASCII on Windows, Unicode on others

if USE_ASCII_SYMBOLS:
    CHECKMARK = '[OK]'
    CROSSMARK = '[X]'
    WARNING = '[!]'
else:
    CHECKMARK = '✓'
    CROSSMARK = '✗'
    WARNING = '⚠'


class FaceEmotionRecognizer:
    """
    Complete Face Detection and Emotion Recognition System
    
    This class provides a comprehensive solution for detecting faces
    and recognizing emotions in real-time video streams, images, and video files.
    
    Attributes:
        emotion_labels (list): List of emotion class names
        emotion_colors (dict): Color mapping for each emotion
        face_detector: OpenCV face detector
        emotion_model: TensorFlow emotion recognition model
        confidence_threshold (float): Minimum confidence for face detection
    """
    
    # Class constants
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    IMG_SIZE = (48, 48)
    
    EMOTION_COLORS = {
        'Happy': (0, 255, 0),      # Green
        'Sad': (255, 0, 0),        # Blue
        'Angry': (0, 0, 255),      # Red
        'Surprise': (0, 255, 255), # Yellow
        'Fear': (128, 0, 128),     # Purple
        'Disgust': (0, 128, 128),  # Teal
        'Neutral': (128, 128, 128) # Gray
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the FaceEmotionRecognizer
        
        Args:
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
        """
        self.emotion_labels = self.EMOTION_LABELS
        self.emotion_colors = self.EMOTION_COLORS
        self.confidence_threshold = confidence_threshold
        self.face_detector = None
        self.emotion_model = None
        
        # Performance tracking
        self.fps_history = []
        self.detection_count = 0
        
        logger.info("FaceEmotionRecognizer initialized")
    
    # =========================================================================
    # FACE DETECTION METHODS
    # =========================================================================
    
    def load_face_detector_haar(self) -> bool:
        """
        Load OpenCV Haar Cascade face detector
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                logger.error("Failed to load Haar Cascade classifier")
                return False
            
            logger.info(f"{CHECKMARK} Haar Cascade face detector loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
            return False
    
    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using Haar Cascade classifier
        
        Args:
            frame: Input image/frame (BGR format)
            
        Returns:
            List of tuples (x1, y1, x2, y2, confidence)
        """
        if self.face_detector is None:
            logger.warning("Face detector not loaded")
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to (x1, y1, x2, y2, confidence) format
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append((x, y, x+w, y+h, 1.0))
            
            return face_list
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    # =========================================================================
    # EMOTION RECOGNITION METHODS
    # =========================================================================
    
    def build_emotion_model(self) -> Sequential:
        """
        Build CNN model for emotion recognition
        
        Architecture:
        - 4 Convolutional blocks with batch normalization
        - 2 Fully connected layers
        - Dropout for regularization
        - 7-class softmax output
        
        Returns:
            Compiled Keras Sequential model
        """
        logger.info("Building emotion recognition CNN model...")
        
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully Connected
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(len(self.emotion_labels), activation='softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"{CHECKMARK} Model built with {model.count_params():,} parameters")
        return model
    
    def load_emotion_model(self, model_path: str) -> bool:
        """
        Load pre-trained emotion recognition model
        
        Args:
            model_path: Path to saved model file (.h5)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.emotion_model = load_model(model_path)
            logger.info(f"{CHECKMARK} Emotion model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            return False
    
    def save_emotion_model(self, model_path: str) -> bool:
        """
        Save emotion recognition model
        
        Args:
            model_path: Path to save model file (.h5)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.emotion_model is None:
                logger.error("No model to save")
                return False
            
            self.emotion_model.save(model_path)
            logger.info(f"{CHECKMARK} Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def preprocess_face_for_emotion(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for emotion recognition
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Preprocessed image ready for model input (1, 48, 48, 1)
        """
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_img
            
            # Resize to model input size
            resized_face = cv2.resize(gray_face, self.IMG_SIZE)
            
            # Normalize pixel values to [0, 1]
            normalized_face = resized_face / 255.0
            
            # Reshape for model input: (1, height, width, channels)
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            
            return reshaped_face
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None
    
    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float, Optional[np.ndarray]]:
        """
        Predict emotion from face image
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Tuple of (emotion_label, confidence, all_probabilities)
        """
        if self.emotion_model is None:
            return "No Model", 0.0, None
        
        try:
            # Preprocess
            processed_face = self.preprocess_face_for_emotion(face_img)
            if processed_face is None:
                return "Error", 0.0, None
            
            # Predict
            predictions = self.emotion_model.predict(processed_face, verbose=0)
            
            # Get top emotion
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            emotion = self.emotion_labels[emotion_idx]
            
            return emotion, float(confidence), predictions[0]
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return "Error", 0.0, None
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def draw_results(
        self, 
        frame: np.ndarray, 
        faces: List[Tuple[int, int, int, int, float]], 
        emotions: List[Tuple[str, float, Optional[np.ndarray]]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and emotion labels on frame
        
        Args:
            frame: Input frame
            faces: List of detected faces
            emotions: List of predicted emotions
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        for i, (x1, y1, x2, y2, face_conf) in enumerate(faces):
            if i < len(emotions):
                emotion, emotion_conf, _ = emotions[i]
                
                # Get color for emotion
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                label = f"{emotion}: {emotion_conf*100:.1f}%"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    output,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return output
    
    # =========================================================================
    # PROCESSING METHODS
    # =========================================================================
    
    def process_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Complete pipeline: detect faces and recognize emotions
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, faces, emotions)
        """
        # Detect faces
        faces = self.detect_faces_haar(frame)
        
        # Recognize emotions for each face
        emotions = []
        for (x1, y1, x2, y2, _) in faces:
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                emotion, confidence, all_probs = self.predict_emotion(face_roi)
                emotions.append((emotion, confidence, all_probs))
        
        # Draw results
        output_frame = self.draw_results(frame, faces, emotions)
        
        # Update statistics
        self.detection_count += len(faces)
        
        return output_frame, faces, emotions
    
    def process_image(
        self, 
        image_path: str, 
        save_output: bool = True,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Process a single image file
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated output
            output_path: Path to save output (auto-generated if None)
            
        Returns:
            Tuple of (annotated_frame, faces, emotions)
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return None, [], []
        
        # Process
        output_frame, faces, emotions = self.process_frame(frame)
        
        # Print results
        logger.info(f"Detected {len(faces)} face(s)")
        for i, (emotion, confidence, _) in enumerate(emotions):
            logger.info(f"  Face {i+1}: {emotion} ({confidence*100:.2f}%)")
        
        # Save output
        if save_output:
            if output_path is None:
                path = Path(image_path)
                output_path = str(path.parent / f"{path.stem}_output{path.suffix}")
            
            cv2.imwrite(output_path, output_frame)
            logger.info(f"{CHECKMARK} Output saved to: {output_path}")
        
        return output_frame, faces, emotions
    
    def process_video(
        self, 
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        skip_frames: int = 1
    ) -> Dict[str, Any]:
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
            skip_frames: Process every Nth frame (for speed)
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_faces': 0,
            'emotions_detected': [],
            'processing_time': 0
        }
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % skip_frames == 0:
                    output_frame, faces, emotions = self.process_frame(frame)
                    stats['processed_frames'] += 1
                    stats['total_faces'] += len(faces)
                    
                    for emotion, conf, _ in emotions:
                        stats['emotions_detected'].append({
                            'frame': frame_count,
                            'emotion': emotion,
                            'confidence': float(conf)
                        })
                    
                    # Write to output
                    if writer:
                        writer.write(output_frame)
                    
                    # Display
                    if display:
                        cv2.imshow('Video Processing', output_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Processing interrupted by user")
                            break
                
                stats['total_frames'] = frame_count
                
                # Progress update
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}%")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        stats['processing_time'] = time.time() - start_time
        
        logger.info(f"{CHECKMARK} Video processing complete")
        logger.info(f"  Processed {stats['processed_frames']}/{stats['total_frames']} frames")
        logger.info(f"  Detected {stats['total_faces']} faces")
        logger.info(f"  Processing time: {stats['processing_time']:.2f}s")
        
        return stats
    
    def run_webcam(self, camera_index: int = 0) -> None:
        """
        Run real-time emotion recognition on webcam
        
        Args:
            camera_index: Camera device index (default: 0)
        """
        logger.info("Starting webcam emotion recognition...")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            print(f"\n{CROSSMARK} Error: Could not open webcam")
            print("Possible solutions:")
            print("  1. Check if camera is connected")
            print("  2. Check camera permissions")
            print("  3. Try different camera index (0, 1, 2)")
            return
        
        logger.info(f"{CHECKMARK} Webcam opened successfully")
        
        print("\n" + "="*70)
        print("REAL-TIME EMOTION RECOGNITION")
        print("="*70)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Start/Stop recording")
        print("  SPACE - Pause/Resume")
        print("="*70 + "\n")
        
        # State variables
        recording = False
        paused = False
        video_writer = None
        screenshot_count = 0
        fps_list = []
        
        # Create output directories
        os.makedirs('../results/images', exist_ok=True)
        os.makedirs('../results/videos', exist_ok=True)
        
        try:
            while True:
                if not paused:
                    start_time = time.time()
                    
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame")
                        break
                    
                    # Process frame
                    output_frame, faces, emotions = self.process_frame(frame)
                    
                    # Calculate FPS
                    elapsed = time.time() - start_time
                    fps = 1.0 / elapsed if elapsed > 0 else 0
                    fps_list.append(fps)
                    if len(fps_list) > 30:
                        fps_list.pop(0)
                    avg_fps = np.mean(fps_list)
                    
                    # Add info overlay
                    info_y = 30
                    cv2.putText(output_frame, f"Faces: {len(faces)}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    info_y += 30
                    cv2.putText(output_frame, f"FPS: {avg_fps:.1f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # Recording indicator
                    if recording:
                        cv2.circle(output_frame, 
                                 (output_frame.shape[1] - 30, 30), 
                                 10, (0, 0, 255), -1)
                        cv2.putText(output_frame, "REC", 
                                  (output_frame.shape[1] - 80, 35),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 0, 255), 2)
                    
                    # Display
                    cv2.imshow('Face Emotion Recognition', output_frame)
                    
                    # Record if enabled
                    if recording and video_writer is not None:
                        video_writer.write(output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                
                elif key == ord('s'):
                    filename = f"../results/images/screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, output_frame)
                    logger.info(f"{CHECKMARK} Screenshot saved: {filename}")
                    print(f"{CHECKMARK} Screenshot saved: {filename}")
                    screenshot_count += 1
                
                elif key == ord('r'):
                    if not recording:
                        # Start recording
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"../results/videos/recording_{timestamp}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        h, w = frame.shape[:2]
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                        recording = True
                        logger.info(f"{CHECKMARK} Recording started: {filename}")
                        print(f"{CHECKMARK} Recording started: {filename}")
                    else:
                        # Stop recording
                        recording = False
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
                        logger.info(f"{CHECKMARK} Recording stopped")
                        print(f"{CHECKMARK} Recording stopped")
                
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    logger.info(status)
                    print(f"{CHECKMARK} {status}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            
            logger.info(f"{CHECKMARK} Webcam session ended")
            logger.info(f"  Total detections: {self.detection_count}")
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def train_emotion_model(
        self,
        train_data_path: str,
        validation_data_path: str,
        epochs: int = 50,
        batch_size: int = 64,
        save_best_path: str = 'best_model.h5',
        save_final_path: str = 'final_model.h5'
    ) -> Dict[str, Any]:
        """
        Train emotion recognition model on dataset
        
        Args:
            train_data_path: Path to training data directory
            validation_data_path: Path to validation data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_best_path: Path to save best model
            save_final_path: Path to save final model
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")
        logger.info(f"  Training data: {train_data_path}")
        logger.info(f"  Validation data: {validation_data_path}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_data_path,
            target_size=self.IMG_SIZE,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            validation_data_path,
            target_size=self.IMG_SIZE,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        logger.info(f"  Training samples: {train_generator.samples}")
        logger.info(f"  Validation samples: {validation_generator.samples}")
        
        # Build model
        self.emotion_model = self.build_emotion_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                save_best_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Training started...")
        history = self.emotion_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.emotion_model.save(save_final_path)
        
        logger.info(f"{CHECKMARK} Training completed!")
        logger.info(f"  Best model saved: {save_best_path}")
        logger.info(f"  Final model saved: {save_final_path}")
        
        return history.history


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for command-line usage"""
    
    print("\n" + "="*70)
    print("FACE EMOTION RECOGNITION SYSTEM")
    print("Production Version 1.1 - Windows Compatible")
    print("="*70 + "\n")
    
    # Initialize recognizer
    recognizer = FaceEmotionRecognizer()
    
    # Load face detector
    print("Loading face detector...")
    if not recognizer.load_face_detector_haar():
        print(f"{CROSSMARK} Failed to load face detector. Exiting...")
        return
    
    # Load or create emotion model
    print("\nLoading emotion model...")
    model_loaded = recognizer.load_emotion_model('emotion_model.h5')
    
    if not model_loaded:
        print(f"{WARNING} Pre-trained model not found.")
        print("Creating demo model for testing...")
        recognizer.emotion_model = recognizer.build_emotion_model()
        print(f"{WARNING} Note: This model is untrained. Train it for real predictions.")
    
    # Main menu
    while True:
        print("\n" + "="*70)
        print("SELECT MODE")
        print("="*70)
        print("1. Webcam (real-time)")
        print("2. Process image")
        print("3. Process video")
        print("4. Train model")
        print("5. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            recognizer.run_webcam()
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            recognizer.process_image(image_path, save_output=True)
        
        elif choice == '3':
            video_path = input("Enter video path: ").strip()
            output_path = input("Save output video? (y/n): ").strip().lower()
            if output_path == 'y':
                output_path = input("Enter output path: ").strip()
            else:
                output_path = None
            recognizer.process_video(video_path, output_path=output_path)
        
        elif choice == '4':
            print("\nTrain Emotion Model")
            train_path = input("Training data path: ").strip()
            val_path = input("Validation data path: ").strip()
            print("\nEpochs set to 5 due to dataset size constraints.")
            epochs = int(input("Number of epochs (default 50): ").strip() or "50")
            recognizer.train_emotion_model(train_path, val_path, epochs=epochs)
        
        elif choice == '5':
            print(f"\n{CHECKMARK} Thank you for using Face Emotion Recognition System!")
            break
        
        else:
            print(f"{CROSSMARK} Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{CHECKMARK} Interrupted by user. Exiting gracefully...")
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n{CROSSMARK} Unexpected error: {e}")
        print("Check emotion_recognition.log for details.")
