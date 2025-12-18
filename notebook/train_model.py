"""
Step 2: train_model.py
Complete Model Training Pipeline for FER2013

This script:
1. Loads and prepares FER2013 dataset
2. Builds CNN model architecture
3. Configures training with GPU optimization
4. Trains model for 2-4 hours
5. Saves best model and training history
6. Generates training reports

Usage:
    python train_model.py


Version: 1.0
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
CONFIG = {
    'dataset_path': 'data/fer2013',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'results_dir': 'results/training',
    'img_size': (48, 48),
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'num_classes': 7,
    'emotion_labels': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
}

# Create directories
for dir_path in [CONFIG['models_dir'], CONFIG['logs_dir'], CONFIG['results_dir']]:
    os.makedirs(dir_path, exist_ok=True)

print("="*70)
print("FER2013 MODEL TRAINING PIPELINE")
print("="*70 + "\n")


def check_gpu_availability():
    """Check and configure GPU"""
    print("Step 1: Checking GPU availability...")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠ Warning: {e}")
        
        # Set mixed precision for faster training
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision enabled (faster training)")
        except:
            print("⚠ Mixed precision not available")
        
        return True
    else:
        print("⚠ No GPU found - training will use CPU")
        print("  Expected training time: 8-12 hours on CPU")
        print("  Recommendation: Use Google Colab or cloud GPU")
        
        response = input("\nContinue with CPU training? (y/n): ").strip().lower()
        if response != 'y':
            print("Training cancelled. Use GPU for faster training.")
            sys.exit(0)
        
        return False


def build_emotion_model():
    """Build CNN architecture for emotion recognition"""
    print("\n" + "="*70)
    print("Step 2: Building model architecture...")
    print("="*70)
    
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
        Dense(CONFIG['num_classes'], activation='softmax')
    ])
    
    print("\n✓ Model architecture:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


def create_data_generators():
    """Create data generators with augmentation"""
    print("\n" + "="*70)
    print("Step 3: Creating data generators...")
    print("="*70)
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=CONFIG['validation_split']
    )
    
    # Test data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    print("\nLoading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'train'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print("\nLoading test data...")
    test_generator = test_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'test'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✓ Data loaded:")
    print(f"  Training samples: {train_generator.samples:,}")
    print(f"  Validation samples: {validation_generator.samples:,}")
    print(f"  Test samples: {test_generator.samples:,}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Steps per epoch: {train_generator.samples // CONFIG['batch_size']}")
    
    return train_generator, validation_generator, test_generator


def setup_callbacks(timestamp):
    """Configure training callbacks"""
    print("\n" + "="*70)
    print("Step 4: Setting up callbacks...")
    print("="*70)
    
    callbacks = []
    
    # ModelCheckpoint - save best model
    checkpoint_path = os.path.join(CONFIG['models_dir'], 'best_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    print(f"✓ ModelCheckpoint: {checkpoint_path}")
    
    # EarlyStopping - prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    print("✓ EarlyStopping: patience=10")
    
    # ReduceLROnPlateau - adaptive learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    print("✓ ReduceLROnPlateau: factor=0.5, patience=5")
    
    # CSVLogger - save training history
    csv_path = os.path.join(CONFIG['logs_dir'], f'training_log_{timestamp}.csv')
    csv_logger = CSVLogger(csv_path, append=False)
    callbacks.append(csv_logger)
    print(f"✓ CSVLogger: {csv_path}")
    
    # TensorBoard - visualization
    tensorboard_dir = os.path.join(CONFIG['logs_dir'], f'tensorboard_{timestamp}')
    tensorboard = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    print(f"✓ TensorBoard: {tensorboard_dir}")
    print(f"  View with: tensorboard --logdir={tensorboard_dir}")
    
    return callbacks


def train_model(model, train_gen, val_gen, callbacks):
    """Train the model"""
    print("\n" + "="*70)
    print("Step 5: Starting model training...")
    print("="*70)
    
    # Compile model
    optimizer = Adam(learning_rate=CONFIG['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✓ Model compiled")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']})")
    print(f"  Loss: categorical_crossentropy")
    print(f"  Metrics: accuracy")
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // CONFIG['batch_size']
    validation_steps = val_gen.samples // CONFIG['batch_size']
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    print(f"  Total training steps: {steps_per_epoch * CONFIG['epochs']:,}")
    
    # Estimate training time
    print(f"\nEstimated training time:")
    print(f"  On GPU: 2-4 hours")
    print(f"  On CPU: 8-12 hours")
    
    print("\n" + "-"*70)
    print("TRAINING STARTED")
    print("-"*70)
    print("Monitor progress below. Press Ctrl+C to stop training.")
    print("-"*70 + "\n")
    
    start_time = time.time()
    
    try:
        # Train model
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=CONFIG['epochs'],
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "-"*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("-"*70)
        print(f"Total training time: {training_time/3600:.2f} hours")
        
        return history, training_time
        
    except KeyboardInterrupt:
        print("\n\n✓ Training interrupted by user")
        training_time = time.time() - start_time
        print(f"Training time before interruption: {training_time/3600:.2f} hours")
        return None, training_time


def plot_training_history(history, timestamp):
    """Plot and save training history"""
    print("\n" + "="*70)
    print("Step 6: Generating training plots...")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(CONFIG['results_dir'], f'training_history_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved: {plot_path}")
    plt.close()
    
    # Print best metrics
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_acc = history.history['val_accuracy'][best_epoch]
    best_val_loss = history.history['val_loss'][best_epoch]
    
    print(f"\n✓ Best results:")
    print(f"  Epoch: {best_epoch + 1}/{len(history.history['accuracy'])}")
    print(f"  Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Validation Loss: {best_val_loss:.4f}")


def evaluate_model(model, test_gen):
    """Evaluate model on test set"""
    print("\n" + "="*70)
    print("Step 7: Evaluating model on test set...")
    print("="*70)
    
    print("\nRunning evaluation...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n✓ Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return test_loss, test_accuracy


def save_training_report(history, training_time, test_loss, test_accuracy, timestamp):
    """Save comprehensive training report"""
    print("\n" + "="*70)
    print("Step 8: Generating training report...")
    print("="*70)
    
    report_path = os.path.join(CONFIG['results_dir'], f'training_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Duration: {training_time/3600:.2f} hours\n\n")
        
        # Configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset: {CONFIG['dataset_path']}\n")
        f.write(f"Image Size: {CONFIG['img_size']}\n")
        f.write(f"Batch Size: {CONFIG['batch_size']}\n")
        f.write(f"Epochs: {CONFIG['epochs']}\n")
        f.write(f"Learning Rate: {CONFIG['learning_rate']}\n")
        f.write(f"Validation Split: {CONFIG['validation_split']}\n\n")
        
        # Model Architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write("4-Block CNN with Batch Normalization\n")
        f.write("- Block 1: Conv(32) -> Conv(64) -> MaxPool -> Dropout(0.25)\n")
        f.write("- Block 2: Conv(128) -> MaxPool -> Conv(128) -> MaxPool -> Dropout(0.25)\n")
        f.write("- FC: Dense(1024) -> Dropout(0.5) -> Dense(7)\n")
        f.write(f"Total Parameters: {history.model.count_params():,}\n\n")
        
        # Training Results
        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        
        best_epoch = np.argmax(history.history['val_accuracy'])
        f.write(f"Best Epoch: {best_epoch + 1}/{len(history.history['accuracy'])}\n")
        f.write(f"Best Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}\n")
        f.write(f"Best Validation Loss: {history.history['val_loss'][best_epoch]:.4f}\n\n")
        
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n\n")
        
        # Test Results
        f.write("TEST SET EVALUATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        
        # Performance Analysis
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        
        if test_accuracy >= 0.70:
            f.write("✓ EXCELLENT: Performance exceeds state-of-the-art!\n")
        elif test_accuracy >= 0.65:
            f.write("✓ GOOD: Performance matches state-of-the-art\n")
        elif test_accuracy >= 0.60:
            f.write("✓ ACCEPTABLE: Performance is reasonable\n")
        else:
            f.write("⚠ NEEDS IMPROVEMENT: Consider more training or tuning\n")
        
        f.write(f"\nComparison with FER2013 benchmarks:\n")
        f.write(f"- Human Baseline: ~65%\n")
        f.write(f"- Your Model: {test_accuracy*100:.2f}%\n")
        f.write(f"- State-of-the-art: ~73%\n\n")
        
        # Next Steps
        f.write("NEXT STEPS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Run 'python test_model.py' to perform detailed evaluation\n")
        f.write("2. Run 'python realtime_emotion_analytics.py' to test live\n")
        f.write("3. Generate confusion matrix and per-class metrics\n")
        f.write("4. Export results for thesis documentation\n")
    
    print(f"✓ Training report saved: {report_path}")
    
    # Save history as JSON
    history_path = os.path.join(CONFIG['results_dir'], f'training_history_{timestamp}.json')
    history_dict = {
        'config': CONFIG,
        'training_time_hours': training_time / 3600,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'test_results': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"✓ Training history saved: {history_path}")


def main():
    """Main training pipeline"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Check GPU
    has_gpu = check_gpu_availability()
    
    # Step 2: Build model
    model = build_emotion_model()
    
    # Step 3: Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Step 4: Setup callbacks
    callbacks = setup_callbacks(timestamp)
    
    # Step 5: Train model
    history, training_time = train_model(model, train_gen, val_gen, callbacks)
    
    if history is None:
        print("\nTraining was interrupted. Partial model may be saved.")
        return
    
    # Step 6: Plot training history
    plot_training_history(history, timestamp)
    
    # Step 7: Evaluate on test set
    # Load best model
    best_model_path = os.path.join(CONFIG['models_dir'], 'best_model.h5')
    best_model = keras.models.load_model(best_model_path)
    test_loss, test_accuracy = evaluate_model(best_model, test_gen)
    
    # Step 8: Generate report
    save_training_report(history, training_time, test_loss, test_accuracy, timestamp)
    
    # Save final model
    final_model_path = os.path.join(CONFIG['models_dir'], 'final_model.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Also save as emotion_model.h5 for easy use
    emotion_model_path = os.path.join(CONFIG['models_dir'], 'emotion_model.h5')
    best_model.save(emotion_model_path)
    print(f"✓ Model saved as: {emotion_model_path}")
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nTraining time: {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"\nFiles saved:")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - For use: {emotion_model_path}")
    print(f"  - Results: {CONFIG['results_dir']}/")
    print(f"\nNext step: Run 'python test_model.py' for detailed evaluation")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()