"""
Step 3: test_model.py
Comprehensive Model Testing and Evaluation

This script:
1. Loads trained model
2. Evaluates on test set
3. Generates confusion matrix
4. Calculates per-class metrics
5. Performs error analysis
6. Creates evaluation reports

Usage:
    python test_model.py

Version: 1.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
CONFIG = {
    'dataset_path': 'data/fer2013',
    'models_dir': 'models',
    'results_dir': 'results/evaluation',
    'img_size': (48, 48),
    'batch_size': 64,
    'emotion_labels': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
}

# Create results directory
os.makedirs(CONFIG['results_dir'], exist_ok=True)

print("="*70)
print("MODEL TESTING AND EVALUATION")
print("="*70 + "\n")


def load_model():
    """Load trained model"""
    print("Step 1: Loading trained model...")
    
    model_path = os.path.join(CONFIG['models_dir'], 'emotion_model.h5')
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("\nTrying alternative paths...")
        
        # Try best_model.h5
        alt_path = os.path.join(CONFIG['models_dir'], 'best_model.h5')
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"✓ Found model at: {model_path}")
        else:
            print("✗ No trained model found")
            print("Please run 'python train_model.py' first")
            return None
    
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def load_test_data():
    """Load test dataset"""
    print("\n" + "="*70)
    print("Step 2: Loading test data...")
    print("="*70)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(CONFIG['dataset_path'], 'test'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✓ Test data loaded")
    print(f"  Samples: {test_generator.samples:,}")
    print(f"  Batches: {len(test_generator)}")
    
    return test_generator


def evaluate_model(model, test_gen):
    """Evaluate model and get predictions"""
    print("\n" + "="*70)
    print("Step 3: Evaluating model...")
    print("="*70)
    
    print("\nRunning model evaluation...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n✓ Evaluation complete")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\nGenerating predictions...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    print(f"✓ Predictions generated")
    print(f"  Total predictions: {len(y_pred):,}")
    
    return test_loss, test_accuracy, y_true, y_pred, predictions


def plot_confusion_matrix(y_true, y_pred, timestamp):
    """Generate and save confusion matrix"""
    print("\n" + "="*70)
    print("Step 4: Generating confusion matrix...")
    print("="*70)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['emotion_labels'],
                yticklabels=CONFIG['emotion_labels'],
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CONFIG['emotion_labels'],
                yticklabels=CONFIG['emotion_labels'],
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    cm_path = os.path.join(CONFIG['results_dir'], f'confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    return cm


def generate_classification_report(y_true, y_pred, timestamp):
    """Generate detailed classification report"""
    print("\n" + "="*70)
    print("Step 5: Generating classification report...")
    print("="*70)
    
    # Get detailed metrics
    report = classification_report(
        y_true, y_pred,
        target_names=CONFIG['emotion_labels'],
        output_dict=True
    )
    
    # Print report
    print("\nPer-class metrics:")
    print(f"{'Emotion':12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-" * 60)
    
    for emotion in CONFIG['emotion_labels']:
        metrics = report[emotion]
        print(f"{emotion:12s} {metrics['precision']:10.4f} "
              f"{metrics['recall']:10.4f} {metrics['f1-score']:10.4f} "
              f"{int(metrics['support']):10d}")
    
    print("-" * 60)
    print(f"{'Accuracy':12s} {'':<10s} {'':<10s} {report['accuracy']:10.4f} "
          f"{int(report['macro avg']['support']):10d}")
    print(f"{'Macro Avg':12s} {report['macro avg']['precision']:10.4f} "
          f"{report['macro avg']['recall']:10.4f} "
          f"{report['macro avg']['f1-score']:10.4f} "
          f"{int(report['macro avg']['support']):10d}")
    print(f"{'Weighted Avg':12s} {report['weighted avg']['precision']:10.4f} "
          f"{report['weighted avg']['recall']:10.4f} "
          f"{report['weighted avg']['f1-score']:10.4f} "
          f"{int(report['weighted avg']['support']):10d}")
    
    # Save report
    report_path = os.path.join(CONFIG['results_dir'], f'classification_report_{timestamp}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n✓ Classification report saved: {report_path}")
    
    # Visualize per-class metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    emotions = CONFIG['emotion_labels']
    metrics_data = {
        'precision': [report[e]['precision'] for e in emotions],
        'recall': [report[e]['recall'] for e in emotions],
        'f1-score': [report[e]['f1-score'] for e in emotions]
    }
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        axes[idx].barh(emotions, values, color='steelblue', edgecolor='black')
        axes[idx].set_xlabel(metric_name.capitalize())
        axes[idx].set_title(f'{metric_name.capitalize()} per Emotion',
                          fontsize=12, fontweight='bold')
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(values):
            axes[idx].text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    metrics_path = os.path.join(CONFIG['results_dir'], f'per_class_metrics_{timestamp}.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics visualization saved: {metrics_path}")
    plt.close()
    
    return report


def analyze_errors(y_true, y_pred, predictions, cm, timestamp):
    """Analyze prediction errors"""
    print("\n" + "="*70)
    print("Step 6: Performing error analysis...")
    print("="*70)
    
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\nError statistics:")
    print(f"  Total samples: {len(y_true):,}")
    print(f"  Correct predictions: {len(y_true) - len(misclassified_idx):,}")
    print(f"  Incorrect predictions: {len(misclassified_idx):,}")
    print(f"  Error rate: {len(misclassified_idx) / len(y_true) * 100:.2f}%")
    
    # Most confused pairs
    print("\nMost confused emotion pairs:")
    confused_pairs = []
    for i in range(len(CONFIG['emotion_labels'])):
        for j in range(len(CONFIG['emotion_labels'])):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((
                    CONFIG['emotion_labels'][i],
                    CONFIG['emotion_labels'][j],
                    cm[i, j]
                ))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'True Label':12s} {'Predicted As':12s} {'Count':>8s}")
    print("-" * 40)
    for true_label, pred_label, count in confused_pairs[:10]:
        print(f"{true_label:12s} → {pred_label:12s} {int(count):8d}")
    
    # Confidence analysis for misclassifications
    misclass_confidences = predictions[misclassified_idx, y_pred[misclassified_idx]]
    correct_confidences = predictions[y_true == y_pred, y_pred[y_true == y_pred]]
    
    print(f"\nConfidence analysis:")
    print(f"  Mean confidence (correct): {np.mean(correct_confidences):.4f}")
    print(f"  Mean confidence (incorrect): {np.mean(misclass_confidences):.4f}")
    print(f"  Median confidence (correct): {np.median(correct_confidences):.4f}")
    print(f"  Median confidence (incorrect): {np.median(misclass_confidences):.4f}")
    
    # Plot confidence distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(correct_confidences, bins=50, alpha=0.7, label='Correct Predictions',
            color='green', edgecolor='black')
    ax.hist(misclass_confidences, bins=50, alpha=0.7, label='Incorrect Predictions',
            color='red', edgecolor='black')
    
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    conf_path = os.path.join(CONFIG['results_dir'], f'confidence_analysis_{timestamp}.png')
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confidence analysis saved: {conf_path}")
    plt.close()


def generate_final_report(test_loss, test_accuracy, report, timestamp):
    """Generate comprehensive evaluation report"""
    print("\n" + "="*70)
    print("Step 7: Generating final evaluation report...")
    print("="*70)
    
    report_path = os.path.join(CONFIG['results_dir'], f'evaluation_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {CONFIG['models_dir']}/emotion_model.h5\n\n")
        
        # Overall Performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        
        # Comparison with benchmarks
        f.write("BENCHMARK COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"Human Performance on FER2013: ~65%\n")
        f.write(f"Your Model: {test_accuracy*100:.2f}%\n")
        f.write(f"State-of-the-art: ~73%\n\n")
        
        if test_accuracy >= 0.70:
            f.write("✓ Performance Status: EXCELLENT (Above human baseline)\n\n")
        elif test_accuracy >= 0.65:
            f.write("✓ Performance Status: VERY GOOD (Matches human baseline)\n\n")
        elif test_accuracy >= 0.60:
            f.write("✓ Performance Status: GOOD (Acceptable performance)\n\n")
        else:
            f.write("⚠ Performance Status: NEEDS IMPROVEMENT\n\n")
        
        # Per-Class Performance
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Emotion':12s} {'Precision':>10s} {'Recall':>10s} "
               f"{'F1-Score':>10s} {'Support':>10s}\n")
        f.write("-" * 60 + "\n")
        
        for emotion in CONFIG['emotion_labels']:
            metrics = report[emotion]
            f.write(f"{emotion:12s} {metrics['precision']:10.4f} "
                   f"{metrics['recall']:10.4f} {metrics['f1-score']:10.4f} "
                   f"{int(metrics['support']):10d}\n")
        
        f.write("\n")
        
        # Best and Worst Classes
        f1_scores = [(e, report[e]['f1-score']) for e in CONFIG['emotion_labels']]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        f.write("PERFORMANCE HIGHLIGHTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best performing emotion: {f1_scores[0][0]} (F1: {f1_scores[0][1]:.4f})\n")
        f.write(f"Worst performing emotion: {f1_scores[-1][0]} (F1: {f1_scores[-1][1]:.4f})\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Model is ready for deployment if accuracy > 65%\n")
        f.write("2. Test with realtime_emotion_analytics.py for live performance\n")
        f.write("3. Monitor performance on your specific use case\n")
        f.write("4. Consider fine-tuning if accuracy < 60%\n")
        f.write("5. Document results for thesis with confusion matrix and metrics\n\n")
        
        # Files Generated
        f.write("GENERATED FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"- Confusion Matrix: confusion_matrix_{timestamp}.png\n")
        f.write(f"- Per-Class Metrics: per_class_metrics_{timestamp}.png\n")
        f.write(f"- Confidence Analysis: confidence_analysis_{timestamp}.png\n")
        f.write(f"- Classification Report: classification_report_{timestamp}.json\n")
        f.write(f"- This Report: evaluation_report_{timestamp}.txt\n")
    
    print(f"✓ Final report saved: {report_path}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Best Emotion: {f1_scores[0][0]} (F1: {f1_scores[0][1]:.4f})")
    print(f"Worst Emotion: {f1_scores[-1][0]} (F1: {f1_scores[-1][1]:.4f})")
    print(f"\nAll results saved in: {CONFIG['results_dir']}/")


def main():
    """Main evaluation pipeline"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Load model
    model = load_model()
    if model is None:
        return
    
    # Step 2: Load test data
    test_gen = load_test_data()
    
    # Step 3: Evaluate model
    test_loss, test_accuracy, y_true, y_pred, predictions = evaluate_model(model, test_gen)
    
    # Step 4: Confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, timestamp)
    
    # Step 5: Classification report
    report = generate_classification_report(y_true, y_pred, timestamp)
    
    # Step 6: Error analysis
    analyze_errors(y_true, y_pred, predictions, cm, timestamp)
    
    # Step 7: Final report
    generate_final_report(test_loss, test_accuracy, report, timestamp)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Results directory: {CONFIG['results_dir']}/")
    print(f"\nNext step: Run 'python realtime_emotion_analytics.py' to test live")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()