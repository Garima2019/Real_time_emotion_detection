"""
realtime_emotion_analytics.py
Production-Ready Real-Time Emotion Analytics System

Features:
- Live emotion tracking and statistics
- Temporal trend analysis
- Emotion distribution calculations
- Session analytics
- Data export (CSV, JSON, visualizations)
- Performance metrics
- Multi-face tracking

Version: 1.0 - Production Ready
License: MIT
"""

import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional, Any

# Import matplotlib for report generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Report generation disabled.")

# Import pandas for data export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. CSV export disabled.")

# Import from face_emotion_system
try:
    from face_emotion_system import FaceEmotionRecognizer
except ImportError:
    logging.error("face_emotion_system.py not found in same directory")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmotionAnalytics(FaceEmotionRecognizer):
    """
    Real-time emotion analytics with comprehensive statistics tracking
    
    Extends FaceEmotionRecognizer with:
    - Real-time statistics calculation
    - Emotion distribution tracking
    - Temporal trend analysis
    - Data export capabilities
    - Visualization generation
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize EmotionAnalytics
        
        Args:
            window_size: Size of sliding window in seconds for trend analysis
        """
        super().__init__()
        
        self.window_size = window_size
        self.session_start = None
        
        # Analytics storage
        self.emotion_history = deque(maxlen=window_size * 30)  # 30 FPS assumption
        self.emotion_timeline = []
        self.face_timeline = []
        
        # Real-time statistics
        self.current_stats = {
            'session_start_time': None,
            'session_duration': 0.0,
            'total_frames': 0,
            'frames_with_faces': 0,
            'total_faces_detected': 0,
            'unique_face_positions': set(),
            'emotion_counts': Counter(),
            'emotion_confidences': {emotion: [] for emotion in self.EMOTION_LABELS},
            'dominant_emotion': None,
            'average_confidence': 0.0,
            'emotion_changes': 0,
            'last_emotion': None
        }
        
        logger.info("EmotionAnalytics initialized")
    
    # =========================================================================
    # STATISTICS METHODS
    # =========================================================================
    
    def start_session(self) -> None:
        """Start a new analytics session"""
        self.session_start = time.time()
        self.current_stats['session_start_time'] = datetime.now().isoformat()
        self.reset_statistics()
        logger.info("Analytics session started")
    
    def reset_statistics(self) -> None:
        """Reset all statistics while keeping session start time"""
        session_start = self.current_stats['session_start_time']
        
        self.emotion_history.clear()
        self.emotion_timeline.clear()
        self.face_timeline.clear()
        
        self.current_stats = {
            'session_start_time': session_start,
            'session_duration': 0.0,
            'total_frames': 0,
            'frames_with_faces': 0,
            'total_faces_detected': 0,
            'unique_face_positions': set(),
            'emotion_counts': Counter(),
            'emotion_confidences': {emotion: [] for emotion in self.EMOTION_LABELS},
            'dominant_emotion': None,
            'average_confidence': 0.0,
            'emotion_changes': 0,
            'last_emotion': None
        }
        
        logger.info("Statistics reset")
    
    def update_statistics(
        self, 
        faces: List[Tuple], 
        emotions: List[Tuple]
    ) -> None:
        """
        Update real-time statistics with new detections
        
        Args:
            faces: List of detected faces
            emotions: List of predicted emotions
        """
        self.current_stats['total_frames'] += 1
        
        if self.session_start:
            self.current_stats['session_duration'] = time.time() - self.session_start
        
        # Update face statistics
        if faces:
            self.current_stats['frames_with_faces'] += 1
            self.current_stats['total_faces_detected'] += len(faces)
            
            # Track unique face positions (rough approximation)
            for (x1, y1, x2, y2, _) in faces:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Round to grid for approximate position tracking
                grid_pos = (center[0] // 50, center[1] // 50)
                self.current_stats['unique_face_positions'].add(grid_pos)
        
        # Update emotion statistics
        timestamp = time.time() - self.session_start if self.session_start else 0
        
        for face_idx, (emotion, confidence, probs) in enumerate(emotions):
            # Update emotion counts
            self.current_stats['emotion_counts'][emotion] += 1
            
            # Track emotion changes
            if self.current_stats['last_emotion'] and \
               self.current_stats['last_emotion'] != emotion:
                self.current_stats['emotion_changes'] += 1
            self.current_stats['last_emotion'] = emotion
            
            # Update confidence tracking
            self.current_stats['emotion_confidences'][emotion].append(confidence)
            
            # Store in history (sliding window)
            self.emotion_history.append({
                'timestamp': timestamp,
                'frame': self.current_stats['total_frames'],
                'face_id': face_idx,
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probs
            })
            
            # Store in permanent timeline
            self.emotion_timeline.append({
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'frame': self.current_stats['total_frames'],
                'face_id': face_idx,
                'emotion': emotion,
                'confidence': float(confidence)
            })
        
        # Store face information
        if faces:
            self.face_timeline.append({
                'timestamp': timestamp,
                'frame': self.current_stats['total_frames'],
                'face_count': len(faces)
            })
        
        # Calculate dominant emotion
        if self.current_stats['emotion_counts']:
            self.current_stats['dominant_emotion'] = \
                self.current_stats['emotion_counts'].most_common(1)[0][0]
        
        # Calculate average confidence
        all_confidences = []
        for conf_list in self.current_stats['emotion_confidences'].values():
            all_confidences.extend(conf_list)
        if all_confidences:
            self.current_stats['average_confidence'] = float(np.mean(all_confidences))
    
    def get_emotion_distribution(self) -> Dict[str, float]:
        """
        Get current emotion distribution as percentages
        
        Returns:
            Dictionary mapping emotions to percentages
        """
        total = sum(self.current_stats['emotion_counts'].values())
        if total == 0:
            return {emotion: 0.0 for emotion in self.EMOTION_LABELS}
        
        distribution = {}
        for emotion in self.EMOTION_LABELS:
            count = self.current_stats['emotion_counts'][emotion]
            distribution[emotion] = (count / total) * 100
        
        return distribution
    
    def get_recent_trend(self, seconds: int = 10) -> Counter:
        """
        Get emotion trend for recent time window
        
        Args:
            seconds: Size of time window in seconds
            
        Returns:
            Counter object with emotion frequencies
        """
        if not self.session_start:
            return Counter()
        
        cutoff_time = time.time() - self.session_start - seconds
        recent_emotions = [
            entry['emotion'] for entry in self.emotion_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        return Counter(recent_emotions)
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics summary
        
        Returns:
            Dictionary with all statistics
        """
        distribution = self.get_emotion_distribution()
        recent_trend = self.get_recent_trend(10)
        
        # Calculate additional metrics
        detection_rate = 0.0
        if self.current_stats['total_frames'] > 0:
            detection_rate = (self.current_stats['frames_with_faces'] / 
                            self.current_stats['total_frames']) * 100
        
        avg_faces_per_frame = 0.0
        if self.current_stats['frames_with_faces'] > 0:
            avg_faces_per_frame = (self.current_stats['total_faces_detected'] / 
                                  self.current_stats['frames_with_faces'])
        
        return {
            'session': {
                'start_time': self.current_stats['session_start_time'],
                'duration_seconds': self.current_stats['session_duration'],
                'duration_formatted': self._format_duration(
                    self.current_stats['session_duration']
                )
            },
            'frames': {
                'total': self.current_stats['total_frames'],
                'with_faces': self.current_stats['frames_with_faces'],
                'detection_rate': detection_rate
            },
            'faces': {
                'total_detected': self.current_stats['total_faces_detected'],
                'average_per_frame': avg_faces_per_frame,
                'unique_positions': len(self.current_stats['unique_face_positions'])
            },
            'emotions': {
                'total_detections': sum(self.current_stats['emotion_counts'].values()),
                'dominant': self.current_stats['dominant_emotion'],
                'distribution': distribution,
                'counts': dict(self.current_stats['emotion_counts']),
                'changes': self.current_stats['emotion_changes']
            },
            'confidence': {
                'average': self.current_stats['average_confidence'],
                'per_emotion': {
                    emotion: float(np.mean(confs)) if confs else 0.0
                    for emotion, confs in 
                    self.current_stats['emotion_confidences'].items()
                }
            },
            'recent_trend': dict(recent_trend)
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def draw_analytics_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw comprehensive analytics overlay on frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with analytics overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Configuration
        panel_width = 200
        panel_x = 10
        panel_y = 10
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Semi-transparent background
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + 450),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = panel_y + 25
        
        # Title
        cv2.putText(frame, "EMOTION ANALYTICS", (panel_x + 10, y_pos),
                   font, 0.7, (255, 255, 255), 2)
        y_pos += 35
        
        # Session info
        duration_str = self._format_duration(self.current_stats['session_duration'])
        cv2.putText(frame, f"Session: {duration_str}", (panel_x + 10, y_pos),
                   font, font_scale, (200, 200, 200), thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Frames: {self.current_stats['total_frames']}", 
                   (panel_x + 10, y_pos), font, font_scale, (200, 200, 200), thickness)
        y_pos += line_height
        
        cv2.putText(frame, f"Faces: {self.current_stats['total_faces_detected']}", 
                   (panel_x + 10, y_pos), font, font_scale, (200, 200, 200), thickness)
        y_pos += line_height
        
        # FPS
        if len(self.fps_history) > 0:
            fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, y_pos),
                       font, font_scale, (200, 200, 200), thickness)
        y_pos += 35
        
        # Dominant emotion
        if self.current_stats['dominant_emotion']:
            dom_emotion = self.current_stats['dominant_emotion']
            cv2.putText(frame, "DOMINANT EMOTION:", (panel_x + 10, y_pos),
                       font, font_scale, (255, 255, 0), thickness)
            y_pos += line_height
            
            color = self.emotion_colors.get(dom_emotion, (255, 255, 255))
            cv2.putText(frame, dom_emotion, (panel_x + 10, y_pos),
                       font, 0.8, color, 2)
        y_pos += 35
        
        # Emotion distribution
        cv2.putText(frame, "DISTRIBUTION:", (panel_x + 10, y_pos),
                   font, font_scale, (255, 255, 0), thickness)
        y_pos += line_height
        
        distribution = self.get_emotion_distribution()
        sorted_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, percentage in sorted_emotions[:7]:  # Show all emotions
            # Emotion label
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.putText(frame, emotion[:4], (panel_x + 10, y_pos),
                       font, 0.45, color, thickness)
            
            # Percentage bar
            bar_x = panel_x + 70
            bar_y = y_pos - 12
            bar_max_width = panel_width - 140
            bar_width = int(bar_max_width * (percentage / 100))
            bar_height = 15
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_max_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Filled bar
            if bar_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y),
                             (bar_x + bar_width, bar_y + bar_height),
                             color, -1)
            
            # Percentage text
            cv2.putText(frame, f"{percentage:.1f}%",
                       (bar_x + bar_max_width + 5, y_pos),
                       font, 0.4, (255, 255, 255), thickness)
            
            y_pos += line_height
        
        # Recent trend panel (top right)
        trend_width = 250
        trend_x = w - trend_width - 10
        trend_y = 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (trend_x, trend_y),
                     (w - 10, trend_y + 180),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = trend_y + 25
        cv2.putText(frame, "LAST 10 SECONDS", (trend_x + 10, y_pos),
                   font, 0.6, (255, 255, 0), 2)
        y_pos += 30
        
        recent_trend = self.get_recent_trend(10)
        if recent_trend:
            total_recent = sum(recent_trend.values())
            for emotion, count in recent_trend.most_common(5):
                percentage = (count / total_recent) * 100
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                text = f"{emotion}: {percentage:.0f}%"
                cv2.putText(frame, text, (trend_x + 10, y_pos),
                           font, 0.5, color, thickness)
                y_pos += line_height
        
        return frame
    
    # =========================================================================
    # REAL-TIME PROCESSING
    # =========================================================================
    
    def process_frame_with_analytics(self, frame: np.ndarray) -> Tuple[np.ndarray, List, List]:
        """
        Process frame and update analytics
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, faces, emotions)
        """
        start_time = time.time()
        
        # Detect and recognize
        faces = self.detect_faces_haar(frame)
        
        emotions = []
        for (x1, y1, x2, y2, _) in faces:
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                emotion, confidence, probs = self.predict_emotion(face_roi)
                emotions.append((emotion, confidence, probs))
        
        # Update statistics
        self.update_statistics(faces, emotions)
        
        # Draw results
        output = self.draw_results(frame, faces, emotions)
        
        # Draw analytics overlay
        output = self.draw_analytics_overlay(output)
        
        # Track FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        return output, faces, emotions
    
    def run_webcam_analytics(self, camera_index: int = 0) -> None:
        """
        Run real-time emotion analytics on webcam
        
        Args:
            camera_index: Camera device index
        """
        logger.info("Starting webcam analytics...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            print("\n✗ Error: Could not open webcam")
            return
        
        self.start_session()
        
        print("\n" + "="*70)
        print("REAL-TIME EMOTION ANALYTICS")
        print("="*70)
        print("Controls:")
        print("  'q' - Quit and save statistics")
        print("  's' - Save screenshot")
        print("  'r' - Reset statistics")
        print("  'e' - Export data")
        print("  SPACE - Pause/Resume")
        print("="*70 + "\n")
        
        paused = False
        screenshot_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process with analytics
                    output, faces, emotions = self.process_frame_with_analytics(frame)
                    
                    # Display
                    cv2.imshow('Emotion Analytics', output)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"../results/images/analytics_screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, output)
                    print(f"✓ Screenshot saved: {filename}")
                    screenshot_count += 1
                elif key == ord('r'):
                    self.reset_statistics()
                    print("✓ Statistics reset")
                elif key == ord('e'):
                    self.export_analytics()
                    print("✓ Data exported")
                elif key == ord(' '):
                    paused = not paused
                    print(f"✓ {'PAUSED' if paused else 'RESUMED'}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final export
            self.print_final_statistics()
            self.export_analytics()
    
    # =========================================================================
    # DATA EXPORT METHODS
    # =========================================================================
    
    def print_final_statistics(self) -> None:
        """Print comprehensive final statistics to console"""
        stats = self.get_statistics_summary()
        
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Duration: {stats['session']['duration_formatted']}")
        print(f"Total Frames: {stats['frames']['total']}")
        print(f"Frames with Faces: {stats['frames']['with_faces']}")
        print(f"Detection Rate: {stats['frames']['detection_rate']:.1f}%")
        print(f"Total Faces: {stats['faces']['total_detected']}")
        print(f"Avg Faces/Frame: {stats['faces']['average_per_frame']:.2f}")
        
        if stats['emotions']['dominant']:
            print(f"\nDominant Emotion: {stats['emotions']['dominant']}")
        
        print(f"Total Emotion Detections: {stats['emotions']['total_detections']}")
        print(f"Emotion Changes: {stats['emotions']['changes']}")
        print(f"Average Confidence: {stats['confidence']['average']*100:.2f}%")
        
        print("\nEMOTION DISTRIBUTION:")
        for emotion, percentage in sorted(stats['emotions']['distribution'].items(),
                                         key=lambda x: x[1], reverse=True):
            bar = '█' * int(percentage / 5)
            print(f"  {emotion:10s}: {bar:20s} {percentage:5.2f}%")
        
        print("="*70 + "\n")
    
    def export_analytics(self, output_dir: str = ".") -> None:
        """
        Export analytics data to files
        
        Args:
            output_dir: Directory to save export files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export timeline to CSV
        if PANDAS_AVAILABLE and self.emotion_timeline:
            try:
                df = pd.DataFrame(self.emotion_timeline)
                csv_file = output_path / f"../results/csv/emotion_timeline_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"✓ Timeline exported: {csv_file}")
                print(f"✓ Timeline CSV: {csv_file}")
            except Exception as e:
                logger.error(f"Error exporting CSV: {e}")
        
        # Export statistics to JSON
        try:
            stats = self.get_statistics_summary()
            json_file = output_path / f"../results/json/emotion_statistics_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(stats, f, indent=4)
            logger.info(f"✓ Statistics exported: {json_file}")
            print(f"✓ Statistics JSON: {json_file}")
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
        
        # Generate visualization report
        if MATPLOTLIB_AVAILABLE:
            try:
                plot_file = output_path / f"../results/images/emotion_report_{timestamp}.png"
                self.generate_report_plots(str(plot_file))
                logger.info(f"✓ Report generated: {plot_file}")
                print(f"✓ Report PNG: {plot_file}")
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
    
    def generate_report_plots(self, output_file: str) -> None:
        """
        Generate comprehensive visualization report
        
        Args:
            output_file: Path to save plot image
        """
        if not self.emotion_timeline or not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Emotion Analytics Report', fontsize=16, fontweight='bold')
        
        # 1. Emotion distribution pie chart
        distribution = self.get_emotion_distribution()
        non_zero = {k: v for k, v in distribution.items() if v > 0}
        
        if non_zero:
            axes[0, 0].pie(
                non_zero.values(),
                labels=non_zero.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            axes[0, 0].set_title('Overall Emotion Distribution')
        
        # 2. Emotion timeline
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(self.emotion_timeline)
            for emotion in self.EMOTION_LABELS:
                emotion_data = df[df['emotion'] == emotion]
                if not emotion_data.empty:
                    axes[0, 1].scatter(
                        emotion_data['timestamp'],
                        [emotion] * len(emotion_data),
                        label=emotion, alpha=0.6, s=20
                    )
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Emotion')
            axes[0, 1].set_title('Emotion Timeline')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence distribution
        all_confidences = []
        for conf_list in self.current_stats['emotion_confidences'].values():
            all_confidences.extend(conf_list)
        
        if all_confidences:
            axes[1, 0].hist(all_confidences, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Confidence Distribution')
            mean_conf = np.mean(all_confidences)
            axes[1, 0].axvline(mean_conf, color='red', linestyle='--',
                             label=f'Mean: {mean_conf:.2f}')
            axes[1, 0].legend()
        
        # 4. Emotion counts bar chart
        counts = self.current_stats['emotion_counts']
        if counts:
            emotions = list(counts.keys())
            values = list(counts.values())
            axes[1, 1].bar(emotions, values, color='steelblue', edgecolor='black')
            axes[1, 1].set_xlabel('Emotion')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Emotion Detection Counts')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for command-line usage"""
    
    print("\n" + "="*70)
    print("REAL-TIME EMOTION ANALYTICS SYSTEM")
    print("Production Version 1.0")
    print("="*70 + "\n")
    
    # Initialize analytics
    analytics = EmotionAnalytics(window_size=30)
    
    # Load models
    print("Loading face detector...")
    if not analytics.load_face_detector_haar():
        print("✗ Failed to load face detector. Exiting...")
        return
    
    print("\nLoading emotion model...")
    model_loaded = analytics.load_emotion_model('emotion_model.h5')
    
    if not model_loaded:
        print("⚠ Pre-trained model not found.")
        print("Creating demo model for testing...")
        analytics.emotion_model = analytics.build_emotion_model()
        print("⚠ Note: This model is untrained. Train it for real predictions.")
    
    # Run analytics
    print("\nStarting webcam analytics...")
    analytics.run_webcam_analytics()
    
    print("\n✓ Session complete!")
    print("Check exported files for detailed analytics.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user. Exiting gracefully...")
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n✗ Unexpected error: {e}")
        print("Check emotion_analytics.log for details.")