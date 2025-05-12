import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
from collections import deque
import math

class RepetitiveBehaviorDetector:
    def __init__(self, window_size: int = 30, threshold: float = 0.3):
        """
        Initialize the repetitive behavior detector.
        
        Args:
            window_size: Number of frames to analyze for repetitive patterns
            threshold: Threshold for considering a movement as repetitive
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Parameters
        self.window_size = window_size
        self.threshold = threshold
        
        # Movement history
        self.hand_movements = deque(maxlen=window_size)
        self.body_movements = deque(maxlen=window_size)
        
        # Keypoint indices for different body parts
        self.LEFT_HAND = [15, 17, 19, 21]  # Left wrist, elbow, shoulder
        self.RIGHT_HAND = [16, 18, 20, 22]  # Right wrist, elbow, shoulder
        self.BODY = [11, 12, 23, 24]  # Shoulders and hips
        
        # Movement patterns to detect
        self.patterns = {
            'hand_flapping': self._detect_hand_flapping,
            'rocking': self._detect_rocking,
            'spinning': self._detect_spinning
        }
    
    def _calculate_movement(self, landmarks, indices: List[int]) -> np.ndarray:
        """Calculate movement vector for a set of body landmarks."""
        if not landmarks:
            return np.zeros(2)
        
        # Get current positions
        current_pos = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                             for idx in indices], axis=0)
        
        # Calculate movement from previous frame
        if self.hand_movements:
            prev_pos = self.hand_movements[-1]
            movement = current_pos - prev_pos
        else:
            movement = np.zeros(2)
        
        return movement
    
    def _detect_hand_flapping(self, movements: List[np.ndarray]) -> float:
        """Detect hand flapping pattern."""
        if len(movements) < 3:
            return 0.0
        
        # Calculate movement frequencies using FFT
        movements_x = [m[0] for m in movements]
        movements_y = [m[1] for m in movements]
        
        # Simple frequency analysis
        x_variance = np.var(movements_x)
        y_variance = np.var(movements_y)
        
        # Hand flapping typically shows high vertical movement
        return min(1.0, (y_variance / (x_variance + 1e-6)) * 2)
    
    def _detect_rocking(self, movements: List[np.ndarray]) -> float:
        """Detect rocking pattern."""
        if len(movements) < 3:
            return 0.0
        
        # Calculate movement frequencies
        movements_x = [m[0] for m in movements]
        
        # Rocking typically shows regular back-and-forth movement
        zero_crossings = sum(1 for i in range(len(movements_x)-1) 
                           if movements_x[i] * movements_x[i+1] < 0)
        
        return min(1.0, zero_crossings / (len(movements) / 2))
    
    def _detect_spinning(self, movements: List[np.ndarray]) -> float:
        """Detect spinning pattern."""
        if len(movements) < 3:
            return 0.0
        
        # Calculate angular movement
        angles = []
        for i in range(len(movements)-1):
            m1 = movements[i]
            m2 = movements[i+1]
            angle = math.atan2(m2[1], m2[0]) - math.atan2(m1[1], m1[0])
            angles.append(angle)
        
        # Spinning shows consistent angular movement
        angle_variance = np.var(angles)
        return min(1.0, angle_variance * 5)
    
    def detect_behaviors(self, frame: np.ndarray) -> Tuple[dict, float, Optional[np.ndarray]]:
        """
        Detect repetitive behaviors in the given frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple containing:
            - Dictionary of detected behaviors and their scores
            - Overall repetitive behavior score
            - Annotated frame
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return {}, 0.0, frame
        
        # Calculate movements
        hand_movement = self._calculate_movement(results.pose_landmarks, 
                                               self.LEFT_HAND + self.RIGHT_HAND)
        body_movement = self._calculate_movement(results.pose_landmarks, self.BODY)
        
        # Update movement history
        self.hand_movements.append(hand_movement)
        self.body_movements.append(body_movement)
        
        # Detect patterns
        behavior_scores = {}
        for pattern_name, detector in self.patterns.items():
            if pattern_name in ['hand_flapping']:
                score = detector(list(self.hand_movements))
            else:
                score = detector(list(self.body_movements))
            behavior_scores[pattern_name] = score
        
        # Calculate overall score
        overall_score = max(behavior_scores.values()) if behavior_scores else 0.0
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Add behavior labels
        y_offset = 30
        for behavior, score in behavior_scores.items():
            if score > self.threshold:
                label = f"{behavior}: {score:.2f}"
                cv2.putText(annotated_frame, label, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
        
        return behavior_scores, overall_score, annotated_frame
    
    def reset(self):
        """Reset the detector's state."""
        self.hand_movements.clear()
        self.body_movements.clear()
    
    def __del__(self):
        """Clean up resources."""
        self.pose.close() 