import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from collections import deque
import math

class SocialReciprocityDetector:
    def __init__(self, window_size: int = 30, threshold: float = 0.3):
        """
        Initialize the social reciprocity detector.
        
        Args:
            window_size: Number of frames to analyze for social interaction patterns
            threshold: Threshold for considering an interaction as significant
        """
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize pose and face mesh detectors
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,  # Detect up to 2 faces for interaction analysis
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Parameters
        self.window_size = window_size
        self.threshold = threshold
        
        # Interaction history
        self.interaction_history = deque(maxlen=window_size)
        self.attention_history = deque(maxlen=window_size)
        
        # Keypoint indices
        self.FACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Face landmarks
        self.BODY = [11, 12, 23, 24]  # Shoulders and hips
        
        # Interaction patterns to detect
        self.patterns = {
            'joint_attention': self._detect_joint_attention,
            'response_to_name': self._detect_name_response,
            'gesture_use': self._detect_gestures,
            'social_proximity': self._detect_social_proximity
        }
    
    def _calculate_attention_direction(self, landmarks) -> np.ndarray:
        """Calculate the direction of attention based on face orientation."""
        if not landmarks:
            return np.zeros(2)
        
        # Get face landmarks
        face_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                              for idx in self.FACE])
        
        # Calculate face orientation
        face_center = np.mean(face_points, axis=0)
        face_direction = face_points[0] - face_center  # Using nose as reference
        
        return face_direction
    
    def _detect_joint_attention(self, frame: np.ndarray, landmarks_list: List) -> float:
        """Detect joint attention between multiple people."""
        if len(landmarks_list) < 2:
            return 0.0
        
        # Get attention directions for each person
        attention_directions = []
        for landmarks in landmarks_list:
            direction = self._calculate_attention_direction(landmarks)
            attention_directions.append(direction)
        
        # Calculate similarity of attention directions
        similarity = np.dot(attention_directions[0], attention_directions[1])
        similarity = (similarity + 1) / 2  # Normalize to [0, 1]
        
        return similarity
    
    def _detect_name_response(self, frame: np.ndarray, landmarks_list: List) -> float:
        """Detect response to name calling."""
        if not landmarks_list:
            return 0.0
        
        # This is a simplified version - in reality, you would need audio input
        # or a more sophisticated visual analysis of head movement patterns
        
        # For now, we'll use head movement as a proxy
        if len(self.attention_history) < 2:
            return 0.0
        
        # Calculate head movement
        current_attention = self._calculate_attention_direction(landmarks_list[0])
        prev_attention = self.attention_history[-1]
        
        movement = np.linalg.norm(current_attention - prev_attention)
        return min(1.0, movement * 5)
    
    def _detect_gestures(self, frame: np.ndarray, landmarks_list: List) -> float:
        """Detect use of gestures for communication."""
        if not landmarks_list:
            return 0.0
        
        # Get hand positions
        landmarks = landmarks_list[0]
        left_hand = np.array([landmarks.landmark[15].x, landmarks.landmark[15].y])
        right_hand = np.array([landmarks.landmark[16].x, landmarks.landmark[16].y])
        
        # Calculate hand movement
        if len(self.interaction_history) > 0:
            prev_hands = self.interaction_history[-1]
            left_movement = np.linalg.norm(left_hand - prev_hands[0])
            right_movement = np.linalg.norm(right_hand - prev_hands[1])
            
            # Gestures typically involve deliberate hand movements
            movement_score = (left_movement + right_movement) / 2
            return min(1.0, movement_score * 3)
        
        return 0.0
    
    def _detect_social_proximity(self, frame: np.ndarray, landmarks_list: List) -> float:
        """Detect social proximity and interaction distance."""
        if len(landmarks_list) < 2:
            return 0.0
        
        # Calculate distance between people
        person1_center = np.mean([(landmarks_list[0].landmark[idx].x, 
                                 landmarks_list[0].landmark[idx].y) 
                                for idx in self.BODY], axis=0)
        
        person2_center = np.mean([(landmarks_list[1].landmark[idx].x, 
                                 landmarks_list[1].landmark[idx].y) 
                                for idx in self.BODY], axis=0)
        
        distance = np.linalg.norm(person1_center - person2_center)
        
        # Convert distance to proximity score (closer = higher score)
        proximity_score = 1.0 - min(1.0, distance * 2)
        return proximity_score
    
    def detect_interactions(self, frame: np.ndarray) -> Tuple[Dict[str, float], float, Optional[np.ndarray]]:
        """
        Detect social reciprocity patterns in the given frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple containing:
            - Dictionary of detected interactions and their scores
            - Overall social reciprocity score
            - Annotated frame
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with both pose and face mesh
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        
        if not pose_results.pose_landmarks or not face_results.multi_face_landmarks:
            return {}, 0.0, frame
        
        # Get landmarks for all detected people
        landmarks_list = face_results.multi_face_landmarks
        
        # Update history
        if landmarks_list:
            self.attention_history.append(
                self._calculate_attention_direction(landmarks_list[0])
            )
            
            # Store hand positions for gesture detection
            if pose_results.pose_landmarks:
                left_hand = np.array([pose_results.pose_landmarks.landmark[15].x,
                                    pose_results.pose_landmarks.landmark[15].y])
                right_hand = np.array([pose_results.pose_landmarks.landmark[16].x,
                                     pose_results.pose_landmarks.landmark[16].y])
                self.interaction_history.append((left_hand, right_hand))
        
        # Detect patterns
        interaction_scores = {}
        for pattern_name, detector in self.patterns.items():
            score = detector(frame, landmarks_list)
            interaction_scores[pattern_name] = score
        
        # Calculate overall score
        overall_score = np.mean(list(interaction_scores.values())) if interaction_scores else 0.0
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw face mesh landmarks
        for face_landmarks in landmarks_list:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Add interaction labels
        y_offset = 30
        for interaction, score in interaction_scores.items():
            if score > self.threshold:
                label = f"{interaction}: {score:.2f}"
                cv2.putText(annotated_frame, label, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
        
        return interaction_scores, overall_score, annotated_frame
    
    def reset(self):
        """Reset the detector's state."""
        self.interaction_history.clear()
        self.attention_history.clear()
    
    def __del__(self):
        """Clean up resources."""
        self.pose.close()
        self.face_mesh.close() 