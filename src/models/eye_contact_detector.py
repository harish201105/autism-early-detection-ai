import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class EyeContactDetector:
    def __init__(self):
        """Initialize the eye contact detector with MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices in MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Parameters for eye contact detection
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.2
        self.GAZE_DIRECTION_THRESHOLD = 0.3
        self.MIN_FRAMES_FOR_CONTACT = 5
        
        # State variables
        self.contact_frames = 0
        self.total_frames = 0
        self.last_contact_state = False
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices: list) -> float:
        """Calculate the eye aspect ratio for a given eye."""
        points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                          for idx in eye_indices])
        
        # Calculate vertical distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        
        # Calculate horizontal distance
        h = np.linalg.norm(points[0] - points[3])
        
        # Calculate eye aspect ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def estimate_gaze_direction(self, landmarks) -> Tuple[float, float]:
        """Estimate the gaze direction based on eye landmarks."""
        # Get eye center points
        left_eye_center = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                                 for idx in self.LEFT_EYE_INDICES], axis=0)
        right_eye_center = np.mean([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                                  for idx in self.RIGHT_EYE_INDICES], axis=0)
        
        # Get iris/pupil positions (simplified)
        left_iris = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y])
        right_iris = np.array([landmarks.landmark[362].x, landmarks.landmark[362].y])
        
        # Calculate gaze vectors
        left_gaze = left_iris - left_eye_center
        right_gaze = right_iris - right_eye_center
        
        # Average gaze direction
        gaze_x = np.mean([left_gaze[0], right_gaze[0]])
        gaze_y = np.mean([left_gaze[1], right_gaze[1]])
        
        return gaze_x, gaze_y
    
    def is_looking_at_camera(self, gaze_x: float, gaze_y: float) -> bool:
        """Determine if the person is looking at the camera based on gaze direction."""
        # Simple threshold-based approach
        return abs(gaze_x) < self.GAZE_DIRECTION_THRESHOLD and abs(gaze_y) < self.GAZE_DIRECTION_THRESHOLD
    
    def detect_eye_contact(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect eye contact in the given frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple containing:
            - Boolean indicating if eye contact is detected
            - Confidence score (0-1)
            - Annotated frame (if requested)
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            self.total_frames += 1
            return False, 0.0, frame
        
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate eye aspect ratios
        left_ear = self.calculate_eye_aspect_ratio(face_landmarks, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_eye_aspect_ratio(face_landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Estimate gaze direction
        gaze_x, gaze_y = self.estimate_gaze_direction(face_landmarks)
        
        # Determine if eyes are open and looking at camera
        eyes_open = avg_ear > self.EYE_ASPECT_RATIO_THRESHOLD
        looking_at_camera = self.is_looking_at_camera(gaze_x, gaze_y)
        
        # Update contact state
        current_contact = eyes_open and looking_at_camera
        if current_contact:
            self.contact_frames += 1
        self.total_frames += 1
        
        # Calculate confidence score
        if self.total_frames > 0:
            confidence = self.contact_frames / self.total_frames
        else:
            confidence = 0.0
        
        # Create annotated frame
        annotated_frame = frame.copy()
        if current_contact:
            cv2.putText(annotated_frame, "Eye Contact", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Eye Contact", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw eye landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        return current_contact, confidence, annotated_frame
    
    def reset(self):
        """Reset the detector's state variables."""
        self.contact_frames = 0
        self.total_frames = 0
        self.last_contact_state = False
    
    def __del__(self):
        """Clean up resources."""
        self.face_mesh.close() 