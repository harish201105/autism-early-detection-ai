import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib
from pathlib import Path
import atexit
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import os
import uuid

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
EAR_THRESHOLD = 0.2
MOTION_HISTORY_LENGTH = 30
RESPONSE_TIME_WINDOW = 5.0  # seconds
ATTENTION_WINDOW = 3.0  # seconds

# Face Mesh Constants
FACE_MESH_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
FACE_MESH_CONTOURS = mp_face_mesh.FACEMESH_CONTOURS
FACE_MESH_IRISES = mp_face_mesh.FACEMESH_IRISES
FACE_MESH_FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL
FACE_MESH_LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
FACE_MESH_RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
FACE_MESH_LEFT_EYEBROW = mp_face_mesh.FACEMESH_LEFT_EYEBROW
FACE_MESH_RIGHT_EYEBROW = mp_face_mesh.FACEMESH_RIGHT_EYEBROW
FACE_MESH_LIPS = mp_face_mesh.FACEMESH_LIPS

@dataclass
class AnalysisMetrics:
    """Store all analysis metrics."""
    eye_contact: float = 0.0
    eye_contact_duration: float = 0.0
    eye_contact_frequency: int = 0
    repetitive_score: float = 0.0
    pattern_periods: List[float] = None
    social_score: float = 0.0
    expression_metrics: Dict = None
    gesture_scores: Dict = None
    response_time: float = 0.0
    response_history: List[float] = None
    attention_score: float = 0.0
    attention_events: List[Dict] = None

    def __post_init__(self):
        if self.pattern_periods is None:
            self.pattern_periods = []
        if self.expression_metrics is None:
            self.expression_metrics = {}
        if self.gesture_scores is None:
            self.gesture_scores = {}
        if self.response_history is None:
            self.response_history = []
        if self.attention_events is None:
            self.attention_events = []

def ensure_model_directory():
    """Ensure the models directory exists."""
    model_path = Path(__file__).parent.parent / 'models'
    model_path.mkdir(exist_ok=True)
    return model_path

def train_screening_model():
    """Train the screening model if not already trained."""
    try:
        # Try both dataset files
        data_paths = [
            Path(__file__).parent.parent / 'data' / 'raw' / 'screening' / 'Toddler Autism dataset July 2018.csv',
            Path(__file__).parent.parent / 'data' / 'raw' / 'screening' / 'Autism_Screening_Data_Combined.csv'
        ]
        
        df = None
        used_path = None
        
        for path in data_paths:
            if path.exists():
                temp_df = pd.read_csv(path)
                # Print columns for debugging
                st.write(f"Columns in {path.name}:", temp_df.columns.tolist())
                if 'Class/ASD Traits ' in temp_df.columns or 'Class' in temp_df.columns:  # Check if this is the right dataset
                    df = temp_df
                    used_path = path
                    break
        
        if df is None:
            st.error("No suitable dataset found. Please ensure one of the following files exists:")
            for path in data_paths:
                st.error(f"- {path}")
            return None
            
        st.info(f"Using dataset: {used_path.name}")
        
        # Rename columns if using the Toddler dataset
        if 'Class/ASD Traits ' in df.columns:
            df = df.rename(columns={
                'Class/ASD Traits ': 'Class',
                'Age_Mons': 'Age',
                'Family_mem_with_ASD': 'Family_ASD'
            })
        
        # Prepare features
        # First, ensure all categorical columns are strings
        categorical_cols = ['Sex', 'Jaundice', 'Family_ASD', 'Class']
        available_cols = df.columns.tolist()
        
        # Check which categorical columns are available
        missing_cols = [col for col in categorical_cols if col not in available_cols]
        if missing_cols:
            st.warning(f"Some expected columns are missing: {missing_cols}")
            st.warning("Available columns are: " + ", ".join(available_cols))
            
            # If Jaundice is missing, we'll use a default value
            if 'Jaundice' in missing_cols:
                df['Jaundice'] = 'no'  # Default value
                st.info("Added default 'Jaundice' column with value 'no'")
            
            # If Family_ASD is missing, we'll use a default value
            if 'Family_ASD' in missing_cols:
                df['Family_ASD'] = 'no'  # Default value
                st.info("Added default 'Family_ASD' column with value 'no'")
        
        # Process available categorical columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        
        # Map categorical variables
        df['Sex'] = df['Sex'].map({'m': 0, 'f': 1})
        df['Jaundice'] = df['Jaundice'].map({'no': 0, 'yes': 1})
        df['Family_ASD'] = df['Family_ASD'].map({'no': 0, 'yes': 1})
        df['Class'] = df['Class'].map({'no': 0, 'yes': 1})
        
        # Handle any missing values
        df = df.fillna({
            'Age': df['Age'].median() if 'Age' in df.columns else 5,  # Default age if missing
            'Sex': df['Sex'].mode()[0] if 'Sex' in df.columns else 0,
            'Jaundice': df['Jaundice'].mode()[0] if 'Jaundice' in df.columns else 0,
            'Family_ASD': df['Family_ASD'].mode()[0] if 'Family_ASD' in df.columns else 0
        })
        
        # Verify all categorical columns have been properly encoded
        for col in categorical_cols[:-1]:  # Exclude 'Class' as it's the target
            if col in df.columns:
                if df[col].isnull().any():
                    st.error(f"Error: Null values found in {col} after encoding")
                    return None
                if not df[col].isin([0, 1]).all():
                    st.error(f"Error: Invalid values found in {col} after encoding")
                    return None
        
        # Define the exact order of features we want to use
        feature_cols = ['Age', 'Sex', 'Jaundice', 'Family_ASD', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
        
        # Verify all required features are present
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return None
        
        # Select features in the exact order we want
        X = df[feature_cols]
        y = df['Class']
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X, y)
        
        # Save model and feature names
        model_path = ensure_model_directory()
        model_info = {
            'model': model,
            'feature_names': feature_cols,
            'feature_order': feature_cols  # Add explicit feature order
        }
        joblib.dump(model_info, model_path / 'screening_model.joblib')
        
        st.success("Model trained and saved successfully!")
        return model_info
    except Exception as e:
        st.error(f"Error training screening model: {str(e)}")
        st.error("Please check if the dataset files exist and have the correct format.")
        return None

def load_screening_model():
    """Load the trained screening model."""
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'screening_model.joblib'
        if model_path.exists():
            loaded_data = joblib.load(model_path)
            # Check if we loaded a model info dictionary or just the model
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                return loaded_data
            else:
                # If we loaded just the model, create a model info dictionary
                st.warning("Found old model format. Retraining with new format...")
                return train_screening_model()
        else:
            st.info("Screening model not found. Training new model...")
            return train_screening_model()
    except Exception as e:
        st.error(f"Error loading screening model: {str(e)}")
        st.error("Attempting to train a new model...")
        return train_screening_model()

def cleanup_resources():
    """Cleanup function to release resources."""
    if 'face_mesh' in st.session_state:
        st.session_state.face_mesh.close()
    if 'pose' in st.session_state:
        st.session_state.pose.close()
    if 'cap' in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()

def init_session_state(force_reinit=False):
    """Initialize session state variables."""
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = {}
    if force_reinit or 'face_mesh' not in st.session_state:
        try:
            if 'face_mesh' in st.session_state and st.session_state.face_mesh is not None:
                st.session_state.face_mesh.close()
            st.session_state.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            st.session_state.debug_info['face_mesh_reinit'] = 'Face Mesh re-initialized'
        except Exception as e:
            st.session_state.debug_info['face_mesh_error'] = f"Error initializing Face Mesh: {str(e)}"
    if force_reinit or 'pose' not in st.session_state:
        try:
            if 'pose' in st.session_state and st.session_state.pose is not None:
                st.session_state.pose.close()
            st.session_state.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            st.session_state.debug_info['pose_reinit'] = 'Pose re-initialized'
        except Exception as e:
            st.session_state.debug_info['pose_error'] = f"Error initializing Pose: {str(e)}"
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = AnalysisMetrics()
    if 'screening_model_info' not in st.session_state:
        st.session_state.screening_model_info = load_screening_model()
    if 'motion_history' not in st.session_state:
        st.session_state.motion_history = deque(maxlen=MOTION_HISTORY_LENGTH)
    if 'last_response_time' not in st.session_state:
        st.session_state.last_response_time = time.time()
    if 'attention_history' not in st.session_state:
        st.session_state.attention_history = []
    if 'face_mesh_results' not in st.session_state:
        st.session_state.face_mesh_results = None

def process_frame(frame, current_time):
    """Process a single frame for all analyses."""
    if frame is None:
        return None
        
    try:
        # Ensure frame is in the correct format and size
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            st.error("Invalid image format. Expected RGB/BGR image with 3 channels.")
            return None
            
        # Resize frame to a standard size if needed
        target_width = 640
        target_height = 480
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh
        try:
            # Use the persistent FaceMesh object for real-time
            if 'face_mesh' not in st.session_state or st.session_state.face_mesh is None:
                mp_face_mesh = mp.solutions.face_mesh
                st.session_state.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                st.session_state.debug_info['face_mesh_reinit'] = 'Face Mesh re-initialized (auto in loop)'
            mp_face_mesh = st.session_state.face_mesh
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_mesh_results = mp_face_mesh.process(frame_rgb)
        except Exception as e:
            st.session_state.debug_info['face_mesh_error'] = f'Error processing face mesh: {str(e)}'
            face_mesh_results = None
        st.session_state.face_mesh_results = face_mesh_results
        face_detected = bool(face_mesh_results and face_mesh_results.multi_face_landmarks)
        st.session_state.debug_info['realtime_face_detected'] = face_detected
        
        # Get pose results
        try:
            pose_results = st.session_state.pose.process(frame_rgb)
            st.session_state.debug_info['pose_detected'] = pose_results.pose_landmarks is not None
        except Exception as e:
            st.error(f"Error processing pose: {str(e)}")
            st.session_state.debug_info['pose_error'] = str(e)
            return None
        
        # Process all analyses
        metrics = AnalysisMetrics()
        
        # Eye contact analysis
        if st.session_state.face_mesh_results.multi_face_landmarks:
            metrics.eye_contact, metrics.eye_contact_duration, metrics.eye_contact_frequency = \
                enhance_eye_contact_analysis(frame, st.session_state.face_mesh_results)
        
        # Repetitive behavior analysis
        metrics.repetitive_score, metrics.pattern_periods = \
            enhance_repetitive_behavior_analysis(frame)
        
        # Social reciprocity analysis
        if st.session_state.face_mesh_results.multi_face_landmarks:
            metrics.social_score, metrics.expression_metrics = \
                enhance_social_reciprocity_analysis(frame, st.session_state.face_mesh_results)
        
        # Gesture analysis
        if pose_results.pose_landmarks:
            metrics.gesture_scores = analyze_gestures(frame, pose_results)
        
        # Response time analysis
        metrics.response_time, metrics.response_history = \
            analyze_response_time(frame, current_time)
        
        # Joint attention analysis
        if st.session_state.face_mesh_results.multi_face_landmarks and pose_results.pose_landmarks:
            metrics.attention_score, metrics.attention_events = \
                analyze_joint_attention(frame, st.session_state.face_mesh_results, pose_results)
        
        return metrics
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        st.session_state.debug_info['frame_error'] = str(e)
        return None

def draw_face_guides(img, face_mesh_results, show_keypoints=False, show_bbox=True):
    if not face_mesh_results or not face_mesh_results.multi_face_landmarks:
        height, width = img.shape[:2]
        box_color = (0, 0, 255)
        cv2.rectangle(img, (width//4, height//4), (3*width//4, 3*height//4), box_color, 2)
        cv2.putText(img, "No Face Detected", (width//4, height//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        return
    try:
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        h, w = img.shape[:2]
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1)
        )
        if show_keypoints:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
        if show_bbox:
            x_min = w
            x_max = 0
            y_min = h
            y_max = 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    except Exception as e:
        st.error(f"Error in draw_face_guides: {str(e)}")

def draw_pose_guides(img, pose_results):
    if not pose_results or not pose_results.pose_landmarks:
        return
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        img,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    # Draw pose center and alignment guides
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        h, w = img.shape[:2]
        # Get center of pose
        center_x = int(landmarks[mp_pose.PoseLandmark.MID_HIP].x * w)
        center_y = int(landmarks[mp_pose.PoseLandmark.MID_HIP].y * h)
        # Draw pose center
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)
        # Draw pose alignment guides
        guide_color = (255, 255, 0)  # Yellow for guides
        cv2.line(img, (0, center_y), (w, center_y), guide_color, 1)
        cv2.line(img, (center_x, 0), (center_x, h), guide_color, 1)

def draw_annotations(frame, metrics, face_mesh_results=None, pose_results=None, show_face_mesh=True, show_grid=True, show_keypoints=False, show_bbox=True, show_metrics_on_image=True):
    if frame is None:
        return None
    annotated_frame = frame.copy()
    if show_face_mesh:
        draw_face_guides(annotated_frame, face_mesh_results, show_keypoints=show_keypoints, show_bbox=show_bbox)
    if show_grid:
        draw_grid(annotated_frame)
    if show_metrics_on_image:
        def draw_text_with_background(img, text, position, color=(0, 255, 0), bg_color=(0, 0, 0, 128)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45  # Tiny font
            thickness = 1
            padding = 3
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            overlay = img.copy()
            cv2.rectangle(overlay, (position[0] - padding, position[1] - text_height - padding), (position[0] + text_width + padding, position[1] + padding), bg_color, -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.putText(img, text, position, font, font_scale, color, thickness)
        y_position = 22
        line_height = 18
        # Color coding for key metrics
        def metric_color(val, good_thresh=0.7, warn_thresh=0.4):
            if val >= good_thresh:
                return (0, 255, 0)  # Green
            elif val >= warn_thresh:
                return (0, 255, 255)  # Yellow
            else:
                return (0, 0, 255)  # Red
        draw_text_with_background(
            annotated_frame,
            f"Eye Contact: {metrics.eye_contact:.2f} (Dur: {metrics.eye_contact_duration:.1f}s, Freq: {metrics.eye_contact_frequency})",
            (10, y_position), color=metric_color(metrics.eye_contact))
        y_position += line_height
        draw_text_with_background(
            annotated_frame,
            f"Repetitive: {metrics.repetitive_score:.2f}",
            (10, y_position), color=metric_color(metrics.repetitive_score, good_thresh=0.3, warn_thresh=0.15))
        y_position += line_height
        draw_text_with_background(
            annotated_frame,
            f"Social: {metrics.social_score:.2f}",
            (10, y_position), color=metric_color(metrics.social_score))
        y_position += line_height
        if metrics.expression_metrics:
            draw_text_with_background(
                annotated_frame,
                f"Mouth: {metrics.expression_metrics.get('mouth_openness', 0):.2f}  Eyebrow: {metrics.expression_metrics.get('eyebrow_movement', 0):.2f}",
                (10, y_position), color=(200, 200, 255))
            y_position += line_height
        if metrics.gesture_scores:
            gesture_str = ", ".join([f"{k}:{v:.2f}" for k, v in metrics.gesture_scores.items() if v > 0])
            if gesture_str:
                draw_text_with_background(annotated_frame, f"Gestures: {gesture_str}", (10, y_position), color=(255, 200, 0))
                y_position += line_height
        draw_text_with_background(
            annotated_frame,
            f"Resp Time: {metrics.response_time:.2f}s  Joint Attn: {metrics.attention_score:.2f}",
            (10, y_position), color=(0, 255, 255))
        y_position += line_height
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        draw_text_with_background(annotated_frame, timestamp, (10, frame.shape[0] - 10), color=(0, 255, 0))
    return annotated_frame

def enhance_eye_contact_analysis(frame, face_mesh_results):
    """Enhanced eye contact analysis using MediaPipe Face Mesh."""
    if not face_mesh_results.multi_face_landmarks:
        return 0.0, 0.0, 0
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    # Calculate EAR for both eyes
    left_eye = np.array([(face_landmarks.landmark[33].x, face_landmarks.landmark[33].y),
                         (face_landmarks.landmark[160].x, face_landmarks.landmark[160].y),
                         (face_landmarks.landmark[158].x, face_landmarks.landmark[158].y),
                         (face_landmarks.landmark[133].x, face_landmarks.landmark[133].y),
                         (face_landmarks.landmark[153].x, face_landmarks.landmark[153].y),
                         (face_landmarks.landmark[144].x, face_landmarks.landmark[144].y)])
    
    right_eye = np.array([(face_landmarks.landmark[362].x, face_landmarks.landmark[362].y),
                          (face_landmarks.landmark[385].x, face_landmarks.landmark[385].y),
                          (face_landmarks.landmark[387].x, face_landmarks.landmark[387].y),
                          (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y),
                          (face_landmarks.landmark[373].x, face_landmarks.landmark[373].y),
                          (face_landmarks.landmark[380].x, face_landmarks.landmark[380].y)])
    
    # Calculate EAR
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0
    
    # Update metrics
    if 'last_ear' not in st.session_state:
        st.session_state.last_ear = ear
        st.session_state.eye_contact_duration = 0.0
        st.session_state.eye_contact_frequency = 0
        st.session_state.last_eye_contact_time = time.time()
    
    # Calculate eye contact duration and frequency
    if ear > EAR_THRESHOLD:
        st.session_state.eye_contact_duration += time.time() - st.session_state.last_eye_contact_time
        if st.session_state.last_ear <= EAR_THRESHOLD:
            st.session_state.eye_contact_frequency += 1
    st.session_state.last_eye_contact_time = time.time()
    st.session_state.last_ear = ear
    
    # Normalize score
    eye_contact_score = min(1.0, st.session_state.eye_contact_duration / 10.0)
    
    return eye_contact_score, st.session_state.eye_contact_duration, st.session_state.eye_contact_frequency

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR)."""
    # Calculate vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Calculate horizontal distance
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Calculate EAR
    ear = (v1 + v2) / (2.0 * h)
    return ear

def enhance_repetitive_behavior_analysis(frame):
    """Enhanced repetitive behavior analysis using motion detection."""
    try:
        # Ensure frame is in the correct format
        if frame is None or len(frame.shape) != 3:
            return 0.0, []
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize motion history
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = gray
            return 0.0, []
        
        # Ensure last_frame has the same size as current frame
        if st.session_state.last_frame.shape != gray.shape:
            st.session_state.last_frame = cv2.resize(st.session_state.last_frame, 
                                                   (gray.shape[1], gray.shape[0]))
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(st.session_state.last_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Update motion history
        motion_score = np.mean(thresh) / 255.0
        st.session_state.motion_history.append(motion_score)
        st.session_state.last_frame = gray
        
        # Calculate periodicity
        if len(st.session_state.motion_history) >= MOTION_HISTORY_LENGTH:
            # Calculate autocorrelation
            motion_array = np.array(st.session_state.motion_history)
            autocorr = np.correlate(motion_array, motion_array, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            
            # Calculate pattern periods
            periods = np.array([])  # Initialize as empty array
            if len(peaks) > 1:
                periods = np.diff(peaks)
            
            # Calculate repetitive score
            if len(periods) > 0:  # Check length instead of truth value
                period_std = np.std(periods)
                period_mean = np.mean(periods)
                if period_mean > 0:  # Avoid division by zero
                    repetitive_score = 1.0 - min(1.0, period_std / period_mean)
                else:
                    repetitive_score = 0.0
            else:
                repetitive_score = 0.0
        else:
            repetitive_score = 0.0
            periods = np.array([])  # Empty array for consistency
        
        return repetitive_score, periods.tolist()  # Convert to list for easier handling
    except Exception as e:
        st.error(f"Error in repetitive behavior analysis: {str(e)}")
        return 0.0, []

def enhance_social_reciprocity_analysis(frame, face_mesh_results):
    """Enhanced social reciprocity analysis using facial expressions."""
    if not face_mesh_results.multi_face_landmarks:
        return 0.0, {}
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    # Calculate mouth openness
    mouth_top = face_landmarks.landmark[13]
    mouth_bottom = face_landmarks.landmark[14]
    mouth_openness = abs(mouth_bottom.y - mouth_top.y)
    
    # Calculate eyebrow movement
    left_eyebrow = np.mean([face_landmarks.landmark[70].y, face_landmarks.landmark[63].y])
    right_eyebrow = np.mean([face_landmarks.landmark[300].y, face_landmarks.landmark[293].y])
    eyebrow_movement = abs(left_eyebrow - right_eyebrow)
    
    # Calculate head orientation
    nose_tip = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    head_orientation = abs((left_eye.x + right_eye.x) / 2 - nose_tip.x)
    
    # Calculate social engagement score
    social_score = (mouth_openness + eyebrow_movement + (1 - head_orientation)) / 3.0
    
    # Store detailed metrics
    expression_metrics = {
        'mouth_openness': mouth_openness,
        'eyebrow_movement': eyebrow_movement,
        'head_orientation': head_orientation
    }
    
    return social_score, expression_metrics

def analyze_gestures(frame, pose_results):
    """Analyze gestures using pose landmarks."""
    if not pose_results.pose_landmarks:
        return {}
    
    landmarks = pose_results.pose_landmarks.landmark
    
    # Get hand positions
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Calculate gesture scores
    gesture_scores = {
        'pointing_left': 0.0,
        'pointing_right': 0.0,
        'showing_left': 0.0,
        'showing_right': 0.0
    }
    
    # Detect pointing gestures
    if left_wrist.y < left_shoulder.y:
        gesture_scores['pointing_left'] = 1.0 - (left_wrist.y - left_shoulder.y)
    if right_wrist.y < right_shoulder.y:
        gesture_scores['pointing_right'] = 1.0 - (right_wrist.y - right_shoulder.y)
    
    # Detect showing gestures
    if abs(left_wrist.x - left_shoulder.x) > 0.2:
        gesture_scores['showing_left'] = 1.0
    if abs(right_wrist.x - right_shoulder.x) > 0.2:
        gesture_scores['showing_right'] = 1.0
    
    return gesture_scores

def analyze_response_time(frame, current_time):
    """Analyze response time to visual stimuli."""
    if 'last_stimulus_time' not in st.session_state:
        st.session_state.last_stimulus_time = current_time
        st.session_state.response_history = []
    
    # Simulate stimulus presentation every 5 seconds
    if current_time - st.session_state.last_stimulus_time > RESPONSE_TIME_WINDOW:
        # Check for response (simplified - using motion as response)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if 'last_frame' in st.session_state:
            frame_diff = cv2.absdiff(st.session_state.last_frame, gray)
            motion = np.mean(frame_diff) > 25
            
            if motion:
                response_time = current_time - st.session_state.last_stimulus_time
                st.session_state.response_history.append(response_time)
                st.session_state.last_stimulus_time = current_time
        
        st.session_state.last_frame = gray
    
    # Calculate average response time
    if st.session_state.response_history:
        avg_response_time = np.mean(st.session_state.response_history)
    else:
        avg_response_time = 0.0
    
    return avg_response_time, st.session_state.response_history

def analyze_joint_attention(frame, face_mesh_results, pose_results):
    """Analyze joint attention using gaze and pose information."""
    if not face_mesh_results.multi_face_landmarks or not pose_results.pose_landmarks:
        return 0.0, []
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    pose_landmarks = pose_results.pose_landmarks.landmark
    
    # Calculate gaze direction
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    nose_tip = face_landmarks.landmark[1]
    
    gaze_direction = np.array([
        (left_eye.x + right_eye.x) / 2 - nose_tip.x,
        (left_eye.y + right_eye.y) / 2 - nose_tip.y
    ])
    
    # Get hand positions
    left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Calculate attention to hands
    attention_to_hands = False
    if abs(gaze_direction[0] - (left_wrist.x - nose_tip.x)) < 0.1 or \
       abs(gaze_direction[0] - (right_wrist.x - nose_tip.x)) < 0.1:
        attention_to_hands = True
    
    # Update attention history
    if 'attention_history' not in st.session_state:
        st.session_state.attention_history = []
    
    attention_event = {
        'timestamp': time.time(),
        'gaze_direction': gaze_direction.tolist(),
        'attention_to_hands': attention_to_hands
    }
    st.session_state.attention_history.append(attention_event)
    
    # Calculate attention score
    recent_events = [e for e in st.session_state.attention_history 
                    if time.time() - e['timestamp'] < ATTENTION_WINDOW]
    if recent_events:
        attention_score = sum(1 for e in recent_events if e['attention_to_hands']) / len(recent_events)
    else:
        attention_score = 0.0
    
    return attention_score, recent_events

def draw_grid(img, grid_size=50, color=(50, 50, 50), thickness=1):
    height, width = img.shape[:2]
    # Draw vertical lines
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), color, thickness)
    # Draw horizontal lines
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), color, thickness)
    # Draw center lines with different color
    center_color = (100, 100, 100)
    cv2.line(img, (width//2, 0), (width//2, height), center_color, thickness)
    cv2.line(img, (0, height//2), (width, height//2), center_color, thickness)

def save_metrics(metrics, label, filename="behavior_metrics_labeled.csv"):
    data = {
        "eye_contact": metrics.eye_contact,
        "eye_contact_duration": metrics.eye_contact_duration,
        "eye_contact_frequency": metrics.eye_contact_frequency,
        "repetitive_score": metrics.repetitive_score,
        "social_score": metrics.social_score,
        "response_time": metrics.response_time,
        "joint_attention": metrics.attention_score,
        "label": label
    }
    df = pd.DataFrame([data])
    file_path = Path(filename)
    if file_path.exists():
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)
    print(f"[DEBUG] Saved metrics: {data} to {filename}")

def get_final_behavior_result(metrics,
    eye_contact_thresh=0.3,
    repetitive_thresh=0.7,
    social_thresh=0.3
):
    model_path = Path("behavior_classifier.joblib")
    features = [
        metrics.eye_contact,
        metrics.eye_contact_duration,
        metrics.eye_contact_frequency,
        metrics.repetitive_score,
        metrics.social_score,
        metrics.response_time,
        metrics.attention_score
    ]
    # Debug: Show metric values
    st.write("**[DEBUG] Metric values:**", {
        "eye_contact": metrics.eye_contact,
        "repetitive_score": metrics.repetitive_score,
        "social_score": metrics.social_score,
        "response_time": metrics.response_time,
        "joint_attention": metrics.attention_score
    })
    # Get thresholds from sidebar (with fallback defaults)
    eye_contact_thresh = st.session_state.get("eye_contact_thresh", 0.3)
    repetitive_thresh = st.session_state.get("repetitive_thresh", 0.7)
    social_thresh = st.session_state.get("social_thresh", 0.3)
    if model_path.exists():
        try:
            clf = joblib.load(model_path)
            prediction = clf.predict([features])[0]
            proba = clf.predict_proba([features])[0]
            reasons = [f"ML Model Confidence: {max(proba):.2f}"]
            color = "green" if prediction == "non_autistic" else "red"
            label = "Non-Autistic Behavior Detected" if prediction == "non_autistic" else "Autistic Behavior Detected"
            st.write("**[DEBUG] ML model used for prediction.**")
            return label, color, reasons
        except Exception as e:
            st.write(f"**[DEBUG] ML model failed: {e}**")
            # Fallback to rule-based if model fails
            pass
    st.write("**[DEBUG] Rule-based logic used for prediction.**")
    # Rule-based fallback (now more sensitive: score >= 1)
    reasons = []
    score = 0
    if metrics.eye_contact < eye_contact_thresh:
        reasons.append(f"Low Eye Contact ({metrics.eye_contact:.2f})")
        score += 1
    if metrics.repetitive_score > repetitive_thresh:
        reasons.append(f"High Repetitive Behavior ({metrics.repetitive_score:.2f})")
        score += 1
    if metrics.social_score < social_thresh:
        reasons.append(f"Low Social Reciprocity ({metrics.social_score:.2f})")
        score += 1
    if score >= 1:
        return "Autistic Behavior Detected", "red", reasons
    else:
        return "Non-Autistic Behavior Detected", "green", ["All key metrics in typical range"]

def main():
    st.set_page_config(
        page_title="Autism Detection System",
        page_icon="🧠",
        layout="wide"
    )
    
    # Sidebar: Add button to force re-initialize Face Mesh and Pose
    with st.sidebar:
        st.subheader("Debug Information & Controls")
        force_reinit = st.button("Re-initialize Face Mesh & Pose")
        if force_reinit:
            init_session_state(force_reinit=True)
            st.success("Face Mesh and Pose re-initialized!")
        else:
            init_session_state()
        # Show debug info
        if 'debug_info' in st.session_state:
            for key, value in st.session_state.debug_info.items():
                st.write(f"{key}: {value}")
        # Show MediaPipe errors if any
        if 'face_mesh_error' in st.session_state.debug_info:
            st.error(f"Face Mesh Error: {st.session_state.debug_info['face_mesh_error']}")
        if 'frame_error' in st.session_state.debug_info:
            st.error(f"Frame Error: {st.session_state.debug_info['frame_error']}")
        if 'pose_error' in st.session_state.debug_info:
            st.error(f"Pose Error: {st.session_state.debug_info['pose_error']}")
        # Calibration mode selection
        st.subheader("Calibration Mode")
        calibration_mode = st.radio(
            "Choose calibration mode:",
            ["Default Calibration", "Rule-based Threshold Calibration"],
            key="calibration_mode"
        )
        # Set thresholds based on mode
        if calibration_mode == "Rule-based Threshold Calibration":
            st.subheader("Rule-based Threshold Calibration")
            eye_contact_thresh = st.slider("Eye Contact Threshold (low)", 0.0, 1.0, 0.3, 0.01, key="eye_contact_thresh")
            repetitive_thresh = st.slider("Repetitive Score Threshold (high)", 0.0, 1.0, 0.7, 0.01, key="repetitive_thresh")
            social_thresh = st.slider("Social Score Threshold (low)", 0.0, 1.0, 0.3, 0.01, key="social_thresh")
        else:
            eye_contact_thresh = 0.3
            repetitive_thresh = 0.7
            social_thresh = 0.3
    
    st.title("AI-Driven Early Detection of Autism in Toddlers")
    
    # Add sidebar toggles at the start of main()
    st.sidebar.subheader("Overlay Options")
    show_face_mesh = st.sidebar.checkbox("Show Face Mesh", value=True)
    show_grid = st.sidebar.checkbox("Show Grid Lines", value=True)
    show_keypoints = st.sidebar.checkbox("Show Keypoints", value=False)
    show_bbox = st.sidebar.checkbox("Show Bounding Box", value=True)
    show_metrics_on_image = st.sidebar.checkbox("Show Metrics on Image", value=True)
    
    # Sidebar for mode selection
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        ["Real-time Video", "Video Upload", "Image Upload", "Screening Questionnaire"]
    )
    
    if analysis_mode == "Real-time Video":
        st.header("Real-time Video Analysis")
        
        # Create a container for the camera controls
        camera_container = st.container()
        
        with camera_container:
            if st.button("Start Camera", key="start_camera"):
                if st.session_state.cap is None or not st.session_state.cap.isOpened():
                    st.session_state.cap = cv2.VideoCapture(0)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    st.session_state.camera_running = True
                    # Always create a new FaceMesh object on camera start
                    mp_face_mesh = mp.solutions.face_mesh
                    if 'face_mesh' in st.session_state and st.session_state.face_mesh is not None:
                        st.session_state.face_mesh.close()
                    st.session_state.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    st.session_state.debug_info['face_mesh_reinit'] = 'Face Mesh re-initialized (camera start)'
            
            if st.button("Stop Camera", key="stop_camera"):
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                    st.session_state.camera_running = False
                    # Close FaceMesh object on camera stop
                    if 'face_mesh' in st.session_state and st.session_state.face_mesh is not None:
                        st.session_state.face_mesh.close()
                        st.session_state.face_mesh = None
                    st.session_state.debug_info['face_mesh_closed'] = 'Face Mesh closed (camera stop)'
                    st.rerun()
        
        # Create placeholders for video and metrics
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
        raw_frame_placeholder = st.empty()  # New: for raw frame
        debug_placeholder = st.empty()      # New: for debug info
        
        try:
            while st.session_state.cap is not None and st.session_state.cap.isOpened():
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Show raw frame for debugging
                raw_frame_placeholder.image(frame, channels="BGR", caption="Raw Camera Frame", use_container_width=True)
                
                # Add debug info about the frame
                debug_info = {
                    "frame_shape": frame.shape if frame is not None else None,
                    "frame_dtype": str(frame.dtype) if frame is not None else None,
                    "frame_min": np.min(frame) if frame is not None else None,
                    "frame_max": np.max(frame) if frame is not None else None
                }
                debug_placeholder.write("**Frame Debug Info:**")
                debug_placeholder.json(debug_info)
                
                # Process frame
                current_time = time.time()
                # Use the persistent FaceMesh object for real-time
                if 'face_mesh' not in st.session_state or st.session_state.face_mesh is None:
                    mp_face_mesh = mp.solutions.face_mesh
                    st.session_state.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    st.session_state.debug_info['face_mesh_reinit'] = 'Face Mesh re-initialized (auto in loop)'
                mp_face_mesh = st.session_state.face_mesh
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    face_mesh_results = mp_face_mesh.process(frame_rgb)
                except Exception as e:
                    st.session_state.debug_info['face_mesh_error'] = f'Error processing face mesh: {str(e)}'
                    face_mesh_results = None
                st.session_state.face_mesh_results = face_mesh_results
                face_detected = bool(face_mesh_results and face_mesh_results.multi_face_landmarks)
                st.session_state.debug_info['realtime_face_detected'] = face_detected
                metrics = process_frame(frame, current_time)
                
                if metrics:
                    # Draw annotations on frame
                    annotated_frame = draw_annotations(frame, metrics, face_mesh_results=face_mesh_results, show_face_mesh=show_face_mesh, show_grid=show_grid, show_keypoints=show_keypoints, show_bbox=show_bbox, show_metrics_on_image=show_metrics_on_image)
                    
                    # Display video with annotations
                    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                    
                    # Display metrics in sidebar for additional reference
                    with metrics_placeholder.container():
                        st.sidebar.subheader("Real-time Metrics")
                        st.sidebar.metric("Eye Contact Score", f"{metrics.eye_contact:.2f}")
                        st.sidebar.metric("Repetitive Behaviors", f"{metrics.repetitive_score:.2f}")
                        st.sidebar.metric("Social Reciprocity", f"{metrics.social_score:.2f}")
                        st.sidebar.metric("Gesture Score", f"{sum(metrics.gesture_scores.values())/4:.2f}")
                        st.sidebar.metric("Response Time", f"{metrics.response_time:.2f}s")
                        st.sidebar.metric("Joint Attention", f"{metrics.attention_score:.2f}")
                    # Final result and explanation
                    final_result, color, reasons = get_final_behavior_result(
                        metrics,
                        eye_contact_thresh=eye_contact_thresh,
                        repetitive_thresh=repetitive_thresh,
                        social_thresh=social_thresh
                    )
                    st.markdown(f"<h2 style='color:{color};text-align:center;'>Final Result: {final_result}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color};text-align:center;'>Reason: {', '.join(reasons)}</p>", unsafe_allow_html=True)
                    # Data collection controls
                    st.sidebar.subheader("Data Collection")
                    unique_id = uuid.uuid4().hex[:8]
                    label_key = f"label_{analysis_mode.replace(' ', '_')}_{unique_id}"
                    save_key = f"save_{analysis_mode.replace(' ', '_')}_{unique_id}"
                    label = st.sidebar.selectbox("Label for this sample", ["non_autistic", "autistic"], key=label_key)
                    if st.sidebar.button("Save Metrics with Label", key=save_key):
                        save_metrics(metrics, label)
                        st.sidebar.success("Metrics saved!")
                
                # Add a small delay to prevent high CPU usage
                time.sleep(0.1)
        
        finally:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            if 'face_mesh' in st.session_state and st.session_state.face_mesh is not None:
                st.session_state.face_mesh.close()
                st.session_state.face_mesh = None
    
    elif analysis_mode == "Video Upload":
        st.header("Video Upload Analysis")
        
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                cap = cv2.VideoCapture(str(temp_path))
                if not cap.isOpened():
                    st.error("Error opening video file")
                    return
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create progress bar
                progress_bar = st.progress(0)
                video_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                frame_count = 0
                # Use a fresh FaceMesh object for this video
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Convert to RGB for MediaPipe
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_mesh_results = face_mesh.process(frame_rgb)
                        # Patch: temporarily override st.session_state.face_mesh_results for this call
                        st.session_state.face_mesh_results = face_mesh_results
                        # Process frame
                        current_time = frame_count / fps
                        metrics = process_frame(frame, current_time)
                        if metrics:
                            # Draw annotations on frame, pass in face_mesh_results
                            annotated_frame = draw_annotations(frame, metrics, face_mesh_results=face_mesh_results, show_face_mesh=show_face_mesh, show_grid=show_grid, show_keypoints=show_keypoints, show_bbox=show_bbox, show_metrics_on_image=show_metrics_on_image)
                            # Display video with annotations
                            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                            # Update progress
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            # Display metrics in sidebar
                            with metrics_placeholder.container():
                                st.sidebar.subheader("Video Analysis Metrics")
                                st.sidebar.metric("Eye Contact Score", f"{metrics.eye_contact:.2f}")
                                st.sidebar.metric("Repetitive Behaviors", f"{metrics.repetitive_score:.2f}")
                                st.sidebar.metric("Social Reciprocity", f"{metrics.social_score:.2f}")
                                st.sidebar.metric("Gesture Score", f"{sum(metrics.gesture_scores.values())/4:.2f}")
                                st.sidebar.metric("Response Time", f"{metrics.response_time:.2f}s")
                                st.sidebar.metric("Joint Attention", f"{metrics.attention_score:.2f}")
                        frame_count += 1
                # Cleanup
                cap.release()
                temp_path.unlink()
                # After video analysis, show final result
                if frame_count > 0:
                    # Average metrics over all frames
                    avg_metrics = AnalysisMetrics()
                    # Collect lists for each metric
                    eye_contacts, eye_durations, eye_freqs = [], [], []
                    reps, socials, resp_times, attns = [], [], [], []
                    for m in getattr(st.session_state, 'video_metrics', []):
                        eye_contacts.append(m.eye_contact)
                        eye_durations.append(m.eye_contact_duration)
                        eye_freqs.append(m.eye_contact_frequency)
                        reps.append(m.repetitive_score)
                        socials.append(m.social_score)
                        resp_times.append(m.response_time)
                        attns.append(m.attention_score)
                    if eye_contacts:
                        avg_metrics.eye_contact = sum(eye_contacts) / len(eye_contacts)
                        avg_metrics.eye_contact_duration = sum(eye_durations) / len(eye_durations)
                        avg_metrics.eye_contact_frequency = int(sum(eye_freqs) / len(eye_freqs))
                        avg_metrics.repetitive_score = sum(reps) / len(reps)
                        avg_metrics.social_score = sum(socials) / len(socials)
                        avg_metrics.response_time = sum(resp_times) / len(resp_times)
                        avg_metrics.attention_score = sum(attns) / len(attns)
                        final_result, color, reasons = get_final_behavior_result(
                            avg_metrics,
                            eye_contact_thresh=eye_contact_thresh,
                            repetitive_thresh=repetitive_thresh,
                            social_thresh=social_thresh
                        )
                        st.markdown(f"<h2 style='color:{color};text-align:center;'>Final Result: {final_result}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color:{color};text-align:center;'>Reason: {', '.join(reasons)}</p>", unsafe_allow_html=True)
                    # Data collection controls
                    st.sidebar.subheader("Data Collection")
                    label_key = f"label_{analysis_mode.replace(' ', '_')}"
                    save_key = f"save_{analysis_mode.replace(' ', '_')}"
                    label = st.sidebar.selectbox("Label for this sample", ["non_autistic", "autistic"], key=label_key)
                    if st.sidebar.button("Save Metrics with Label", key=save_key):
                        save_metrics(avg_metrics, label)
                        st.sidebar.success("Metrics saved!")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                if temp_path.exists():
                    temp_path.unlink()
    
    elif analysis_mode == "Image Upload":
        st.header("Image Upload Analysis")
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if frame is None:
                    st.error("Error reading image")
                    return
                
                # Debug: Show image properties
                st.write("**Image Debug Info (Streamlit):**")
                st.json({
                    "shape": frame.shape,
                    "dtype": str(frame.dtype),
                    "min": int(np.min(frame)),
                    "max": int(np.max(frame))
                })
                
                # Convert to RGB for MediaPipe (as in standalone script)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use a fresh FaceMesh object with static_image_mode=True for image analysis
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                    face_mesh_results = face_mesh.process(frame_rgb)
                    face_detected = bool(face_mesh_results.multi_face_landmarks)
                    st.write(f"**Standalone MediaPipe FaceMesh Test:** Face detected: {face_detected}")
                
                # If face detected, proceed with rest of analysis
                if not face_detected:
                    st.warning("MediaPipe Face Mesh did not detect a face in this image.")
                
                # Process frame (original pipeline), but pass in face_mesh_results from above
                current_time = time.time()
                # Patch: temporarily override st.session_state.face_mesh_results for this call
                st.session_state.face_mesh_results = face_mesh_results
                metrics = process_frame(frame, current_time)
                
                if metrics:
                    # Draw annotations on frame, pass in face_mesh_results
                    annotated_frame = draw_annotations(frame, metrics, face_mesh_results=face_mesh_results, show_face_mesh=show_face_mesh, show_grid=show_grid, show_keypoints=show_keypoints, show_bbox=show_bbox, show_metrics_on_image=show_metrics_on_image)
                    
                    # Display image with annotations
                    st.image(annotated_frame, channels="BGR", use_container_width=True)
                    
                    # Display metrics in sidebar
                    st.sidebar.subheader("Image Analysis Metrics")
                    st.sidebar.metric("Eye Contact Score", f"{metrics.eye_contact:.2f}")
                    st.sidebar.write(f"Duration: {metrics.eye_contact_duration:.1f}s, Frequency: {metrics.eye_contact_frequency}")
                    st.sidebar.metric("Repetitive Behaviors", f"{metrics.repetitive_score:.2f}")
                    st.sidebar.metric("Social Reciprocity", f"{metrics.social_score:.2f}")
                    if metrics.expression_metrics: st.sidebar.write(metrics.expression_metrics)
                    if metrics.gesture_scores: st.sidebar.write(metrics.gesture_scores)
                    st.sidebar.metric("Response Time", f"{metrics.response_time:.2f}s")
                    st.sidebar.metric("Joint Attention", f"{metrics.attention_score:.2f}")
                    # Final result and explanation
                    final_result, color, reasons = get_final_behavior_result(
                        metrics,
                        eye_contact_thresh=eye_contact_thresh,
                        repetitive_thresh=repetitive_thresh,
                        social_thresh=social_thresh
                    )
                    st.markdown(f"<h2 style='color:{color};text-align:center;'>Final Result: {final_result}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color};text-align:center;'>Reason: {', '.join(reasons)}</p>", unsafe_allow_html=True)
                    # Data collection controls
                    st.sidebar.subheader("Data Collection")
                    label_key = f"label_{analysis_mode.replace(' ', '_')}"
                    save_key = f"save_{analysis_mode.replace(' ', '_')}"
                    label = st.sidebar.selectbox("Label for this sample", ["non_autistic", "autistic"], key=label_key)
                    if st.sidebar.button("Save Metrics with Label", key=save_key):
                        save_metrics(metrics, label)
                        st.sidebar.success("Metrics saved!")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Please ensure the image is a valid color image (JPG, JPEG, or PNG).")
    
    else:  # Screening Questionnaire
        st.header("Autism Screening Questionnaire")
        st.markdown("""
        Please answer the following questions about the child's behavior.
        This questionnaire is based on standard autism screening tools.
        """)
        
        # Create questionnaire form
        with st.form("screening_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Demographic Information")
                age = st.number_input("Age (years)", min_value=0, max_value=18, value=5)
                sex = st.radio("Sex", ["m", "f"])
                jaundice = st.radio("Jaundice at birth", ["no", "yes"])
                family_asd = st.radio("Family history of ASD", ["no", "yes"])
            
            with col2:
                st.subheader("Screening Questions")
                st.markdown("""
                Please answer the following questions about the child's behavior
                (1 = Yes, 0 = No):
                """)
                a1 = st.radio("A1: Does your child look at you when you call his/her name?", [0, 1])
                a2 = st.radio("A2: Does your child have difficulty making eye contact?", [0, 1])
                a3 = st.radio("A3: Does your child engage in repetitive behaviors?", [0, 1])
                a4 = st.radio("A4: Does your child have unusual sensory interests?", [0, 1])
                a5 = st.radio("A5: Does your child have difficulty with social interaction?", [0, 1])
                a6 = st.radio("A6: Does your child have delayed speech development?", [0, 1])
                a7 = st.radio("A7: Does your child have difficulty understanding others' emotions?", [0, 1])
                a8 = st.radio("A8: Does your child have restricted interests?", [0, 1])
                a9 = st.radio("A9: Does your child have difficulty with changes in routine?", [0, 1])
                a10 = st.radio("A10: Does your child have difficulty with pretend play?", [0, 1])
            
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if st.session_state.screening_model_info is None:
                    st.warning("Screening model is not available. Please try again.")
                    return
                
                try:
                    # Verify we have a model info dictionary
                    if not isinstance(st.session_state.screening_model_info, dict):
                        st.error("Invalid model format. Retraining model...")
                        st.session_state.screening_model_info = train_screening_model()
                        if st.session_state.screening_model_info is None:
                            st.error("Failed to train new model")
                            return
                    
                    # Get the model and feature names
                    model = st.session_state.screening_model_info.get('model')
                    feature_names = st.session_state.screening_model_info.get('feature_names', [])
                    
                    if model is None:
                        st.error("Model not found in model info")
                        return
                    
                    if not feature_names:
                        st.error("Feature names not found in model info")
                        return
                    
                    # Create input data as a numpy array in the correct order
                    input_values = [
                        age,
                        0 if sex == 'm' else 1,
                        0 if jaundice == 'no' else 1,
                        0 if family_asd == 'no' else 1,
                        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10
                    ]
                    
                    # Convert to numpy array and reshape for prediction
                    input_array = np.array(input_values).reshape(1, -1)
                    
                    # Get prediction
                    try:
                        prediction = model.predict_proba(input_array)[0][1]
                    except Exception as pred_error:
                        st.error(f"Prediction error: {str(pred_error)}")
                        st.write("Input array shape:", input_array.shape)
                        st.write("Model feature count:", model.n_features_in_)
                        return
                    
                    st.session_state.analysis_results.eye_contact = prediction
                    
                    # Display results
                    st.subheader("Screening Results")
                    st.metric(
                        "ASD Likelihood Score",
                        f"{prediction:.2%}",
                        delta=None
                    )
                    
                    # Display interpretation
                    if prediction > 0.7:
                        st.warning("""
                        The screening results suggest a higher likelihood of ASD.
                        Please consult with:
                        - Pediatrician
                        - Child psychologist
                        - Developmental specialist
                        """)
                    else:
                        st.success("""
                        The screening results suggest a lower likelihood of ASD.
                        Continue regular developmental monitoring.
                        """)
                    
                    st.markdown("""
                    **Note**: This is a screening tool only and not a diagnostic tool.
                    Always consult healthcare professionals for proper assessment.
                    """)
                    # Data collection controls
                    st.sidebar.subheader("Data Collection")
                    label_key = f"label_{analysis_mode.replace(' ', '_')}"
                    save_key = f"save_{analysis_mode.replace(' ', '_')}"
                    label = st.sidebar.selectbox("Label for this sample", ["non_autistic", "autistic"], key=label_key)
                    if st.sidebar.button("Save Metrics with Label", key=save_key):
                        save_metrics(st.session_state.analysis_results, label)
                        st.sidebar.success("Metrics saved!")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.error("Please check if all inputs are valid and try again.")
                    # Add more detailed error information
                    st.write("Error details:", str(e))
                    st.write("Model info type:", type(st.session_state.screening_model_info))

    # Only show feedback/calibration/model validation if metrics is defined
    metrics_defined = 'metrics' in locals() or 'metrics' in globals()
    labeled_data_exists = os.path.exists("behavior_metrics_labeled.csv")

    if metrics_defined:
        # Sidebar: Dataset stats
        if labeled_data_exists:
            try:
                df = pd.read_csv("behavior_metrics_labeled.csv")
                st.sidebar.subheader("Dataset Stats")
                st.sidebar.write(f"Samples: {len(df)}")
                st.sidebar.write(df['label'].value_counts().to_dict())
            except Exception:
                pass
        # Sidebar: Threshold calibration and visualization
        if labeled_data_exists and st.sidebar.checkbox("Show Metric Distributions (Calibration)"):
            try:
                df = pd.read_csv("behavior_metrics_labeled.csv")
                metric = st.sidebar.selectbox("Metric to visualize", ["eye_contact", "repetitive_score", "social_score", "response_time", "joint_attention"])
                fig, ax = plt.subplots()
                sns.histplot(df[metric], kde=True, hue=df['label'], ax=ax)
                st.sidebar.pyplot(fig)
                # Interactive threshold adjustment
                thresh = st.sidebar.slider(f"Adjust threshold for {metric}", float(df[metric].min()), float(df[metric].max()), float(df[metric].mean()))
                st.sidebar.write(f"Current sample: {getattr(metrics, metric, None):.2f}")
                st.sidebar.write(f"Threshold: {thresh:.2f}")
            except Exception:
                st.sidebar.warning("No data for calibration yet.")
        elif not labeled_data_exists:
            st.sidebar.info("No labeled data yet. Save some samples to enable calibration tools.")
        # Sidebar: Model validation
        if labeled_data_exists and st.sidebar.button("Run Model Validation"):
            try:
                df = pd.read_csv("behavior_metrics_labeled.csv")
                
                # Check minimum sample size
                MIN_SAMPLES = 5  # Minimum samples needed for meaningful validation
                if len(df) < MIN_SAMPLES:
                    st.sidebar.warning(f"Not enough samples for validation. Need at least {MIN_SAMPLES} samples, but have {len(df)}.")
                    st.sidebar.info("Please collect more labeled data using the data collection features.")
                    return
                
                # Check class balance
                class_counts = df['label'].value_counts()
                if len(class_counts) < 2:
                    st.sidebar.warning("Need samples from both classes (autistic and non_autistic) for validation.")
                    st.sidebar.info("Current class distribution:", class_counts.to_dict())
                    return
                
                X = df.drop("label", axis=1)
                y = df["label"]
                
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score, StratifiedKFold
                from sklearn.metrics import classification_report, confusion_matrix
                
                # Use cross-validation for small datasets
                if len(df) < 20:  # Use cross-validation for small datasets
                    st.sidebar.info("Using 5-fold cross-validation due to small dataset size")
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    cv = StratifiedKFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
                    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
                    
                    st.sidebar.write("Cross-validation results:")
                    st.sidebar.write(f"Mean accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
                    
                    # Train on full dataset for confusion matrix
                    clf.fit(X, y)
                    y_pred = clf.predict(X)
                    
                else:  # Use stratified train/test split for larger datasets
                    from sklearn.model_selection import train_test_split
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                    except ValueError as e:
                        st.sidebar.error(f"Stratified split failed: {e}")
                        st.sidebar.warning("This usually means one class is too small or missing. Please collect more balanced data.")
                        return
                    # Check if both classes are present in train and test sets
                    train_class_counts = pd.Series(y_train).value_counts()
                    test_class_counts = pd.Series(y_test).value_counts()
                    if len(train_class_counts) < 2 or len(test_class_counts) < 2:
                        st.sidebar.warning(f"Class imbalance detected after split. Train set: {train_class_counts.to_dict()}, Test set: {test_class_counts.to_dict()}")
                        st.sidebar.warning("Please collect more balanced data for reliable validation.")
                        return
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_true = y_test
                    acc = clf.score(X_test, y_test)
                    st.sidebar.write(f"Test set accuracy: {acc:.2f}")
                    st.sidebar.text(classification_report(y_true, y_pred))
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=["non_autistic", "autistic"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                           xticklabels=["non_autistic", "autistic"], 
                           yticklabels=["non_autistic", "autistic"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix")
                st.sidebar.pyplot(fig)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': clf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax2)
                ax2.set_title('Feature Importance')
                st.sidebar.pyplot(fig2)
                
                # ROC curve (only if we have both classes)
                if len(set(y_true)) == 2:
                    y_score = clf.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve((y_true == "autistic").astype(int), y_score)
                    roc_auc = auc(fpr, tpr)
                    fig3, ax3 = plt.subplots()
                    ax3.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
                    ax3.plot([0, 1], [0, 1], 'k--')
                    ax3.set_xlabel('False Positive Rate')
                    ax3.set_ylabel('True Positive Rate')
                    ax3.set_title('Receiver Operating Characteristic')
                    ax3.legend(loc="lower right")
                    st.sidebar.pyplot(fig3)
                
                # Save the trained model
                model_path = ensure_model_directory()
                joblib.dump(clf, model_path / 'behavior_classifier.joblib')
                st.sidebar.success("Model trained and saved successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Validation failed: {str(e)}")
                st.sidebar.error("Please ensure your data is properly formatted and try again.")
        # Sidebar: User feedback form
        st.sidebar.subheader("User Feedback")
        feedback = st.sidebar.radio("Was the prediction correct?", ["Yes", "No"])
        feedback_type = st.sidebar.selectbox("If No, was it a...", ["False Positive", "False Negative", "N/A"])
        if st.sidebar.button("Submit Feedback"):
            feedback_data = {
                "timestamp": pd.Timestamp.now(),
                "metrics": str(metrics),
                "prediction": get_final_behavior_result(metrics, eye_contact_thresh, repetitive_thresh, social_thresh)[0],
                "feedback": feedback,
                "feedback_type": feedback_type
            }
            feedback_df = pd.DataFrame([feedback_data])
            feedback_file = Path("user_feedback.csv")
            if feedback_file.exists():
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_file, mode='w', header=True, index=False)
            st.sidebar.success("Feedback submitted!")
    else:
        st.sidebar.info("Run an analysis to enable feedback, calibration, and validation tools.")

if __name__ == "__main__":
    main() 