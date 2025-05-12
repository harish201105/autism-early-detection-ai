# Model Documentation

## Overview
This project implements advanced behavioral analysis models for early autism detection using computer vision and machine learning. The models are integrated into a Streamlit app with robust, mode-specific MediaPipe FaceMesh initialization and customizable visualization overlays.

## Behavioral Metrics
- **Eye Contact Analysis**
  - Eye Contact Score (0-1)
  - Duration (seconds)
  - Frequency (count)
- **Repetitive Behavior Detection**
  - Repetitive Score (0-1)
- **Social Reciprocity Assessment**
  - Social Score (0-1)
  - Mouth Openness
  - Eyebrow Movement
  - Head Orientation
- **Gesture Recognition**
  - Pointing, Showing, and other gestures
- **Response Time Analysis**
  - Response Time (seconds)
- **Joint Attention Tracking**
  - Joint Attention Score (0-1)

## Model Architecture
- **Face Mesh Model:**
  - Uses MediaPipe FaceMesh with mode-specific initialization:
    - `static_image_mode=True` for images
    - `static_image_mode=False` for video/real-time
  - Re-initialized as needed for reliability
- **Random Forest Classifier:**
  - Used for questionnaire-based screening

## Visualization & UI/UX
- **Customizable Overlays:**
  - Face Mesh, Grid Lines, Keypoints, Bounding Box, Metrics
- **Large Image/Video Display:**
  - `use_container_width=True` for maximum visibility
- **Compact, Color-Coded Metric Overlay:**
  - Tiny font, semi-transparent background
  - Key metrics on image, all details in sidebar
- **Sidebar:**
  - All detailed metrics, including frequency, duration, gesture breakdown, etc.

## Error Handling
- Robust session state management
- Automatic re-initialization of models
- Debug sidebar for troubleshooting

## Example Output
- Eye Contact: 0.85 (Duration: 6.2s, Freq: 3)
- Repetitive Score: 0.12
- Social Score: 0.67 (Mouth: 0.12, Eyebrow: 0.08)
- Gestures: Pointing Right: 1.00
- Response Time: 0.45s
- Joint Attention: 0.90

## Behavioral Analysis Models

### 1. Eye Contact Analysis

#### Implementation
```python
def enhance_eye_contact_analysis(frame, face_mesh_results):
    """
    Analyzes eye contact using MediaPipe Face Mesh landmarks.
    
    Features:
    - Eye Aspect Ratio (EAR) calculation
    - Duration tracking
    - Frequency measurement
    
    Returns:
    - eye_contact_score (0-1)
    - duration (seconds)
    - frequency (count)
    """
```

#### Technical Details
- Uses MediaPipe Face Mesh (468 facial landmarks)
- Calculates EAR for both eyes
- Normalizes scores to 0-1 range
- Tracks temporal patterns

### 2. Repetitive Behavior Detection

#### Implementation
```python
def enhance_repetitive_behavior_analysis(frame):
    """
    Detects repetitive behaviors using motion analysis.
    
    Features:
    - Motion detection
    - Pattern periodicity
    - Autocorrelation analysis
    
    Returns:
    - repetitive_score (0-1)
    - pattern_periods (list)
    """
```

#### Technical Details
- Frame differencing for motion detection
- Autocorrelation for pattern detection
- Periodicity analysis
- Motion history tracking

### 3. Social Reciprocity Assessment

#### Implementation
```python
def enhance_social_reciprocity_analysis(frame, face_mesh_results):
    """
    Analyzes social reciprocity through facial expressions.
    
    Features:
    - Facial expression analysis
    - Head orientation tracking
    - Social engagement scoring
    
    Returns:
    - social_score (0-1)
    - expression_metrics (dict)
    """
```

#### Technical Details
- Facial landmark tracking
- Expression intensity measurement
- Head pose estimation
- Engagement scoring

### 4. Gesture Recognition

#### Implementation
```python
def analyze_gestures(frame, pose_results):
    """
    Detects and analyzes gestures using pose landmarks.
    
    Features:
    - Pointing gesture detection
    - Showing behavior analysis
    - Hand-to-face interaction
    
    Returns:
    - gesture_scores (dict)
    """
```

#### Technical Details
- MediaPipe Pose tracking
- Hand position analysis
- Gesture classification
- Interaction detection

### 5. Response Time Analysis

#### Implementation
```python
def analyze_response_time(frame, current_time):
    """
    Measures response times to visual stimuli.
    
    Features:
    - Stimulus presentation
    - Response detection
    - Latency measurement
    
    Returns:
    - response_time (seconds)
    """
```

#### Technical Details
- Stimulus timing
- Response detection
- Latency calculation
- History tracking

### 6. Joint Attention Tracking

#### Implementation
```python
def analyze_joint_attention(frame, face_mesh_results, pose_results):
    """
    Tracks joint attention behaviors.
    
    Features:
    - Gaze direction monitoring
    - Shared attention detection
    - Sustained attention analysis
    
    Returns:
    - attention_score (0-1)
    - attention_events (list)
    """
```

#### Technical Details
- Gaze vector calculation
- Attention event detection
- Sustained attention tracking
- Event history

## Screening Questionnaire Model

### Implementation
```python
def train_screening_model():
    """
    Trains Random Forest Classifier on screening data.
    
    Features:
    - 10 screening questions
    - Demographic information
    - Medical history
    
    Returns:
    - trained_model (RandomForestClassifier)
    """
```

### Model Architecture
- Type: Random Forest Classifier
- Parameters:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 2
  - min_samples_leaf: 1
  - random_state: 42

### Features
1. **Demographic Information**
   - Age
   - Sex
   - Jaundice at birth
   - Family history of ASD

2. **Screening Questions**
   - A1: Eye contact when called
   - A2: Difficulty with eye contact
   - A3: Repetitive behaviors
   - A4: Sensory interests
   - A5: Social interaction
   - A6: Speech development
   - A7: Emotional understanding
   - A8: Restricted interests
   - A9: Routine changes
   - A10: Pretend play

### Performance Metrics
- Accuracy: ~85%
- Precision: ~0.83
- Recall: ~0.87
- F1-Score: ~0.85

## Model Integration

### Analysis Pipeline
1. **Frame Processing**
   ```python
   def process_frame(frame, current_time):
       """
       Centralized frame processing function.
       
       Steps:
       1. Face mesh detection
       2. Pose estimation
       3. Behavioral analysis
       4. Metric calculation
       
       Returns:
       - AnalysisMetrics object
       """
   ```

2. **Metric Aggregation**
   ```python
   @dataclass
   class AnalysisMetrics:
       """
       Stores all analysis metrics.
       
       Attributes:
       - eye_contact metrics
       - repetitive behavior metrics
       - social reciprocity metrics
       - gesture metrics
       - response time metrics
       - joint attention metrics
       """
   ```

### Resource Management
- Automatic cleanup of resources
- Efficient model initialization
- Memory management
- Progress tracking

## Model Limitations

### Technical Limitations
1. **Performance**
   - Real-time processing at 30 FPS
   - Memory usage optimization
   - GPU acceleration (optional)

2. **Accuracy**
   - Lighting conditions
   - Camera quality
   - Subject cooperation
   - Frame rate stability

### Clinical Limitations
1. **Scope**
   - Not a diagnostic tool
   - Screening aid only
   - Professional interpretation needed

2. **Validation**
   - Clinical validation pending
   - Limited to visible behaviors
   - May miss subtle signs

## Future Improvements

1. **Technical Enhancements**
   - Deep learning for gesture recognition
   - Multi-person tracking
   - Audio analysis integration
   - Mobile device support

2. **Clinical Enhancements**
   - Additional behavioral metrics
   - Longitudinal analysis
   - Integration with medical records
   - Clinical validation studies

## References

1. MediaPipe Documentation
   - Face Mesh: [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)
   - Pose: [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)

2. Research Papers
   - Eye Contact Analysis: [EAR-based Eye Blink Detection](https://doi.org/10.1109/ICPR.2016.7899650)
   - Autism Screening: [M-CHAT-R/F](https://doi.org/10.1542/peds.2013-1813)

3. Datasets
   - Autism Screening: [Kaggle Dataset](https://www.kaggle.com/fabdelja/autism-screening)
   - Autism Images: [Kaggle Dataset](https://www.kaggle.com/cihan063/autism-image-data)

## Calibration Mode

The app provides two calibration modes for rule-based predictions:
- **Default Calibration:** Uses fixed thresholds for key metrics (eye contact, repetitive behavior, social reciprocity).
- **Rule-based Threshold Calibration:** Lets users adjust thresholds for these metrics using sliders in the sidebar.

The selected thresholds are used in the rule-based logic to determine when "Autistic Behavior Detected" is flagged (if any one metric is in the red flag range).

## Debugging and Transparency
- The app displays the actual metric values for each sample.
- It shows whether the ML model or rule-based logic was used for the prediction, and any errors if the ML model fails.

## Rule-Based vs. ML Model
- The ML model is used for predictions if available and trained; otherwise, the rule-based logic (with user-selected thresholds if calibration is enabled) is used.

## Validation and Metrics (Updated)

- All validation metrics (confusion matrix, ROC curve, accuracy, etc.) are now computed only on the test set (y_test/y_true and y_pred), not the full dataset.
- Feature importance is shown for the full model, but all performance metrics and plots use only the test set.
- The code always uses stratified splitting (stratify=y) for train/test split to ensure class balance.
- If stratified splitting fails or if class imbalance is detected in the train or test set, user-friendly warnings are shown and validation is skipped until more balanced data is collected. 