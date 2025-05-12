# API Documentation

This document provides detailed information about the APIs and interfaces used in the AI-Driven Early Detection of Autism in Toddlers project.

## Streamlit Interface

### Main Application (`src/streamlit_app.py`)

#### Components
1. **Navigation**
   ```python
   st.sidebar.title("Navigation")
   page = st.sidebar.selectbox(
       "Choose a page",
       ["Home", "Real-time Analysis", "Video Upload", "Image Upload", "Results"]
   )
   ```

2. **Real-time Analysis**
   ```python
   def start_camera_analysis():
       """
       Initialize camera and start real-time analysis
       Returns:
           - Video stream
           - Analysis results
       """
   ```

3. **Video Upload**
   ```python
   def process_video(video_file):
       """
       Process uploaded video file
       Args:
           video_file: Uploaded video file
       Returns:
           - Analysis results
           - Confidence scores
       """
   ```

4. **Image Upload**
   ```python
   def process_image(image_file):
       """
       Process uploaded image file
       Args:
           image_file: Uploaded image file
       Returns:
           - Analysis results
           - Confidence scores
       """
   ```

### Model Interfaces

#### 1. Eye Contact Detector (`src/models/eye_contact_detector.py`)

```python
class EyeContactDetector:
    def __init__(self, model_path: str):
        """
        Initialize eye contact detector
        Args:
            model_path: Path to trained model
        """
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect eye contact in a frame
        Args:
            frame: Input video frame
        Returns:
            Dictionary containing:
            - eye_contact_probability: float
            - duration: float
            - confidence: float
        """
    
    def analyze_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze sequence of frames
        Args:
            frames: List of video frames
        Returns:
            Dictionary containing:
            - eye_contact_pattern: List[float]
            - total_duration: float
            - confidence: float
        """
```

#### 2. Repetitive Behavior Detector (`src/models/repetitive_behavior_detector.py`)

```python
class RepetitiveBehaviorDetector:
    def __init__(self, model_path: str):
        """
        Initialize repetitive behavior detector
        Args:
            model_path: Path to trained model
        """
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect repetitive behaviors in a frame
        Args:
            frame: Input video frame
        Returns:
            Dictionary containing:
            - behavior_type: str
            - probability: float
            - confidence: float
        """
    
    def analyze_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze sequence of frames
        Args:
            frames: List of video frames
        Returns:
            Dictionary containing:
            - behavior_patterns: List[Dict]
            - frequency: float
            - confidence: float
        """
```

#### 3. Social Reciprocity Detector (`src/models/social_reciprocity_detector.py`)

```python
class SocialReciprocityDetector:
    def __init__(self, model_path: str):
        """
        Initialize social reciprocity detector
        Args:
            model_path: Path to trained model
        """
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect social reciprocity in a frame
        Args:
            frame: Input video frame
        Returns:
            Dictionary containing:
            - interaction_type: str
            - probability: float
            - confidence: float
        """
    
    def analyze_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze sequence of frames
        Args:
            frames: List of video frames
        Returns:
            Dictionary containing:
            - interaction_patterns: List[Dict]
            - quality_score: float
            - confidence: float
        """
```

## Utility Functions

### Data Processing (`src/utils/data_processing.py`)

```python
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess video frame
    Args:
        frame: Input frame
    Returns:
        Preprocessed frame
    """

def extract_features(frames: List[np.ndarray]) -> np.ndarray:
    """
    Extract features from frames
    Args:
        frames: List of frames
    Returns:
        Feature array
    """
```

### Model Management (`src/utils/model_management.py`)

```python
def load_model(model_path: str) -> Any:
    """
    Load trained model
    Args:
        model_path: Path to model file
    Returns:
        Loaded model
    """

def save_model(model: Any, model_path: str):
    """
    Save trained model
    Args:
        model: Model to save
        model_path: Path to save model
    """
```

## Data Structures

### Analysis Results

```python
@dataclass
class AnalysisResult:
    timestamp: datetime
    eye_contact: Dict[str, float]
    repetitive_behavior: Dict[str, float]
    social_reciprocity: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any]
```

### Model Output

```python
@dataclass
class ModelOutput:
    prediction: float
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any]
```

## Error Handling

### Custom Exceptions

```python
class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class InputError(Exception):
    """Exception for invalid input data"""
    pass

class ProcessingError(Exception):
    """Exception for data processing errors"""
    pass
```

## Configuration

### Model Configuration

```python
MODEL_CONFIG = {
    "eye_contact_detector": {
        "model_path": "models/eye_contact_model.h5",
        "input_size": (224, 224),
        "batch_size": 32
    },
    "repetitive_behavior_detector": {
        "model_path": "models/behavior_model.h5",
        "input_size": (224, 224),
        "batch_size": 32
    },
    "social_reciprocity_detector": {
        "model_path": "models/social_model.h5",
        "input_size": (224, 224),
        "batch_size": 32
    }
}
```

### Application Configuration

```python
APP_CONFIG = {
    "max_video_size": 500 * 1024 * 1024,  # 500MB
    "supported_formats": ["mp4", "avi", "mov"],
    "max_frames": 1000,
    "confidence_threshold": 0.7
}
```

## Usage Examples

### Real-time Analysis

```python
# Initialize detectors
eye_detector = EyeContactDetector(MODEL_CONFIG["eye_contact_detector"]["model_path"])
behavior_detector = RepetitiveBehaviorDetector(MODEL_CONFIG["repetitive_behavior_detector"]["model_path"])
social_detector = SocialReciprocityDetector(MODEL_CONFIG["social_reciprocity_detector"]["model_path"])

# Process frame
frame = get_frame_from_camera()
results = {
    "eye_contact": eye_detector.detect(frame),
    "repetitive_behavior": behavior_detector.detect(frame),
    "social_reciprocity": social_detector.detect(frame)
}
```

### Video Analysis

```python
# Process video file
video_path = "path/to/video.mp4"
frames = extract_frames(video_path)
results = {
    "eye_contact": eye_detector.analyze_sequence(frames),
    "repetitive_behavior": behavior_detector.analyze_sequence(frames),
    "social_reciprocity": social_detector.analyze_sequence(frames)
}
```

## Performance Considerations

1. **Memory Management**
   - Batch processing for large videos
   - Frame skipping for real-time analysis
   - Efficient data structures

2. **Processing Speed**
   - GPU acceleration
   - Parallel processing
   - Optimized preprocessing

3. **Resource Usage**
   - Model quantization
   - Efficient data loading
   - Caching mechanisms

## Security Considerations

1. **Data Privacy**
   - Local processing
   - Secure storage
   - Data encryption

2. **Access Control**
   - User authentication
   - Role-based access
   - Session management

3. **Input Validation**
   - File type checking
   - Size limits
   - Format validation

## Contact

For API-related questions:
- Open an issue
- Contact the development team
- Join our community forum 