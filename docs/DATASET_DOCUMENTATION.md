# Dataset Documentation

## Overview

This project utilizes two primary datasets for autism spectrum disorder (ASD) detection:

1. **Autism Screening Dataset**: Questionnaire-based data for initial screening
2. **Autism Image Dataset**: Image data for behavioral pattern analysis

## Behavioral Features
- **Eye Contact:** Score, duration, frequency
- **Repetitive Behaviors:** Score
- **Social Reciprocity:** Score, mouth openness, eyebrow movement, head orientation
- **Gestures:** Pointing, showing, etc.
- **Response Time:** Visual stimulus response
- **Joint Attention:** Gaze and attention tracking

## Data Visualization & Exploration
- All behavioral features are visualized in the app with customizable overlays and color-coded metrics.
- The app uses robust, mode-specific MediaPipe initialization for reliable feature extraction.
- Users can explore data in real time, via video upload, or image upload, with a large, responsive display and detailed sidebar metrics.

## Data Processing
- **Image Data:**
  - Preprocessing: resizing, normalization
  - Feature extraction: MediaPipe FaceMesh, pose, gesture analysis
- **Screening Data:**
  - Categorical and numerical encoding
  - Feature engineering for behavioral metrics

## Example Features
- Eye Contact: 0.85 (Duration: 6.2s, Freq: 3)
- Repetitive Score: 0.12
- Social Score: 0.67 (Mouth: 0.12, Eyebrow: 0.08)
- Gestures: Pointing Right: 1.00
- Response Time: 0.45s
- Joint Attention: 0.90

## Data Privacy & Ethics
- No personally identifiable information is used in behavioral analysis.
- All image data is processed and visualized securely within the app.

## 1. Autism Screening Dataset

### Source
- File: `data/raw/screening/Autism_Screening_Data_Combined.csv`
- Format: CSV
- Size: ~2MB
- Records: 6,076

### Features
1. **Demographic Information**
   - Age (numeric)
   - Sex (binary: m/f)
   - Jaundice at birth (binary: yes/no)
   - Family history of ASD (binary: yes/no)

2. **Screening Questions (A1-A10)**
   - A1: Does your child look at you when you call his/her name?
   - A2: How easy is it for you to get eye contact with your child?
   - A3: Does your child point to indicate that s/he wants something?
   - A4: Does your child point to share interest with you?
   - A5: Does your child pretend?
   - A6: Does your child follow when you're looking at something?
   - A7: If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?
   - A8: Does your child talk back when you talk to him/her?
   - A9: Does your child use simple gestures?
   - A10: Does your child stare at nothing with no apparent purpose?

3. **Target Variable**
   - Class/ASD (binary: YES/NO)

### Preprocessing Steps
```python
def preprocess_screening_data(df):
    """
    Preprocesses screening dataset.
    
    Steps:
    1. Handle missing values
    2. Encode categorical variables
    3. Normalize numeric features
    4. Split into train/test sets
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
```

### Data Storage
- Raw data: `data/raw/screening/`
- Processed data: `data/processed/screening/`
- Features: `data/features/screening/`

## 2. Autism Image Dataset

### Source
- Directory: `data/raw/autism_image/AutismDataset/`
- Format: Images (JPG/PNG)
- Size: ~500MB
- Images: 2,000+ (ASD/Non-ASD)

### Dataset Structure
```
AutismDataset/
├── train/
│   ├── ASD/
│   │   └── [ASD images]
│   └── Non_ASD/
│       └── [Non-ASD images]
├── test/
│   ├── ASD/
│   │   └── [ASD images]
│   └── Non_ASD/
│       └── [Non-ASD images]
└── valid/
    ├── ASD/
    │   └── [ASD images]
    └── Non_ASD/
        └── [Non-ASD images]
```

### Image Processing
```python
def process_image_data(image_path):
    """
    Processes image data for analysis.
    
    Steps:
    1. Load and resize image
    2. Apply preprocessing
    3. Extract features
    4. Prepare for model input
    
    Returns:
    - Processed image tensor
    """
```

### Behavioral Analysis Features
1. **Eye Contact Analysis**
   - Eye Aspect Ratio (EAR)
   - Gaze direction
   - Eye contact duration
   - Blink frequency

2. **Repetitive Behaviors**
   - Motion patterns
   - Movement periodicity
   - Pattern frequency
   - Duration of patterns

3. **Social Reciprocity**
   - Facial expressions
   - Head orientation
   - Social engagement
   - Response patterns

4. **Gesture Recognition**
   - Pointing gestures
   - Showing behaviors
   - Hand-to-face interactions
   - Gesture frequency

5. **Response Time**
   - Visual stimulus response
   - Reaction latency
   - Response consistency
   - Attention shifts

6. **Joint Attention**
   - Gaze following
   - Shared attention
   - Sustained attention
   - Attention switching

### Data Augmentation
```python
def augment_image_data(image):
    """
    Applies data augmentation to images.
    
    Augmentations:
    1. Random horizontal flip
    2. Brightness adjustment
    3. Contrast adjustment
    4. Rotation
    5. Zoom
    
    Returns:
    - Augmented image
    """
```

### Data Storage
- Raw images: `data/raw/autism_image/`
- Processed images: `data/processed/autism_image/`
- Augmented images: `data/processed/autism_image/augmented/`

## Data Privacy and Ethics

### Privacy Considerations
1. **Screening Data**
   - No personally identifiable information
   - Aggregated demographic data
   - Anonymized responses
   - Secure storage

2. **Image Data**
   - Proper consent obtained
   - No identifiable features
   - Secure storage
   - Access control

### Ethical Guidelines
1. **Data Usage**
   - Research purposes only
   - Professional supervision
   - Regular ethics review
   - Transparent practices

2. **Model Development**
   - Bias monitoring
   - Fair representation
   - Regular validation
   - Clinical oversight

## Data Validation and Metrics (Updated)

- All validation metrics and plots (confusion matrix, ROC curve, accuracy, etc.) are now computed only on the test set, not the full dataset.
- Stratified splitting (stratify=y) is always used for train/test split to ensure class balance.
- If stratified splitting fails or if class imbalance is detected in the train or test set, user-friendly warnings are shown and validation is skipped until more balanced data is collected.

## Future Dataset Plans

### Planned Improvements
1. **Screening Dataset**
   - Additional demographic features
   - Longitudinal data
   - More detailed responses
   - Cultural adaptations

2. **Image Dataset**
   - More diverse subjects
   - Different environments
   - Various age groups
   - Multiple sessions

### Data Collection Strategy
1. **New Features**
   - Audio data
   - Video sequences
   - Environmental context
   - Interaction patterns

2. **Collection Methods**
   - Mobile app integration
   - Remote monitoring
   - Clinical settings
   - Home environments

## Contact Information

For dataset-related inquiries:
- Email: your.email@example.com
- GitHub Issues: [Project Issues](https://github.com/yourusername/autism-detection/issues)

## References

1. Autism Screening Dataset
   - Source: [Kaggle Dataset](https://www.kaggle.com/fabdelja/autism-screening)
   - License: CC0: Public Domain

2. Autism Image Dataset
   - Source: [Kaggle Dataset](https://www.kaggle.com/cihan063/autism-image-data)
   - License: CC BY-NC-SA 4.0

3. Research Papers
   - Autism Screening: [M-CHAT-R/F](https://doi.org/10.1542/peds.2013-1813)
   - Behavioral Analysis: [EAR-based Eye Blink Detection](https://doi.org/10.1109/ICPR.2016.7899650)

## Calibration and User Feedback

- User-labeled data and calibration settings (from the app's sidebar) are used to tune the rule-based logic for behavioral detection.
- User feedback and labeled samples are important for improving both rule-based and ML-based predictions.
- The app provides a calibration mode to adjust thresholds for key metrics, enabling more personalized and accurate detection. 