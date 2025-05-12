# Dataset Documentation

This document provides detailed information about the datasets used in the AI-Driven Early Detection of Autism in Toddlers project.

## Dataset Overview

### Autism Screening Dataset

#### Source
- Location: `data/raw/screening/Autism_Screening_Data_Combined.csv`
- Format: CSV
- Size: ~2MB
- Records: 6,076

#### Features
1. **Demographic Information**
   - Age (numeric)
   - Sex (m/f)

2. **Screening Questions (A1-A10)**
   - Binary responses (0/1) to 10 autism screening questions
   - Each question assesses different aspects of behavior and development

3. **Medical History**
   - Jaundice at birth (yes/no)
   - Family history of ASD (yes/no)

4. **Target Variable**
   - Class: ASD diagnosis (YES/NO)

#### Data Structure
```
A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Age,Sex,Jauundice,Family_ASD,Class
1,1,0,1,0,0,1,1,0,0,15,m,no,no,NO
0,1,1,1,0,1,1,0,1,0,15,m,no,no,NO
...
```

## Data Preprocessing

### 1. Data Cleaning

#### Steps
1. **Handle Missing Values**
   ```python
   def handle_missing_values(df):
       """
       Handle missing values in the dataset
       Args:
           df: Input DataFrame
       Returns:
           Cleaned DataFrame
       """
   ```

2. **Feature Encoding**
   ```python
   def encode_features(df):
       """
       Encode categorical features
       Args:
           df: Input DataFrame
       Returns:
           Encoded DataFrame
       """
   ```

3. **Data Validation**
   ```python
   def validate_data(df):
       """
       Validate data quality
       Args:
           df: Input DataFrame
       Returns:
           Validation results
       """
   ```

### 2. Feature Engineering

#### Steps
1. **Create Composite Scores**
   ```python
   def create_composite_scores(df):
       """
       Create composite scores from screening questions
       Args:
           df: Input DataFrame
       Returns:
           DataFrame with new features
       """
   ```

2. **Age Group Categorization**
   ```python
   def categorize_age(df):
       """
       Categorize age into groups
       Args:
           df: Input DataFrame
       Returns:
           DataFrame with age categories
       """
   ```

3. **Feature Selection**
   ```python
   def select_features(df):
       """
       Select relevant features
       Args:
           df: Input DataFrame
       Returns:
           DataFrame with selected features
       """
   ```

## Data Validation

### 1. Quality Checks

#### Data Validation
```python
def validate_dataset(df):
    """
    Validate dataset quality
    Args:
        df: Input DataFrame
    Returns:
        Validation result
    """
    checks = {
        "missing_values": check_missing_values(df),
        "data_types": check_data_types(df),
        "value_ranges": check_value_ranges(df),
        "class_balance": check_class_balance(df)
    }
    return all(checks.values())
```

### 2. Data Cleaning

#### Data Cleaning Steps
1. **Remove Invalid Records**
   - Missing values
   - Outliers
   - Inconsistent data

2. **Standardize Format**
   - Convert data types
   - Normalize values
   - Handle categorical variables

## Data Splitting

### 1. Train-Test Split
```python
def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    Args:
        df: Input DataFrame
        test_size: Proportion of test set
        random_state: Random seed
    Returns:
        Train and test sets
    """
```

### 2. Cross-Validation
```python
def create_cv_splits(df, n_splits=5):
    """
    Create cross-validation splits
    Args:
        df: Input DataFrame
        n_splits: Number of splits
    Returns:
        Cross-validation splits
    """
```

## Data Storage

### 1. Processed Data
- Location: `data/processed/`
- Format: CSV
- Files:
  - `train.csv`: Training data
  - `test.csv`: Test data
  - `validation.csv`: Validation data

### 2. Feature Engineering
- Location: `data/features/`
- Format: CSV
- Files:
  - `engineered_features.csv`: Engineered features
  - `feature_importance.csv`: Feature importance scores

## Usage Guidelines

### 1. Data Access
```python
import pandas as pd

# Load raw data
df = pd.read_csv('data/raw/screening/Autism_Screening_Data_Combined.csv')

# Load processed data
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')
```

### 2. Data Processing
```python
from src.data.preprocessing import preprocess_data

# Preprocess data
processed_df = preprocess_data(df)
```

### 3. Feature Engineering
```python
from src.data.feature_engineering import engineer_features

# Create features
featured_df = engineer_features(processed_df)
```

## Data Privacy and Ethics

### 1. Privacy Considerations
- All data is anonymized
- No personal identifiers are included
- Data is used only for research purposes

### 2. Ethical Guidelines
- Follow medical research ethics
- Maintain data confidentiality
- Use data responsibly

## Future Dataset Plans

### 1. Planned Additions
- Additional screening data from other sources
- Longitudinal data for tracking development
- More diverse demographic representation

### 2. Data Collection
- Partner with healthcare providers
- Collaborate with research institutions
- Follow ethical guidelines for data collection

## Contact

For dataset-related questions:
- Open an issue on GitHub
- Contact the development team
- Join our community forum 