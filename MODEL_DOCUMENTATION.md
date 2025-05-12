# Model Documentation

This document provides detailed information about the machine learning models used in the AI-Driven Early Detection of Autism in Toddlers project.

## Model Overview

### Autism Screening Model

#### Purpose
- Early detection of autism spectrum disorder (ASD) using screening questionnaire data
- Binary classification (ASD/Non-ASD)
- Support for healthcare professionals in initial screening

#### Model Architecture
1. **Input Layer**
   - 14 features:
     - 10 binary screening questions (A1-A10)
     - Age (numeric)
     - Sex (binary)
     - Jaundice history (binary)
     - Family ASD history (binary)

2. **Model Type**
   - Primary: Random Forest Classifier
   - Alternative: XGBoost Classifier
   - Baseline: Logistic Regression

3. **Output Layer**
   - Binary classification (YES/NO)
   - Probability scores for ASD likelihood

## Model Training

### 1. Data Preparation

#### Feature Engineering
```python
def prepare_features(df):
    """
    Prepare features for model training
    Args:
        df: Input DataFrame
    Returns:
        Processed features
    """
    # Handle categorical variables
    df['Sex'] = df['Sex'].map({'m': 0, 'f': 1})
    df['Jaundice'] = df['Jaundice'].map({'no': 0, 'yes': 1})
    df['Family_ASD'] = df['Family_ASD'].map({'no': 0, 'yes': 1})
    df['Class'] = df['Class'].map({'NO': 0, 'YES': 1})
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 2, 5, 12, 18, 100],
                            labels=['Infant', 'Toddler', 'Child', 'Teen', 'Adult'])
    
    return df
```

#### Data Splitting
```python
def split_data(df, test_size=0.2, val_size=0.1):
    """
    Split data into train, validation, and test sets
    Args:
        df: Input DataFrame
        test_size: Test set proportion
        val_size: Validation set proportion
    Returns:
        Train, validation, and test sets
    """
    # Stratified split to maintain class distribution
    train_df, temp_df = train_test_split(df, 
                                       test_size=test_size + val_size,
                                       stratify=df['Class'],
                                       random_state=42)
    
    val_df, test_df = train_test_split(temp_df,
                                      test_size=test_size/(test_size + val_size),
                                      stratify=temp_df['Class'],
                                      random_state=42)
    
    return train_df, val_df, test_df
```

### 2. Model Training

#### Training Process
```python
def train_model(X_train, y_train, model_type='rf'):
    """
    Train the model
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Model type ('rf', 'xgb', or 'lr')
    Returns:
        Trained model
    """
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    elif model_type == 'xgb':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    else:
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model
```

#### Hyperparameter Tuning
```python
def tune_hyperparameters(X_train, y_train, model_type='rf'):
    """
    Tune model hyperparameters
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Model type
    Returns:
        Best parameters
    """
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'xgb':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [1000, 2000]
        }
    
    grid_search = GridSearchCV(
        estimator=get_model(model_type),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
```

### 3. Model Evaluation

#### Evaluation Metrics
```python
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    Returns:
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics
```

#### Cross-Validation
```python
def cross_validate_model(X, y, model_type='rf', n_splits=5):
    """
    Perform cross-validation
    Args:
        X: Features
        y: Labels
        model_type: Model type
        n_splits: Number of splits
    Returns:
        Cross-validation scores
    """
    model = get_model(model_type)
    cv_scores = cross_val_score(
        model, X, y,
        cv=n_splits,
        scoring='f1',
        n_jobs=-1
    )
    
    return {
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std()
    }
```

## Model Deployment

### 1. Model Serialization

#### Save Model
```python
def save_model(model, model_type, version):
    """
    Save trained model
    Args:
        model: Trained model
        model_type: Model type
        version: Model version
    """
    model_path = f'models/{model_type}_v{version}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
```

#### Load Model
```python
def load_model(model_type, version):
    """
    Load saved model
    Args:
        model_type: Model type
        version: Model version
    Returns:
        Loaded model
    """
    model_path = f'models/{model_type}_v{version}.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
```

### 2. API Integration

#### Prediction Endpoint
```python
def predict_asd(data):
    """
    Make ASD prediction
    Args:
        data: Input data dictionary
    Returns:
        Prediction and probability
    """
    # Load model
    model = load_model('rf', '1.0')
    
    # Preprocess input
    features = preprocess_input(data)
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    
    return {
        'prediction': 'YES' if prediction == 1 else 'NO',
        'probability': float(probability)
    }
```

## Model Monitoring

### 1. Performance Monitoring

#### Metrics Tracking
```python
def track_metrics(predictions, actuals):
    """
    Track model performance metrics
    Args:
        predictions: Model predictions
        actuals: Actual labels
    Returns:
        Updated metrics
    """
    metrics = {
        'accuracy': accuracy_score(actuals, predictions),
        'f1': f1_score(actuals, predictions),
        'timestamp': datetime.now()
    }
    
    # Save metrics
    save_metrics(metrics)
    return metrics
```

#### Drift Detection
```python
def detect_drift(new_data, reference_data):
    """
    Detect data drift
    Args:
        new_data: New data
        reference_data: Reference data
    Returns:
        Drift metrics
    """
    drift_metrics = {
        'feature_drift': calculate_feature_drift(new_data, reference_data),
        'distribution_drift': calculate_distribution_drift(new_data, reference_data),
        'timestamp': datetime.now()
    }
    
    return drift_metrics
```

### 2. Model Updates

#### Retraining Triggers
1. **Performance Degradation**
   - Accuracy below threshold
   - F1 score below threshold
   - High false positive/negative rates

2. **Data Drift**
   - Significant feature drift
   - Distribution changes
   - New patterns detected

3. **Regular Updates**
   - Monthly retraining
   - New data available
   - Model improvements

## Model Limitations

### 1. Current Limitations
- Binary classification only
- Based on screening questionnaire
- Not a diagnostic tool
- Requires professional interpretation

### 2. Future Improvements
- Additional features
- Multi-class classification
- Integration with clinical data
- Real-time monitoring

## Usage Guidelines

### 1. Model Usage
```python
from src.models.predict import predict_asd

# Example input
data = {
    'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1,
    'A6': 0, 'A7': 1, 'A8': 0, 'A9': 1, 'A10': 0,
    'Age': 5, 'Sex': 'm', 'Jaundice': 'no', 'Family_ASD': 'no'
}

# Get prediction
result = predict_asd(data)
print(f"ASD Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
```

### 2. Best Practices
1. **Input Validation**
   - Check data types
   - Validate ranges
   - Handle missing values

2. **Interpretation**
   - Use as screening tool only
   - Consider clinical context
   - Review with professionals

3. **Documentation**
   - Record predictions
   - Track model version
   - Monitor performance

## Contact

For model-related questions:
- Open an issue on GitHub
- Contact the development team
- Join our community forum 