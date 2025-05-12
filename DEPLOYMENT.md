# Deployment Guide

This document provides detailed instructions for deploying the AI-Driven Early Detection of Autism in Toddlers application.

## Deployment Options

### 1. Local Deployment

#### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment
- Required system packages:
  ```bash
  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install -y \
      python3-dev \
      python3-pip \
      python3-venv \
      libgl1-mesa-glx \
      libglib2.0-0

  # macOS
  brew install python@3.8
  brew install opencv

  # Windows
  # Install Python 3.8+ from python.org
  # Install Visual C++ Build Tools
  ```

#### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/healthcare.git
   cd healthcare
   ```

2. **Set Up Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your settings
   nano .env
   ```

4. **Download Models**
   ```bash
   # Run download script
   python src/utils/download_models.py
   ```

5. **Start Application**
   ```bash
   # Run Streamlit app
   streamlit run src/streamlit_app.py
   ```

### 2. Docker Deployment

#### Prerequisites
- Docker
- Docker Compose
- Git

#### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/healthcare.git
   cd healthcare
   ```

2. **Build Docker Image**
   ```bash
   # Build image
   docker build -t healthcare-app .
   ```

3. **Run Container**
   ```bash
   # Run with GPU support
   docker run --gpus all -p 8501:8501 healthcare-app
   
   # Run without GPU
   docker run -p 8501:8501 healthcare-app
   ```

4. **Using Docker Compose**
   ```bash
   # Start services
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   ```

### 3. Cloud Deployment

#### AWS Deployment

1. **Prerequisites**
   - AWS account
   - AWS CLI installed
   - Docker installed

2. **Setup AWS Resources**
   ```bash
   # Configure AWS CLI
   aws configure
   
   # Create ECR repository
   aws ecr create-repository --repository-name healthcare-app
   
   # Build and push Docker image
   docker build -t healthcare-app .
   docker tag healthcare-app:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/healthcare-app:latest
   docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/healthcare-app:latest
   ```

3. **Deploy to ECS**
   ```bash
   # Create ECS cluster
   aws ecs create-cluster --cluster-name healthcare-cluster
   
   # Create task definition
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   
   # Create service
   aws ecs create-service --cli-input-json file://service-definition.json
   ```

#### Google Cloud Deployment

1. **Prerequisites**
   - Google Cloud account
   - gcloud CLI installed
   - Docker installed

2. **Setup GCP Resources**
   ```bash
   # Configure gcloud
   gcloud init
   
   # Create project
   gcloud projects create healthcare-app
   
   # Enable APIs
   gcloud services enable containerregistry.googleapis.com
   gcloud services enable run.googleapis.com
   ```

3. **Deploy to Cloud Run**
   ```bash
   # Build and push image
   gcloud builds submit --tag gcr.io/$PROJECT_ID/healthcare-app
   
   # Deploy to Cloud Run
   gcloud run deploy healthcare-app \
     --image gcr.io/$PROJECT_ID/healthcare-app \
     --platform managed \
     --allow-unauthenticated
   ```

## Configuration

### Environment Variables

```bash
# Application
APP_ENV=production
DEBUG=False
SECRET_KEY=your-secret-key

# Model Paths
EYE_CONTACT_MODEL_PATH=models/eye_contact_model.h5
BEHAVIOR_MODEL_PATH=models/behavior_model.h5
SOCIAL_MODEL_PATH=models/social_model.h5

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/healthcare

# Storage
STORAGE_BUCKET=healthcare-data
```

### Model Configuration

```python
# config/models.py
MODEL_CONFIG = {
    "eye_contact_detector": {
        "model_path": os.getenv("EYE_CONTACT_MODEL_PATH"),
        "input_size": (224, 224),
        "batch_size": 32,
        "gpu_memory_fraction": 0.8
    },
    # ... other model configs
}
```

## Monitoring and Maintenance

### Health Checks

1. **Application Health**
   ```bash
   # Check application status
   curl http://localhost:8501/health
   
   # Check model status
   curl http://localhost:8501/api/models/status
   ```

2. **Resource Monitoring**
   ```bash
   # Monitor CPU usage
   top -p $(pgrep -f streamlit)
   
   # Monitor GPU usage
   nvidia-smi
   ```

### Logging

1. **Application Logs**
   ```bash
   # View application logs
   tail -f logs/app.log
   
   # View model logs
   tail -f logs/model.log
   ```

2. **Error Monitoring**
   ```bash
   # Check error logs
   grep ERROR logs/app.log
   
   # Monitor exceptions
   tail -f logs/error.log
   ```

## Backup and Recovery

### Data Backup

1. **Model Backup**
   ```bash
   # Backup models
   tar -czf models_backup.tar.gz models/
   
   # Upload to cloud storage
   aws s3 cp models_backup.tar.gz s3://healthcare-backups/
   ```

2. **Database Backup**
   ```bash
   # Backup database
   pg_dump -U user healthcare > healthcare_backup.sql
   
   # Upload to cloud storage
   aws s3 cp healthcare_backup.sql s3://healthcare-backups/
   ```

### Recovery Procedures

1. **Model Recovery**
   ```bash
   # Download from cloud storage
   aws s3 cp s3://healthcare-backups/models_backup.tar.gz .
   
   # Restore models
   tar -xzf models_backup.tar.gz
   ```

2. **Database Recovery**
   ```bash
   # Download from cloud storage
   aws s3 cp s3://healthcare-backups/healthcare_backup.sql .
   
   # Restore database
   psql -U user healthcare < healthcare_backup.sql
   ```

## Security Considerations

### Access Control

1. **User Authentication**
   ```python
   # Enable authentication
   st.secrets["auth_enabled"] = True
   st.secrets["auth_provider"] = "oauth2"
   ```

2. **API Security**
   ```python
   # Enable API authentication
   API_KEY = os.getenv("API_KEY")
   API_SECRET = os.getenv("API_SECRET")
   ```

### Data Protection

1. **Encryption**
   ```python
   # Enable data encryption
   ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
   ENCRYPTION_ALGORITHM = "AES-256-GCM"
   ```

2. **Secure Storage**
   ```python
   # Configure secure storage
   STORAGE_ENCRYPTION = True
   STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
   ```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check model paths
   - Verify GPU availability
   - Check memory usage

2. **Performance Issues**
   - Monitor resource usage
   - Check batch sizes
   - Verify GPU utilization

3. **Deployment Issues**
   - Check environment variables
   - Verify network connectivity
   - Check service status

### Support

For deployment issues:
- Open an issue on GitHub
- Contact the development team
- Check the troubleshooting guide
- Join our community forum

## Updates and Maintenance

### Version Updates

1. **Application Updates**
   ```bash
   # Pull latest changes
   git pull origin main
   
   # Update dependencies
   pip install -r requirements.txt
   
   # Restart application
   systemctl restart healthcare-app
   ```

2. **Model Updates**
   ```bash
   # Download new models
   python src/utils/download_models.py --update
   
   # Verify model performance
   python src/utils/validate_models.py
   ```

### Regular Maintenance

1. **Daily Tasks**
   - Check application logs
   - Monitor resource usage
   - Verify backup status

2. **Weekly Tasks**
   - Review error logs
   - Check model performance
   - Update dependencies

3. **Monthly Tasks**
   - Full system backup
   - Performance optimization
   - Security updates

## Contact

For deployment support:
- Open an issue on GitHub
- Contact the development team
- Join our community forum
- Check our documentation 