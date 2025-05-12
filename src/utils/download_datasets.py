import os
import kaggle
from pathlib import Path

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks/1_data_collection',
        'notebooks/2_feature_engineering',
        'notebooks/3_model_development',
        'src/models',
        'src/utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def download_datasets():
    """Download datasets from Kaggle."""
    datasets = {
        'fabdelja/autism-screening': 'data/raw/autism_screening',
        'cihan063/autism-image-data': 'data/raw/autism_image',
        'cihan063/autism-videos-dataset': 'data/raw/autism_videos'
    }
    
    for dataset, path in datasets.items():
        print(f"Downloading {dataset}...")
        try:
            kaggle.api.dataset_download_files(
                dataset,
                path=path,
                unzip=True
            )
            print(f"Successfully downloaded {dataset}")
        except Exception as e:
            print(f"Error downloading {dataset}: {str(e)}")

def main():
    """Main function to set up the project structure and download datasets."""
    print("Creating project directories...")
    create_directories()
    
    print("\nDownloading datasets from Kaggle...")
    download_datasets()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main() 