import os
import requests
import zipfile
import io

def download_dataset():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URL for the dataset
    url = "https://raw.githubusercontent.com/anshtanwar/Credit-Risk-Prediction/main/credit_risk.csv"
    
    try:
        # Download the file
        print("Downloading dataset...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the file
        file_path = os.path.join('data', 'credit_risk.csv')
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Dataset downloaded successfully to {file_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

if __name__ == "__main__":
    download_dataset() 