"""
Script to download the AI Job Market Insights dataset from Kaggle.
"""
import kagglehub
import shutil
from pathlib import Path
import os

DATA_DIR = Path(__file__).parent.parent / "data"

def download_dataset():
    print("Downloading dataset from Kaggle...")
    # Download the dataset using kagglehub
    path = kagglehub.dataset_download("kapturovalexander/ai-job-market-insights")
    print("Path to dataset files:", path)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Move the downloaded CSV to data/ai_jobs_global.csv
    # Look for the main CSV file in the downloaded path
    downloaded_files = list(Path(path).glob("*.csv"))
    if not downloaded_files:
        print("No CSV found in downloaded dataset.")
        return
    
    # Kapturov's dataset usually has a file like ai_job_market_insights.csv
    # We rename it to ai_jobs_global.csv for our pipeline
    src_file = downloaded_files[0]
    dest_file = DATA_DIR / "ai_jobs_global.csv"
    
    shutil.copy2(src_file, dest_file)
    print(f"Dataset successfully downloaded and saved to: {dest_file}")

if __name__ == "__main__":
    download_dataset()
