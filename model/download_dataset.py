# model/download_dataset.py

import kagglehub
from pathlib import Path
import shutil

# Step 1: Download using kagglehub
print("📥 Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("ma7555/cat-breeds-dataset")
print(f"✅ Dataset downloaded to: {path}")

# Step 2: Copy to your project data folder
target_dir = Path(__file__).parent / "data" / "cat-breeds-dataset"
print(f"📂 Copying dataset to project directory: {target_dir}")
shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("✅ Dataset setup complete.")
