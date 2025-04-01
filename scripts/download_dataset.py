import kagglehub
import os

# Download dataset
dataset_path = kagglehub.dataset_download("kewalkishang/iphonexreview")

print(f"Dataset downloaded to: {dataset_path}")

# Move dataset to the `data/` folder
os.makedirs("data", exist_ok=True)
for file in os.listdir(dataset_path):
    os.rename(os.path.join(dataset_path, file), os.path.join("data", file))

print("Dataset moved to the data/ directory.")
