import kagglehub
import shutil
import os

print("Downloading Yelp dataset from Kaggle...")
path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
print("Downloaded to:", path)

dest = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(dest, exist_ok=True)

needed = [
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_business.json",
]
for fname in needed:
    src = os.path.join(path, fname)
    if os.path.exists(src):
        print(f"Copying {fname} -> data/")
        shutil.copy2(src, os.path.join(dest, fname))
    else:
        print(f"WARNING: {fname} not found at {src}")

print("Done. Files are in the data/ folder.")
