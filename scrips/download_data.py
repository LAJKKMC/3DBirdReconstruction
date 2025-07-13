import os
import zipfile

target_dir = r"..\data\images"
os.makedirs(target_dir, exist_ok=True)

os.system(f'kaggle datasets download -d davemahony/20-uk-garden-birds -p "{target_dir}"')

zip_path = os.path.join(target_dir, "20-uk-garden-birds.zip")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(zip_path)
