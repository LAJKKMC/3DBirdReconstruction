import torch
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True, skip_validation=True)
model_path = os.path.join("../models", "dpt_hybrid_384.pt")
midas.load_state_dict(torch.load(model_path, map_location=device))
midas.to(device)
midas.eval()

transform = Compose([
    Resize(384),
    ToTensor(),
    Normalize(mean=0.5, std=0.5),
])

csv_path = "../data/images/birds.csv"
df = pd.read_csv(csv_path)

output_dir = "../data/depth_maps"
os.makedirs(output_dir, exist_ok=True)

base_image_dir = "../data/images"
for rel_path in tqdm(df['filepaths'], desc="Processing Images"):
    full_path = os.path.join(base_image_dir, rel_path)
    img = cv2.imread(full_path)
    if img is None:
        print(f"❌ fail: {full_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
        depth = prediction.squeeze().cpu().numpy()

    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    filename_base = rel_path.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".jpg", "")
    save_path = os.path.join(output_dir, f"{filename_base}_depth.png")
    cv2.imwrite(save_path, depth_norm)

print("✅ depthmap to : ../data/depth_maps/")
