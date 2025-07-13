import torch
from PIL import Image
import cv2
import numpy as np
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

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

def process_image(img_path, mask_path=None, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"fail: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mask_path is not None and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)
        img[mask == 0] = 0

    input_image = Image.fromarray(img)
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
        depth = prediction.squeeze().cpu().numpy()

    output_name = os.path.splitext(os.path.basename(img_path))[0]
    np.save(os.path.join(output_dir, f"{output_name}_depth.npy"), depth)

    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    gray_path = os.path.join(output_dir, f"{output_name}_depth.png")
    color_path = os.path.join(output_dir, f"{output_name}_depth_color.png")

    cv2.imwrite(gray_path, depth_norm)
    cv2.imwrite(color_path, cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO))

    return gray_path, color_path, f"{output_name}_depth.npy"

img_path = "../data/images/withBackground/Blackbird/(2).jpg"
mask_path = "../data/images/withBackground/Blackbird/(2)_mask.png"
process_image(img_path, mask_path)
