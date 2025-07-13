import os
import shutil
import pandas as pd
import json
from collections import defaultdict

csv_path = "../data/images/birds.csv"
image_base = "../data/images"
depth_base = "../data/depth_maps"
output_dir = "../data/nerf_data_mixed"
image_out = os.path.join(output_dir, "images")
depth_out = os.path.join(output_dir, "depths")
os.makedirs(image_out, exist_ok=True)
os.makedirs(depth_out, exist_ok=True)

df = pd.read_csv(csv_path)
df = df[df['filepaths'].str.startswith("withBackground")]

grouped = defaultdict(list)
for _, row in df.iterrows():
    parts = row['filepaths'].split('/')
    if len(parts) >= 3:
        species = parts[1]
        grouped[species].append(row['filepaths'])

top16_paths = []
for species, paths in grouped.items():
    top16_paths.extend(paths[:16])

frames = []
for i, rel_path in enumerate(top16_paths):
    filename_base = rel_path.replace("/", "_").replace("\\", "_").replace(".jpg", "")
    depthname = filename_base + "_depth.png"

    src_img = os.path.join(image_base, rel_path)
    src_depth = os.path.join(depth_base, depthname)
    species_dir = rel_path.split('/')[1]
    img_filename = os.path.basename(rel_path)

    dst_img_dir = os.path.join(image_out, species_dir)
    dst_depth_dir = os.path.join(depth_out, species_dir)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_depth_dir, exist_ok=True)

    dst_img = os.path.join(dst_img_dir, img_filename)
    dst_depth = os.path.join(dst_depth_dir, depthname)

    if os.path.exists(src_img):
        shutil.copyfile(src_img, dst_img)
    else:
        print(f"[SKIP] Missing image: {rel_path}")
        continue

    if os.path.exists(src_depth):
        shutil.copyfile(src_depth, dst_depth)
        depth_path = f"./depths/{species_dir}/{depthname}"
    else:
        print(f"[WARNING] Missing depth: {depthname}")
        depth_path = None

    frame = {
        "file_path": f"./images/{species_dir}/{img_filename}",
        "transform_matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    }
    if depth_path:
        frame["depth_path"] = depth_path

    frames.append(frame)

transforms = {
    "camera_angle_x": 0.6911112070083618,
    "frames": frames
}
with open(os.path.join(output_dir, "transforms.json"), "w") as f:
    json.dump(transforms, f, indent=4)

print(f"[DONE] NeRF mixed training data prepared at: {output_dir}")
