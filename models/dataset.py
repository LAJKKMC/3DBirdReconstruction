import torch
import os
import json
import cv2
import numpy as np

class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "transforms.json")) as f:
            meta = json.load(f)
        self.frames = meta['frames']

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = cv2.imread(os.path.join(self.data_dir, frame['file_path'][2:]))[..., ::-1] / 255.0  # RGB
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        depth = None
        if 'depth_path' in frame:
            depth = cv2.imread(os.path.join(self.data_dir, frame['depth_path'][2:]), cv2.IMREAD_UNCHANGED)
            depth = torch.from_numpy(depth).float() / 255.0
        transform = torch.tensor(frame['transform_matrix']).float()
        return img, depth, transform