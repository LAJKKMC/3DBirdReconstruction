# models/visualize.py

import os
import torch
import imageio
import numpy as np
from nerf import TinyNeRF
from rays import get_rays

def get_spiral_poses(radius=1.0, n_views=60):

    poses = []
    for theta in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
        c2w = np.eye(4)
        c2w[:3, 3] = np.array([radius * np.sin(theta), 0.0, radius * np.cos(theta)])
        forward = -c2w[:3, 3] / np.linalg.norm(c2w[:3, 3])
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)
        c2w[:3, 0] = right / np.linalg.norm(right)
        c2w[:3, 1] = up / np.linalg.norm(up)
        c2w[:3, 2] = forward
        poses.append(c2w)
    return poses

def main():
    H, W = 400, 400
    focal = 0.5 * W
    n_samples = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, '..', 'renders_0'))
    os.makedirs(output_dir, exist_ok=True)

    model = TinyNeRF().to(device)
    ckpt_path = os.path.normpath(os.path.join(script_dir, '..', 'models/nerf_model.pth'))
    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    poses = get_spiral_poses(radius=2.0, n_views=60)

    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)

    for i, c2w in enumerate(poses):
        c2w_tensor = torch.from_numpy(c2w).float().to(device)
        rays_o, rays_d = get_rays(H, W, focal, c2w_tensor)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
        pts_flat = pts.reshape(-1, 3)

        with torch.no_grad():
            out = model(pts_flat)  # [..., 4]
        density = out[..., 0].reshape(-1, n_samples)
        color   = out[..., 1:].reshape(-1, n_samples, 3)

        weights = torch.softmax(density, dim=-1)
        rgb     = torch.sum(weights[..., None] * color, dim=1)
        depth   = torch.sum(weights * t_vals[None, :], dim=1)

        img       = rgb.reshape(H, W, 3).cpu().numpy()
        depth_map = depth.reshape(H, W).cpu().numpy()

        img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), img_uint8)

        d_norm    = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_uint8 = (d_norm * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"depth_{i:03d}.png"), depth_uint8)

    print(f"save to：{output_dir}")

if __name__ == "__main__":
    main()
