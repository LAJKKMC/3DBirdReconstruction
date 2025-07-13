import torch

def get_rays(H, W, focal, c2w):
    device = c2w.device  # 获取 transform 的设备
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
                          torch.linspace(0, H-1, H, device=device), indexing='ij')
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
