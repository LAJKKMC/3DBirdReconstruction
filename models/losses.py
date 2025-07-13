import torch

def rgb_loss_fn(pred_rgb, gt_rgb):
    return torch.mean((pred_rgb - gt_rgb) ** 2)

def depth_loss_fn(pred_depth, gt_depth, mask=None):
    if mask is not None:
        return torch.mean(torch.abs(pred_depth - gt_depth)[mask])
    else:
        return torch.mean(torch.abs(pred_depth - gt_depth))