import torch
from torch.utils.data import DataLoader
from nerf import TinyNeRF
from dataset import BirdDataset
from losses import rgb_loss_fn, depth_loss_fn
from rays import get_rays
import matplotlib.pyplot as plt

# Hyperparameters
batch_size    = 1
lr            = 1e-3
lambda_depth  = 0.1
num_epochs    = 10

# Dataset
dataset    = BirdDataset("../data/nerf_data_mixed")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Two models: with and without depth supervision
model_with_depth = TinyNeRF().cuda()
model_no_depth   = TinyNeRF().cuda()

optimizer_with_depth = torch.optim.Adam(model_with_depth.parameters(), lr=lr)
optimizer_no_depth   = torch.optim.Adam(model_no_depth.parameters(), lr=lr)

losses_with_depth = []
losses_no_depth   = []

# Training loop
for epoch in range(1, num_epochs + 1):
    epoch_loss_with_depth = 0.0
    epoch_loss_no_depth   = 0.0

    for imgs, depths, transforms in dataloader:
        imgs    = imgs.cuda()
        depths  = depths.cuda() if depths is not None else None

        H, W    = imgs.shape[2], imgs.shape[3]
        focal   = 0.5 * W
        rays_o, rays_d = get_rays(H, W, focal, transforms[0].cuda())

        t_vals  = torch.linspace(0, 1, 64).cuda()
        positions = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
        positions = positions.reshape(-1, 3)

        # Train model with depth supervision
        outputs = model_with_depth(positions)
        density = outputs[..., 0].reshape(rays_o.shape[0], -1)
        color   = outputs[..., 1:].reshape(rays_o.shape[0], -1, 3)

        weights   = torch.softmax(density, dim=-1)
        pred_rgb  = torch.sum(weights[..., None] * color, dim=1)
        pred_depth= torch.sum(weights * t_vals[None, :], dim=1)

        gt_rgb = imgs.permute(0, 2, 3, 1).reshape(-1, 3)
        if depths is not None:
            gt_depth = depths.reshape(-1)[:pred_depth.shape[0]]
        else:
            gt_depth = pred_depth.detach()

        loss_with_depth = rgb_loss_fn(pred_rgb, gt_rgb)
        if depths is not None:
            loss_with_depth += lambda_depth * depth_loss_fn(pred_depth, gt_depth)

        optimizer_with_depth.zero_grad()
        loss_with_depth.backward()
        optimizer_with_depth.step()

        epoch_loss_with_depth += loss_with_depth.item()

        # Train model without depth supervision
        outputs_no_depth = model_no_depth(positions)
        density_no = outputs_no_depth[..., 0].reshape(rays_o.shape[0], -1)
        color_no   = outputs_no_depth[..., 1:].reshape(rays_o.shape[0], -1, 3)

        weights_no = torch.softmax(density_no, dim=-1)
        pred_rgb_no = torch.sum(weights_no[..., None] * color_no, dim=1)

        loss_no_depth = rgb_loss_fn(pred_rgb_no, gt_rgb)

        optimizer_no_depth.zero_grad()
        loss_no_depth.backward()
        optimizer_no_depth.step()

        epoch_loss_no_depth += loss_no_depth.item()

    avg_loss_with_depth = epoch_loss_with_depth / len(dataloader)
    avg_loss_no_depth   = epoch_loss_no_depth / len(dataloader)

    losses_with_depth.append(avg_loss_with_depth)
    losses_no_depth.append(avg_loss_no_depth)

    print(f"Epoch {epoch:2d}/{num_epochs}, Loss with depth: {avg_loss_with_depth:.6f}, Loss without depth: {avg_loss_no_depth:.6f}")

# Save models
torch.save(model_with_depth.state_dict(), "nerf_model_with_depth_1.pth")
torch.save(model_no_depth.state_dict(),   "nerf_model_no_depth_1.pth")
print(" Models saved.")

# Plot comparison
plt.figure()
plt.plot(range(1, num_epochs + 1), losses_with_depth, marker='o', label='With Depth Supervision')
plt.plot(range(1, num_epochs + 1), losses_no_depth, marker='x', label='Without Depth Supervision')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve Comparison')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve_comparison.png")
plt.show()
