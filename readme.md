### Bird3D: 3D Reconstruction of Birds from Images

#### Overview
Bird3D aims to reconstruct detailed 3D models of birds from single or multiple images using a combination of depth estimation and Neural Radiance Fields (NeRF). The project leverages depth maps to improve NeRF training and enhance the quality of the resulting 3D reconstructions.

#### Project Pipeline

##### 1. Data Preparation
- **Original Images**: Stored in `withBackground/`, containing multiple species of birds with several images per species.
- **Mask Images**: Stored in `withoutBackground/`, used to remove the background and isolate birds.
- **Depth Maps**: Generated using a depth estimation model (MiDaS) and saved as 8-bit grayscale PNGs in `depth_maps/`.

##### 2. Depth Estimation
- Model: **DPT-Hybrid (MiDaS)**.
- Process:
  - Original image + corresponding mask → background removed.
  - Input the masked image into MiDaS → output depth map.
- Output: 8-bit PNG depth maps.

##### 3. NeRF Data Formatting
- Select the **first 16 images of each bird species** for training.
- Organize data:
  - `nerf_data_mixed/images/`: Selected images.
  - `nerf_data_mixed/depths/`: Corresponding depth maps.
  - `nerf_data_mixed/transforms.json`: Camera parameters and transform matrices, including:
    - `camera_angle_x`
    - `fl_x`, `fl_y`: Focal lengths.
    - `cx`, `cy`: Principal points.
    - `transform_matrix`: Camera pose for each image.

##### 4. Model Training

###### Custom TinyNeRF with Depth Supervision
- Implemented in PyTorch.
- Key files:
  - `train.py`: Training loop with depth supervision.
  - `nerf.py`: TinyNeRF architecture.
  - `dataset.py`: Loads images, depth maps, and poses.
  - `losses.py`: RGB loss and depth loss functions.
  - `rays.py`: Ray generation for NeRF rendering.
- Loss Functions:
  - RGB loss: Compares predicted RGB to ground truth images.
  - Depth loss: Penalizes deviation between predicted and actual depth maps.

###### Nerfstudio Training (Baseline Comparison)
- Framework: **Nerfstudio**.
- Command:
  ```bash
  ns-train nerfacto --data ../data/nerf_data_mixed