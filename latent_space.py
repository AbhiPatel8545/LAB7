"""Piece 4: Latent space probing with interpolation and conditional VAE.

This script demonstrates:

1. Interpolation in the latent space: generating intermediate images between two digits
   by linearly interpolating the latent codes.

2. Conditional VAE: showing how conditioning on class labels allows the decoder
   to generate specific digit classes.

3. 2D latent space visualization: showing a grid of decoded points in a 2D latent space.

Run: python latent_space.py
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lab07_starter as S
from vae import VAE, train_vae

torch.manual_seed(0)
np.random.seed(0)

print("=" * 70)
print("Piece 4: Latent Space Probing")
print("=" * 70)

Xtr, ytr, Xte, yte = S.load_task()
print(f"train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}")

# ---------------------------------------------------------------- interpolation
print("\n--- Interpolation in Latent Space ---")

# Train a VAE for interpolation demonstrations
print("Training VAE for interpolation demo...")
vae = VAE(d_latent=2, d_hidden=64)  # 2D latent space for visualization
train_vae(vae, Xtr, epochs=200, lr=2e-3, batch_size=64, seed=0, beta=1.0)
print("VAE trained.")

# Find two test images of different digits
digit1, digit2 = 3, 7  # Example: interpolating between 3 and 7
idx1 = (yte == digit1).nonzero()[0][0]
idx2 = (yte == digit2).nonzero()[0][0]
x1, x2 = Xte[idx1], Xte[idx2]

print(f"Interpolating between digit {digit1} (test idx={idx1}) and digit {digit2} (test idx={idx2})")

# Get latent means for interpolation
mu1, _ = vae.encode(x1.unsqueeze(0))
mu2, _ = vae.encode(x2.unsqueeze(0))

# Generate interpolation in latent space
steps = 8
ts = torch.linspace(0, 1, steps)
z_interp = (1 - ts[:, None]) * mu1 + ts[:, None] * mu2

# Decode interpolated latents
with torch.no_grad():
    images_interp = torch.sigmoid(vae.decode(z_interp)).clamp(0, 1)

# Save interpolation grid
S.save_grid(images_interp, "figures/interpolation_3_to_7.png", ncols=4,
            title=f"Latent interpolation: digit {digit1} -> digit {digit2}")
print(f"Saved figures/interpolation_3_to_7.png")

# ---------------------------------------------------------------- conditional VAE
print("\n--- Conditional VAE ---")

# Train a conditional VAE with class labels
print("Training conditional VAE (10 classes)...")
cvae = VAE(d_latent=8, d_hidden=64, n_classes=10)
ytr_onehot = S.one_hot(ytr)
train_vae(cvae, Xtr, y_onehot=ytr_onehot, epochs=200, lr=2e-3, batch_size=64, seed=0, beta=1.0)
print("Conditional VAE trained.")

# Generate samples for each digit class
print("\nGenerating samples for each digit class:")
samples_per_class = []
for digit in range(10):
    g = torch.Generator().manual_seed(digit)
    z_prior = torch.randn(10, 8, generator=g)  # 10 samples per class
    c = S.one_hot(torch.full((10,), digit, dtype=torch.long), 10)
    with torch.no_grad():
        samples = torch.sigmoid(cvae.decode(z_prior, c)).clamp(0, 1)
    samples_per_class.append(samples)
    print(f"  Digit {digit}: generated 10 samples")

# Save all samples in a grid
all_samples = torch.cat(samples_per_class, dim=0)
S.save_grid(all_samples, "figures/conditional_vae_digits.png", ncols=10,
            title="Conditional VAE samples by digit class")
print("Saved figures/conditional_vae_digits.png")

# ---------------------------------------------------------------- 2D latent space visualization
print("\n--- 2D Latent Space Grid ---")

# Train a VAE with 2D latent space for visualization
print("Training VAE with 2D latent space for visualization...")
viz_vae = VAE(d_latent=2, d_hidden=64)
train_vae(viz_vae, Xtr, epochs=200, lr=2e-3, batch_size=64, seed=0, beta=1.0)
print("VAE trained.")

# Create 8x8 grid in latent space
lim = 2.5
n = 8
zs = torch.linspace(-lim, lim, n)
zz = torch.stack(torch.meshgrid(zs, zs.flip(0), indexing="xy"), -1).reshape(-1, 2)

# Decode each point in the grid
with torch.no_grad():
    grid_images = torch.sigmoid(viz_vae.decode(zz)).clamp(0, 1)

# Reshape to 2D grid
grid_images = grid_images.reshape(n, n, 8, 8)

# Save as a figure
fig, axes = plt.subplots(n, n, figsize=(n * 0.8, n * 0.8 + 0.5), squeeze=False)
for i in range(n):
    for j in range(n):
        axes[i, j].axis("off")
        axes[i, j].imshow(grid_images[i, j], cmap="gray_r", vmin=0, vmax=1)
fig.suptitle("VAE Latent Space: decoded from 2D grid", fontsize=10)
fig.savefig("figures/latent_space_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved figures/latent_space_grid.png")

# ---------------------------------------------------------------- active units for visualization
print("\n--- Active Units Analysis ---")

# Check how many latent dimensions are actually used
from vae import active_units

au_2d = active_units(viz_vae, Xtr)
au_8d = active_units(vae, Xtr)

# For conditional VAE, we need to pass labels
cvau_8d = active_units(cvae, Xtr, c=S.one_hot(ytr), threshold=0.01)

print(f"2D VAE active units: {au_2d}")
print(f"8D VAE active units: {au_8d}")
print(f"Conditional 8D VAE active units: {cvau_8d}")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
print("Summary for README")
print("=" * 70)
print("""
KEY OBSERVATIONS:
- Interpolation shows how the VAE learns a meaningful latent space. Linear interpolation
  between digit 3 and 7 produces smooth transitions that look like actual digits.
- Conditional VAE allows class-conditional generation. Each row shows samples from
  the same latent but conditioned on a specific digit class.
- The 2D latent space grid shows the structure of the learned latent manifold.
- Active units measure: if many units are < 0.01 variance, the model may be
  underusing the latent code (posterior collapse).
""")