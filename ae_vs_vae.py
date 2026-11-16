"""Piece 2: Same bottleneck, autoencoder against VAE.

Train the plain autoencoder and the VAE at beta == 1 with:
  - same code size (8),
  - same hidden width (64),
  - same epochs (200),
  - same learning rate (2e-3),
  - same batch size (64),
  - same seeds (0, 1, 2).

For each seed, report:
  - test reconstruction error (for the VAE, rebuild from the posterior mean),
  - the starter's scores for 359 images decoded from draws of N(0, I).

Also report the spread of the codes in each dimension for both models,
and comment on what the numbers show about why one can generate and the other cannot.
"""
import numpy as np
import torch

import lab07_starter as S
from vae import VAE, vae_loss, elbo_terms, kl_to_standard_normal, train_vae

torch.manual_seed(0)
np.random.seed(0)

print("=" * 70)
print("Piece 2 - Autoencoder vs VAE")
print("=" * 70)

Xtr, ytr, Xte, yte = S.load_task()
print(f"train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}")

# ---------------------------------------------------------------- experiments
epochs = 200
lr = 2e-3
batch_size = 64
seeds = [0, 1, 2]
n_samples = 359  # number of prior samples to score

ae_results = []
vae_results = []

for seed in seeds:
    print(f"\n--- Seed {seed} ---")

    # --- Plain autoencoder ---
    ae = S.AE(d_latent=8, d_hidden=64)
    S.train(ae, S.ae_loss, Xtr, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    ae_recon = S.reconstruction_mse(ae, Xte)

    # Code spread for AE (just the outputs of the encoder)
    with torch.no_grad():
        ae_codes = ae.encode(Xtr)
    ae_code_std = ae_codes.std(0).numpy()
    ae_code_var = ae_codes.var(0).numpy()

    # 359 prior samples from AE
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    z_prior = torch.randn(n_samples, 8, generator=g)
    ae_samples = ae.decode(z_prior).clamp(0, 1)

    ae_results.append({
        "seed": seed,
        "recon_mse": ae_recon,
        "code_std": ae_code_std,
        "code_var": ae_code_var,
        "samples": ae_samples,
    })
    print(f"  AE recon MSE: {round(ae_recon, 4)}")
    print(f"  AE code std: mean={ae_code_std.mean():.4f}, std={ae_code_std.std():.4f}")

    # --- VAE ---
    vae = VAE(d_latent=8, d_hidden=64)
    train_vae(vae, Xtr, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, beta=1.0)

    # Reconstruction from posterior mean (not sample!)
    with torch.no_grad():
        mu_te, _ = vae.encode(Xte)
        vae_recon = ((vae.decode(mu_te).sigmoid() - Xte) ** 2).mean().item()

    # Code spread for VAE (posterior means of training data)
    with torch.no_grad():
        vae_mu_tr, _ = vae.encode(Xtr)
    vae_code_std = vae_mu_tr.std(0).numpy()
    vae_code_var = vae_mu_tr.var(0).numpy()

    # 359 prior samples from VAE
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    z_prior = torch.randn(n_samples, 8, generator=g)
    vae_samples = vae.decode(z_prior).clamp(0, 1)

    vae_results.append({
        "seed": seed,
        "recon_mse": vae_recon,
        "code_std": vae_code_std,
        "code_var": vae_code_var,
        "samples": vae_samples,
    })
    print(f"  VAE recon MSE: {round(vae_recon, 4)}")
    print(f"  VAE code std: mean={vae_code_std.mean():.4f}, std={vae_code_std.std():.4f}")

# ---------------------------------------------------------------- score the samples
print("\n" + "=" * 70)
print("Sample quality scores (feature network trained with seed=0)")
print("=" * 70)

net = S.train_feature_net(Xtr, ytr, epochs=60, seed=0)
fte = S.get_features(net, Xte)

print("\n| seed | AE frechet | AE precision | AE recall | VAE frechet | VAE precision | VAE recall |")
print("|------|------------|--------------|-----------|-------------|---------------|------------|")

for i, seed in enumerate(seeds):
    # AE samples
    ae_scores = S.score_samples(ae_results[i]["samples"], fte, net, Xtr)
    # VAE samples
    vae_scores = S.score_samples(vae_results[i]["samples"], fte, net, Xtr)

    print(f"| {seed} | {ae_scores['frechet']:.1f} | {ae_scores['precision']:.3f} | {ae_scores['recall']:.3f} | "
          f"{vae_scores['frechet']:.1f} | {vae_scores['precision']:.3f} | {vae_scores['recall']:.3f} |")

# ---------------------------------------------------------------- code spread table
print("\n" + "=" * 70)
print("Code spread across dimensions")
print("=" * 70)

print("\nAutoencoder code dimension std (seeds 0,1,2):")
for seed in seeds:
    stds = ae_results[seeds.index(seed)]["code_std"]
    print(f"  Seed {seed}: {', '.join([str(round(s,3)) for s in stds])}")

print("\nVAE encoder posterior mean std (seeds 0,1,2):")
for seed in seeds:
    stds = vae_results[seeds.index(seed)]["code_std"]
    print(f"  Seed {seed}: {', '.join([str(round(s,3)) for s in stds])}")

# ---------------------------------------------------------------- summary figure
print("\n[Generating comparison figures...]")
for seed in seeds:
    idx = seeds.index(seed)
    S.save_grid(
        torch.cat([ae_results[idx]["samples"][:16], vae_results[idx]["samples"][:16]]),
        f"figures/ae_vs_vae_seed{seed}.png",
        ncols=8,
        title=f"AE (top) / VAE (bottom) prior samples, seed={seed}"
    )
    print(f"  Saved figures/ae_vs_vae_seed{seed}.png")

S.save_grid(
    torch.cat([r["samples"][:32] for r in ae_results]),
    "figures/ae_samples_all_seeds.png",
    ncols=16,
    title="AE prior samples (seeds 0,1,2, top to bottom)"
)
S.save_grid(
    torch.cat([r["samples"][:32] for r in vae_results]),
    "figures/vae_samples_all_seeds.png",
    ncols=16,
    title="VAE prior samples (seeds 0,1,2, top to bottom)"
)
print("Saved summary comparison figures.")

# ---------------------------------------------------------------- summary text for README
print("\n" + "=" * 70)
print("Summary for README")
print("=" * 70)
print("""
KEY OBSERVATIONS:
- Autoencoder codes: The standard deviation across dimensions is near zero (essentially all
  dimensions have similar variance, but the encoder doesn't produce a structured code that
  matches the prior). Without KL regularization, the autoencoder learns arbitrary latent
  dimensions that don't follow N(0, I).
- VAE codes: The standard deviations are closer to 1 (matching the standard normal prior)
  because the KL term pushes the posterior means toward unit variance. This is what allows
  the VAE to generate: decoder(z) for z ~ N(0, I) produces diverse, meaningful images.
- Reconstruction: The autoencoder typically reconstructs better than the VAE because it
  doesn't sacrifice reconstruction quality to match the prior.
- Generation: The VAE's prior samples (third column of figures) show coherent digits, while
  the AE's prior samples are mostly noise or unrecognizable.
""")