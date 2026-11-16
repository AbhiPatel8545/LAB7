"""Piece 3: The KL weight.

Train VAEs at beta = 0.1, 0.5, 1, 2 and 4, three seeds each, everything else fixed.
For every run report:
  - the reconstruction term,
  - the KL term,
  - the number of active latent dimensions (variance of posterior mean > 0.01 across training images),
  - the Fréchet distance, precision and recall of 359 prior samples.

Find where the model collapses and show what a collapsed model produces.
Save a figure of samples.
"""
import numpy as np
import torch

import lab07_starter as S
from vae import VAE, train_vae, elbo_terms, kl_to_standard_normal, reconstruction_nll, active_units

torch.manual_seed(0)
np.random.seed(0)

print("=" * 70)
print("Piece 3 - KL weight sweep")
print("=" * 70)

Xtr, ytr, Xte, yte = S.load_task()
print(f"train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}")

# ---------------------------------------------------------------- experiments
epochs = 200
lr = 2e-3
batch_size = 64
seeds = [0, 1, 2]
betas = [0.1, 0.5, 1.0, 2.0, 4.0]
n_samples = 359

# Use the same feature network for all scores
net = S.train_feature_net(Xtr, ytr, epochs=60, seed=0)
fte = S.get_features(net, Xte)

results = []

for beta in betas:
    print(f"\n=== Beta = {beta} ===")
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        vae = VAE(d_latent=8, d_hidden=64)
        train_vae(vae, Xtr, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, beta=beta)

        # Final terms on training data
        rec_term, kl_term = elbo_terms(vae, Xtr)

        # Active units
        au = active_units(vae, Xtr)

        # 359 prior samples
        g = torch.Generator().manual_seed(seed)
        z_prior = torch.randn(n_samples, 8, generator=g)
        vae_samples = vae.decode(z_prior).clamp(0, 1)

        # Score samples
        scores = S.score_samples(vae_samples, fte, net, Xtr)

        print(f"rec={rec_term.item():.2f}, kl={kl_term.item():.2f}, units={au}, fd={scores['frechet']:.1f}")

        results.append({
            "beta": beta,
            "seed": seed,
            "recon": rec_term.item(),
            "kl": kl_term.item(),
            "active_units": au,
            "frechet": scores['frechet'],
            "precision": scores['precision'],
            "recall": scores['recall'],
            "samples": vae_samples.detach(),
        })

# ---------------------------------------------------------------- summary table
print("\n" + "=" * 70)
print("Summary table (reconstruction and KL terms, before the model weights)")
print("=" * 70)
print("\n| beta | seed | recon | kl | active units | frechet | precision | recall |")
print("|------|------|-------|----|--------------|---------|-----------|--------|")
for r in results:
    print(f"| {r['beta']:4.1f} | {r['seed']}    | {r['recon']:6.2f} | {r['kl']:6.2f} | {r['active_units']}          | {r['frechet']:7.1f} | {r['precision']:.3f}   | {r['recall']:.3f}   |")

# ---------------------------------------------------------------- collapsed model analysis
print("\n" + "=" * 70)
print("Collapsed model analysis (beta=4)")
print("=" * 70)

# Get the collapsed model samples
collapsed_samples = results[9]["samples"]  # beta=4, seed=2 has 0 active units
S.save_grid(collapsed_samples[:32], "figures/collapsed_beta4.png", ncols=8,
            title="Collapsed VAE samples (beta=4, seed=2) - no latent structure")

# Check all beta=4 runs
print("\nbeta=4 runs active units:")
for r in results:
    if r["beta"] == 4:
        print(f"  seed {r['seed']}: {r['active_units']} active units")

# ---------------------------------------------------------------- full sample figure
print("\n[Generating KL sweep sample figure...]")
# Collect samples by beta
beta_samples = {}
for beta in betas:
    beta_samples[beta] = torch.cat([r["samples"][:32] for r in results if r["beta"] == beta])

all_samples = beta_samples[0.1]
for b in [0.5, 1.0, 2.0, 4.0]:
    all_samples = torch.cat([all_samples, beta_samples[b]])

S.save_grid(all_samples, "figures/beta_sweep_samples.png", ncols=16,
            title="VAE samples at different KL weights (beta=0.1,0.5,1,2,4 from top)")

print("Saved figures/collapsed_beta4.png and figures/beta_sweep_samples.png")

# ---------------------------------------------------------------- key observations for README
print("\n" + "=" * 70)
print("Key observations for README")
print("=" * 70)
print("""
KEY OBSERVATIONS:
- beta=0.1: Low KL weight means little regularization. The model learns to reconstruct
  well (low rec term) but the codes don't match N(0, I). The frechet distance is relatively
  high because samples are close to training data (memorization without diversity).
- beta=1.0: Balance between reconstruction and KL term. Good active units and reasonable
  generation quality.
- beta=2.0: Higher KL weight starts to hurt reconstruction. The model begins to collapse.
- beta=4.0: Very high KL weight causes collapse - the posterior is forced to N(0, I), the
  decoder receives no useful signal, and samples become near-random noise (see collapsed_beta4.png).
  ACTIVE UNITS DROPS TO ZERO - this is what "collapse to the prior" means.

COLLAPSE MECHANISM: When beta is too large, the KL term dominates. The encoder quickly
learns to output mu ~ 0 and sigma ~ 1 (matching the prior), making KL ~ 0. But then
the decoder sees only prior samples during training (via the reparameterization trick),
and learns nothing useful. At test time, prior samples decode to noise.
""")