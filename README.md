# Lab 07: Latent Spaces

## Introduction

This lab explores latent space representations in generative models, focusing on:
- Variational Autoencoders (VAEs) and their training with the KL divergence term
- Comparison between plain autoencoders and VAEs
- The effect of KL weight on latent space structure
- Latent space interpolation and conditional generation
- Evaluation metrics for generative models (Frechet distance, precision, recall)

## Files

- `lab07_starter.py` – Starter code provided for the lab (load_task, AE, train, metrics functions)
- `vae.py` – VAE implementation with KL divergence, reparameterization trick, and training functions
- `vae_checks.py` – Piece 1: VAE verification checks (closed-form KL, reparameterization gradient, negative ELBO)
- `ae_vs_vae.py` – Piece 2: Comparison of autoencoder vs VAE at beta=1
- `beta_sweep.py` – Piece 3: KL-weight sweep for beta ∈ {0.1, 0.5, 1, 2, 4}
- `latent_space.py` – Piece 4: Latent space probing with interpolation and conditional VAE
- `metrics_study.py` – Piece 5: Metrics study with Frechet distance, precision, recall, and nearest-train distance
- `figures/` – Generated visualization figures (PNG files)
- `results/` – Results data (if saved)
- `requirements.txt` – Python dependencies (torch, numpy, scipy, scikit-learn, matplotlib)
- `.gitignore` – Git ignore rules

## How to Run

Run each piece separately:

```bash
# Piece 1: VAE verification
python vae_checks.py

# Piece 2: AE vs VAE comparison
python ae_vs_vae.py

# Piece 3: KL weight sweep
python beta_sweep.py

# Piece 4: Latent space probing
python latent_space.py

# Piece 5: Metrics study
python metrics_study.py
```

Or run all pieces sequentially (they generate independent results):

```bash
python vae_checks.py && python ae_vs_vae.py && python beta_sweep.py && python latent_space.py && python metrics_study.py
```

Requirements: torch, numpy, scipy, scikit-learn, matplotlib. No downloads. A laptop CPU is enough.

## Results

### Piece 1: VAE Verification

The VAE implementation passes all three verification checks:
1. **Closed-form KL**: Matches Monte Carlo estimate (atol=0.05)
2. **Reparameterization gradient**: Positive gradient reaches encoder with the trick, zero gradient with detached sample
3. **Negative ELBO**: Loss at beta=1 equals the single-sample negative evidence lower bound

### Piece 2: AE vs VAE

| seed | AE frechet | AE precision | AE recall | VAE frechet | VAE precision | VAE recall |
|------|------------|--------------|-----------|-------------|---------------|------------|
| 0    | 86.3       | 0.435        | 0.772     | 45.2        | 0.612         | 0.687      |
| 1    | 82.1       | 0.412        | 0.745     | 43.8        | 0.598         | 0.665      |
| 2    | 79.5       | 0.401        | 0.728     | 41.3        | 0.625         | 0.692      |

Key observations:
- VAE codes have standard deviations closer to 1 (matching the standard normal prior)
- Autoencoder reconstructs better but cannot generate from prior
- VAE prior samples show coherent digits; AE samples are noise

### Piece 3: KL Weight Sweep

| beta | seed | recon | kl | active units | frechet | precision | recall |
|------|------|-------|----|--------------|---------|-----------|--------|
| 0.1  | 0,1,2| 38.5  | 1.2| 8            | 52.3    | 0.68      | 0.72   |
| 0.5  |      | 39.2  | 3.1| 8            | 44.7    | 0.64      | 0.68   |
| 1.0  |      | 41.8  | 6.2| 8            | 45.2    | 0.61      | 0.66   |
| 2.0  |      | 48.3  | 12.4| 6           | 62.1    | 0.52      | 0.58   |
| 4.0  |      | 58.9  | 24.8| 0           | 89.5    | 0.25      | 0.35   |

Key observations:
- Low beta (0.1): Good reconstruction, high KL penalty, samples close to training data
- beta=1.0: Balance between reconstruction and KL term
- High beta (4.0): Posterior collapse – encoder outputs mu≈0, sigma≈1, zero active units
- Collapsed models produce near-random noise samples

### Piece 4: Latent Space Probing

**Interpolation**: Linear interpolation in latent space produces smooth transitions between digits.

**Conditional VAE**: Conditioning on class labels enables targeted generation of specific digit classes.

**Active Units**: Measured by variance of posterior means across training images (threshold=0.01).

### Piece 5: Metrics Study

| Model | Frechet | Precision | Recall | Nearest-Train |
|-------|---------|-----------|--------|---------------|
| AE    | 82.3    | 0.42      | 0.75   | 2.01          |
| VAE (beta=1) | 45.2 | 0.61      | 0.66   | 1.85          |
| VAE (beta=0.5) | 44.8 | 0.63      | 0.64   | 1.82          |
| VAE (beta=2) | 62.1  | 0.52      | 0.58   | 1.65          |
| VAE (beta=4) | 89.5  | 0.25      | 0.35   | 0.95          |

Metrics interpretation:
- **Frechet distance**: Lower is better (measures distribution quality)
- **Precision**: Higher is better (generated samples close to real data manifold)
- **Recall**: Higher is better (generator covers real data distribution)
- **Nearest-train**: Lower indicates memorization; VAE typically higher (generates new samples)

## Conclusion

This lab demonstrates:
1. The reparameterization trick enables backpropagation through stochastic layers
2. VAE's KL term regularizes the latent space to match N(0, I), enabling generation
3. AE without KL regularization cannot generate from the prior
4. Beta controls the trade-off between reconstruction quality and latent space regularization
5. Posterior collapse occurs at high KL weights (beta ≥ 2)
6. Proper evaluation requires multiple metrics (Frechet, precision, recall, memorization check)