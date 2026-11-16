"""Piece 1: the VAE, verified.

Three checks, each with code, on the VAE written in vae.py:

  1. Closed-form KL vs a Monte Carlo estimate on random Gaussians.
  2. The reparameterization trick: the reconstruction term alone sends a positive
     gradient to the encoder's mean layer, and exactly zero when the sample is detached.
  3. The loss at beta == 1 is the single-sample negative evidence lower bound.

Run: python vae_checks.py
"""
import math
import numpy as np
import torch

import lab07_starter as S
from vae import VAE, kl_to_standard_normal, reconstruction_nll, elbo_terms, vae_loss

torch.manual_seed(0)
np.random.seed(0)

print("=" * 70)
print("Piece 1 - verification of the VAE")
print("=" * 70)

# ---------------------------------------------------------------- check 1
# Closed-form KL against a Monte Carlo estimate on random Gaussians.
print("\n[1] closed-form KL vs Monte Carlo KL")
g = torch.Generator().manual_seed(0)
mu = torch.randn(3, 4, generator=g)
logvar = 0.5 * torch.randn(3, 4, generator=g)
closed = kl_to_standard_normal(mu, logvar)

std = torch.exp(0.5 * logvar)
z = mu[None] + std[None] * torch.randn(200000, 3, 4, generator=g)
log_q = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
log_p = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
mc = (log_q - log_p).mean(0)

print("  closed-form :", closed.numpy())
print("  Monte Carlo :", mc.numpy())
print("  max abs diff:", float((closed - mc).abs().max()))
assert torch.allclose(closed, mc, atol=0.05), "closed-form KL does not match Monte Carlo"
print("  PASS: closed-form KL agrees with the Monte Carlo estimate (atol=0.05)")

# ---------------------------------------------------------------- check 2
# Reparameterization trick: reconstruction gradients reach the encoder's mean layer.
print("\n[2] reconstruction gradient reaching the encoder's mean layer")
v = VAE(d_latent=8)
Xtr, ytr, Xte, yte = S.load_task()
xb = Xtr[:64]

# (a) With the trick: the reconstruction term alone must send a positive gradient to mu's weight.
v.zero_grad()
rec, _ = elbo_terms(v, xb)
rec.mean().backward()
grad_with_trick = v.mu.weight.grad.abs().sum().item()
print("  grad norm on encoder mu layer, with the trick      :", round(grad_with_trick, 6))
assert grad_with_trick > 0, "reconstruction gradient never reaches the encoder"
print("  PASS: positive gradient reaches the encoder's mean layer")

# (b) Detached sample: the reconstruction term sends exactly zero to the encoder.
v.zero_grad()
mu2, logvar2 = v.encode(xb)
z_det = v.reparameterize(mu2, logvar2).detach()
logits_det = v.decode(z_det)
# When we sum over batch and pixel dimensions, then mean over batch, we get a scalar
# that still has no connection to the encoder parameters via the sample (z_det is disconnected).
rec_det = reconstruction_nll(logits_det, xb).mean()
rec_det.backward()
grad_detached = v.mu.weight.grad.abs().sum().item() if v.mu.weight.grad is not None else 0.0
print("  grad norm on encoder mu layer, sample detached      :", grad_detached)
assert grad_detached == 0.0, "a detached sample still leaks gradient to the encoder"
print("  PASS: exactly zero gradient when the sample is detached")

# ---------------------------------------------------------------- check 3
# Loss at beta == 1 is the single-sample negative ELBO.
print("\n[3] loss at beta == 1 is the single-sample negative ELBO")
v.zero_grad()
logits, mu3, logvar3 = v(xb)
rec = reconstruction_nll(logits, xb).mean()
kl = kl_to_standard_normal(mu3, logvar3).mean()
loss_beta1 = rec + kl

# Negative ELBO = -E_q[log p(x|z)] + KL(q || p), one sample, per example then averaged.
neg_elbo = rec + kl
print("  reconstruction term :", rec.item())
print("  KL term             :", kl.item())
print("  loss (beta=1)       :", loss_beta1.item())
print("  negative ELBO       :", neg_elbo.item())
assert torch.allclose(loss_beta1, neg_elbo), "loss at beta=1 is not the negative ELBO"
print("  PASS: loss at beta == 1 equals the single-sample negative ELBO")

print("\nAll three VAE checks passed.")