"""Variational autoencoder written from scratch for Lab 07: Latent Spaces.

The encoder is q(z|x) = N(mu, sigma^2) with a single hidden layer of 64 units;
the decoder is p(x|z), a Bernoulli on the pixel intensities (logits -> sigmoid).
With n_classes > 0 both networks also consume a one-hot label.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Gaussian encoder, Bernoulli decoder, 8-dim code, 64 hidden units per side."""

    def __init__(self, d_in=64, d_hidden=64, d_latent=8, n_classes=0):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.n_classes = n_classes
        # Encoder: image (+ optional one-hot label) -> hidden -> (mu, logvar)
        self.enc = nn.Sequential(nn.Linear(d_in + n_classes, d_hidden), nn.ReLU())
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)
        # Decoder: code (+ optional one-hot label) -> hidden -> image logits
        self.dec = nn.Sequential(
            nn.Linear(d_latent + n_classes, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in),
        )

    def encode(self, x, c=None):
        h = self.enc(x if c is None else torch.cat([x, c], 1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, c=None):
        """Returns logits; apply a sigmoid to get pixel intensities in [0, 1]."""
        return self.dec(z if c is None else torch.cat([z, c], 1))

    @staticmethod
    def reparameterize(mu, logvar, generator=None):
        """z = mu + sigma * eps, eps ~ N(0, I). Gradients flow through mu and sigma."""
        eps = torch.randn(mu.shape, generator=generator)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def kl_to_standard_normal(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, 1) ) SUMMED over latent dimensions, one value per example.

    Closed form: 0.5 * sum_d ( mu_d^2 + sigma_d^2 - 1 - log(sigma_d^2) ).
    """
    return 0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar).sum(1)


def reconstruction_nll(logits, x):
    """Bernoulli negative log-likelihood of the pixel intensities, SUMMED over pixels,
    one value per example. This is the reconstruction term of the ELBO."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)


def elbo_terms(model, x, c=None):
    """One-sample estimate of the two terms of the negative ELBO, each averaged over the batch.
    Returns (reconstruction, kl). Both are summed over their inner dimensions (pixels / latent dims)
    and then averaged over the batch, so they are directly comparable across models."""
    logits, mu, logvar = model(x, c)
    return reconstruction_nll(logits, x).mean(), kl_to_standard_normal(mu, logvar).mean()


def vae_loss(model, xb, yb=None, beta=1.0):
    """reconstruction + beta * KL, each term summed over its inner dimension and
    averaged over the batch. The batch mean is what the starter's train() expects."""
    rec, kl = elbo_terms(model, xb, yb)
    return rec + beta * kl


@torch.no_grad()
def active_units(model, X, threshold=0.01):
    """Number of latent dimensions whose posterior mean varies across the data by more
    than the threshold (variance across the training images)."""
    mu, _ = model.encode(X)
    return int((mu.var(0) > threshold).sum().item())


@torch.no_grad()
def evaluate_terms(model, X, c=None, repeats=10, seed=0):
    """Average (reconstruction, KL) over the set, using several draws of z per example."""
    torch.manual_seed(seed)
    r = k = 0.0
    for _ in range(repeats):
        rec, kl = elbo_terms(model, X, c)
        r += rec.item() / repeats
        k += kl.item() / repeats
    return r, k


def train_vae(model, X, y_onehot=None, epochs=200, lr=2e-3, batch_size=64,
              beta=1.0, anneal_epochs=0, seed=0):
    """Minimize recon + beta * KL with Adam. With anneal_epochs > 0, beta rises linearly
    from 0 to its final value over that many epochs.
    Returns a list of (reconstruction, KL) epoch averages."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        b = beta * min(1.0, ep / anneal_epochs) if anneal_epochs > 0 else beta
        perm = torch.randperm(len(X), generator=g)
        sr = sk = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            cb = y_onehot[idx] if y_onehot is not None else None
            rec, kl = elbo_terms(model, xb, cb)
            loss = rec + b * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            sr += rec.item() * len(xb)
            sk += kl.item() * len(xb)
        hist.append((sr / len(X), sk / len(X)))
    return hist


# -----------------------------------------------------------------------------
# Latent space utilities (for Piece 4)
# -----------------------------------------------------------------------------

@torch.no_grad()
def interpolate_latent(model, x1, x2, steps=8):
    """Decode points on the straight line between the latent means of two images.
    Returns images in [0, 1]."""
    mu1, _ = model.encode(x1.unsqueeze(0))
    mu2, _ = model.encode(x2.unsqueeze(0))
    ts = torch.linspace(0, 1, steps)[:, None]
    z = (1 - ts) * mu1 + ts * mu2
    return torch.sigmoid(model.decode(z)).clamp(0, 1)


def interpolate_pixels(x1, x2, steps=8):
    """The baseline: a straight line between the two images in pixel space."""
    ts = torch.linspace(0, 1, steps)[:, None]
    return (1 - ts) * x1.unsqueeze(0) + ts * x1.unsqueeze(0)


@torch.no_grad()
def decode_grid(model, lim=2.5, n=12):
    """Decode an n by n grid of a two-dimensional latent space.
    Returns an array of shape (n, n, 8, 8), top row = largest z2."""
    zs = torch.linspace(-lim, lim, n)
    zz = torch.stack(torch.meshgrid(zs, zs.flip(0), indexing="xy"), -1).reshape(-1, 2)
    out = torch.sigmoid(model.decode(zz)).clamp(0, 1)
    return out.reshape(n, n, 8, 8)


@torch.no_grad()
def sample_prior(model, n, seed=0):
    """Draw z from the standard normal prior and decode. Intensities in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, model.d_latent, generator=g)
    return torch.sigmoid(model.decode(z)).clamp(0, 1)


@torch.no_grad()
def sample_conditional(model, digit, n, seed=0):
    """Conditional VAE: draw z from the prior and decode with the one-hot label of the requested digit."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, model.d_latent, generator=g)
    c = torch.nn.functional.one_hot(torch.full((n,), digit), model.n_classes).float()
    return torch.sigmoid(model.decode(z, c)).clamp(0, 1)