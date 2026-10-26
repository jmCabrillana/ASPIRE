"""Locate-and-edit for SmolVLA: closed-form rank-one weight edits (the ROME / MEMIT idea, applied to a VLA).

An MLP's down projection maps a key (the neuron activation vector of a token, k in R^2560 for VLM layers) to a value (its
contribution to the residual stream, W k). To make the layer write v* whenever it sees k (e.g. the key of an unknown verb ->
the value the layer writes for a known verb), the minimal-change update under the key covariance C is

    W' = W + (v* - W k) (C^-1 k)^T / (k^T C^-1 k)

which leaves keys orthogonal (under C) to k almost unchanged. With several (k_i, v_i) pairs, the least-squares version is
W' = W + (V - W K) (C^-1 K)^T ((C^-1 K)^T K)^-1  (MEMIT-style, one layer). In a program apply it with ctx.edit_weights so it is
undone after the episode; a verified edit can be stored as a library entry (the delta is a learned fact).
"""

from __future__ import annotations

import torch


def rank_one(W: torch.Tensor, k: torch.Tensor, v_target: torch.Tensor, C: torch.Tensor | None = None) -> torch.Tensor:
    """Delta for W (out, in) so that (W + delta) k = v_target. C: (in, in) key covariance (identity if None)."""
    W, k, v_target = W.float(), k.float().to(W.device), v_target.float().to(W.device)
    ck = k if C is None else torch.linalg.solve(C.float().to(W.device), k)
    return torch.outer(v_target - W @ k, ck) / (ck @ k)


def multi(W: torch.Tensor, K: torch.Tensor, V: torch.Tensor, C: torch.Tensor | None = None, ridge: float = 1e-4) -> torch.Tensor:
    """Delta for several pairs: K (in, n) keys, V (out, n) target values; least-squares under C."""
    W, K, V = W.float(), K.float().to(W.device), V.float().to(W.device)
    CK = K if C is None else torch.linalg.solve(C.float().to(W.device), K)
    A = CK.T @ K + ridge * torch.eye(K.shape[1], device=W.device)
    return (V - W @ K) @ torch.linalg.solve(A, CK.T)


def key_covariance(keys: torch.Tensor, shrink: float = 0.1) -> torch.Tensor:
    """Covariance of neuron activation vectors (n, in) with shrinkage towards the identity (for C^-1 to be stable)."""
    x = keys.float()
    c = torch.cov(x.T)
    return (1 - shrink) * c + shrink * torch.eye(c.shape[0]) * c.diagonal().mean()
