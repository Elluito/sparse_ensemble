"""
Filter Pruning via Geometric Median (FPGM) — Section 3.4, He et al. CVPR 2019.

Two-stage pruning per layer
----------------------------
Stage 1 – norm: remove the (1 - norm_rate) fraction of filters with the
    smallest L2 norm.  These are the conventionally "less important" filters
    and are discarded unconditionally.

Stage 2 – FPGM: among the norm-surviving filters only, compute
    g(x) = sum_{j'} dist(x, F_{j'})  (Eq. 6)
    and remove the dist_rate fraction (of the full layer) that lie closest to
    the geometric median of that subset.  Restricting GM scoring to the
    large-norm survivors keeps the redundancy signal clean — already-zeroed
    filters would otherwise distort the distance matrix.

Net filters kept per layer: norm_rate - dist_rate.

Masks are applied directly to weight data (param.data[fi] = 0) rather than
stored as persistent buffer tensors on the model.

Typical training loop usage
---------------------------
    masks = fpgm_prune(model, layer_indices, norm_rate=0.9, dist_rate=0.1)
    for epoch in range(epochs):
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        mask_gradients(model, masks)   # keep pruned filters zeroed
        optimizer.step()
        if epoch % prune_interval == 0:
            masks = fpgm_prune(model, layer_indices, norm_rate=0.9, dist_rate=0.1)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance as scipy_dist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_split(weight: torch.Tensor, norm_rate: float):
    """
    Split filter indices by L2 norm.

    Returns
    -------
    small_idx : indices of the bottom (1 - norm_rate) filters — pruned by norm
    large_idx : indices of the top norm_rate filters — candidates for FPGM
    """
    n = weight.size(0)
    n_small = int(n * (1 - norm_rate))

    vecs = weight.detach().cpu().view(n, -1)
    norms = torch.norm(vecs, p=2, dim=1).numpy()

    order = norms.argsort()           # ascending: smallest norm first
    small_idx = order[:n_small]
    large_idx = order[n_small:]
    return small_idx, large_idx


def _gm_scores(weight: torch.Tensor, subset_indices: np.ndarray, dist_type: str) -> np.ndarray:
    """
    Compute g(x) = sum_{j'} dist(x, F_{j'}) restricted to the filters in
    subset_indices (Eq. 6).  Lower score = closer to geometric median = more
    replaceable by the other surviving filters.

    Returns a (len(subset_indices),) float64 score array in the same order as
    subset_indices.
    """
    n = weight.size(0)
    vecs_all = weight.detach().cpu().view(n, -1).float().numpy()
    vecs = vecs_all[subset_indices]   # (M, C*kH*kW)

    if dist_type == "l2":
        dm = scipy_dist.cdist(vecs, vecs, "euclidean")
    elif dist_type == "l1":
        dm = scipy_dist.cdist(vecs, vecs, "cityblock")
    elif dist_type == "cos":
        # cosine distance ∈ [0, 2]: smaller = more similar = more redundant
        dm = scipy_dist.cdist(vecs, vecs, "cosine")
    else:
        raise ValueError(f"dist_type must be 'l2', 'l1', or 'cos', got '{dist_type}'")

    return dm.sum(axis=1)   # (M,)


def _build_keep_mask(
    weight: torch.Tensor,
    norm_rate: float,
    dist_rate: float,
    dist_type: str,
) -> np.ndarray:
    """
    Return a bool array of shape (N,): True = keep, False = prune.

    Stage 1: small-norm filters → False
    Stage 2: among large-norm filters, the dist_rate fraction (of N) with the
             lowest GM score → False
    """
    n = weight.size(0)
    small_idx, large_idx = _norm_split(weight, norm_rate)

    # How many to additionally prune via FPGM (counted over full layer)
    n_dist_prune = max(0, int(n * dist_rate))

    scores = _gm_scores(weight, large_idx, dist_type)
    # Indices into large_idx of the most GM-redundant filters
    most_redundant = np.argsort(scores)[:n_dist_prune]
    fpgm_prune_idx = large_idx[most_redundant]

    keep = np.ones(n, dtype=bool)
    keep[small_idx] = False
    keep[fpgm_prune_idx] = False
    return keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fpgm_prune(
    model: nn.Module,
    layer_indices: list,
    norm_rate: float = 0.9,
    dist_rate: float = 0.1,
    dist_type: str = "l2",
) -> dict:
    """
    Apply two-stage FPGM pruning to the specified layers.

    Stage 1 removes the (1 - norm_rate) fraction of filters with smallest L2
    norm.  Stage 2 removes the dist_rate fraction (of the full layer) from the
    norm survivors that are closest to the geometric median of that subset.

    Weights are zeroed in-place on param.data; no mask buffers are registered
    on the model.

    Parameters
    ----------
    model         : network to prune
    layer_indices : indices from enumerate(model.parameters()) to prune;
                    non-4D parameters at those indices are skipped silently
    norm_rate     : fraction of filters to KEEP after the norm stage (default 0.9)
    dist_rate     : additional fraction to prune by FPGM from those survivors
                    (default 0.1); net kept = norm_rate - dist_rate
    dist_type     : distance metric for GM scoring — 'l2' (default), 'l1', 'cos'

    Returns
    -------
    masks : dict[int -> np.ndarray[bool, shape (N,)]]
        True = kept filter, False = pruned.
        Pass this to mask_gradients() after every backward pass.
    """
    masks = {}

    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            if idx not in layer_indices:
                continue
            if param.data.dim() != 4:
                continue

            keep = _build_keep_mask(param.data, norm_rate, dist_rate, dist_type)
            keep_t = torch.tensor(keep, dtype=torch.bool, device=param.device)

            # Zero pruned filter slices directly in-place
            param.data[~keep_t] = 0.0

            masks[idx] = keep

    return masks


def mask_gradients(model: nn.Module, masks: dict) -> None:
    """
    Zero the gradients of every pruned filter so the optimizer cannot revive
    them.  Call after loss.backward() and before optimizer.step().
    """
    for idx, param in enumerate(model.parameters()):
        if idx not in masks or param.grad is None:
            continue
        keep_t = torch.tensor(masks[idx], dtype=torch.bool, device=param.device)
        param.grad.data[~keep_t] = 0.0


def count_pruned(model: nn.Module, masks: dict) -> dict:
    """
    Return {layer_index: (n_pruned, n_total)} for every masked layer.
    """
    info = {}
    for idx, param in enumerate(model.parameters()):
        if idx not in masks:
            continue
        keep = masks[idx]
        info[idx] = (int((~keep).sum()), len(keep))
    return info
