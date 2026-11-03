"""Direct previous-layer sparse-probability transport to a dense-proxy distribution.

This is an opt-in ablation, NOT a claim that sparse-conditional probabilities
are already calibrated dense probabilities.  Each canonical Grid allocates its
CURRENT coarse probability mass jointly to observed History edges and all
unobserved keys.  Special-token mass remains outside this patch-only routine.
"""
import torch


def direct_sparse_history_transport(
    history_ids,
    previous_sparse_probability,
    grid_labels,
    grid_sizes,
    previous_grid_probability,
    current_grid_probability,
    eta_down=1.0,
    eta_up=0.0,
    eps=1e-12,
):
    """Return current-layer dense-proxy probability for each History edge.

    Args:
      history_ids: [G,H], unique previous-layer executed patch IDs or -1.
      previous_sparse_probability: [G,H] NONNEGATIVE per-single-key mass from
        the previous sparse softmax, with representative population already
        divided out.  Must NOT be a current-layer measured probability.
      grid_labels: [num_patches] canonical Grid index of each patch.
      grid_sizes: [R] total patch membership count per Grid.
      previous_grid_probability, current_grid_probability: [G,R], the
        UNCORRECTED coarse Grid probabilities on the SAME normalization scale.
        If normalized over patch keys only, outputs are patch-conditional.
      eta_down, eta_up: M5c-style nonnegative asymmetric confidence exponents.

    Returns:
      history_probability: [G,H], invalid history entries become zero.
      unseen_probability_per_key: [G,R], one current-layer probability for
        each unobserved key in a Grid (not the Grid total).
      beta: [G,R], confidence of inherited history shape.

    For each Grid r:
      score_history(e)=beta_r*p_sparse(e)+(1-beta_r)/n_r
      score_unseen(j)=1/n_r
      denom_r=sum_history(score_history)+(n_r-|H_r|)/n_r
      p_hat(e)=P_current(r)*score_history(e)/denom_r
      p_hat(j)=P_current(r)*(1/n_r)/denom_r

    Hence history and unseen keys together sum to P_current(r), exactly in
    real arithmetic.  Dense attention remains an *estimate*, not an oracle.
    """
    if eta_down < 0 or eta_up < 0:
        raise ValueError("M5c confidence exponents must be nonnegative")
    if history_ids.shape != previous_sparse_probability.shape:
        raise ValueError("History IDs and probabilities must match")
    if current_grid_probability.shape != previous_grid_probability.shape:
        raise ValueError("Current and previous Grid mass must match")
    g, h = history_ids.shape
    r = grid_sizes.numel()
    if current_grid_probability.shape != (g, r):
        raise ValueError("Expected [query groups, Grid count] probabilities")

    p0 = previous_grid_probability.float()
    p1 = current_grid_probability.float()
    sizes = grid_sizes.to(device=p1.device, dtype=torch.float32)
    inv_size = sizes.clamp_min(1).reciprocal()
    ratio = p1 / p0.clamp_min(eps)
    beta = torch.where(
        ratio < 1,
        ratio.clamp_min(eps).pow(eta_down),
        ratio.clamp_min(eps).pow(-eta_up),
    )
    beta = torch.where(p0 > eps, beta, torch.zeros_like(beta)).clamp(0, 1)

    valid = history_ids >= 0
    region = grid_labels[history_ids.long().clamp_min(0)]
    unit_baseline = inv_size[region]
    inherit = beta.gather(1, region)
    sparse = previous_sparse_probability.float().clamp_min(0)
    history_score = torch.where(
        valid, inherit * sparse + (1 - inherit) * unit_baseline, 0.0
    )

    counts = torch.zeros((g, r), device=p1.device, dtype=p1.dtype)
    counts.scatter_add_(1, region, valid.to(p1.dtype))
    accumulated = torch.zeros_like(counts).scatter_add_(1, region, history_score)
    unseen_counts = (sizes[None] - counts).clamp_min(0)
    denominator = accumulated + unseen_counts * inv_size[None]

    # If all Grid members were observed but their inherited scores were zero,
    # the per-Grid normalizer vanishes. Fall back to uniform for that Grid.
    fallback = denominator <= eps
    history_score = torch.where(
        fallback.gather(1, region) & valid, unit_baseline, history_score
    )
    denominator = torch.where(fallback, torch.ones_like(denominator), denominator)

    history_probability = (
        p1.gather(1, region)
        * history_score / denominator.gather(1, region)
    ).masked_fill(~valid, 0)
    unseen_probability_per_key = (
        p1 * inv_size[None] / denominator
    )
    return history_probability, unseen_probability_per_key, beta
