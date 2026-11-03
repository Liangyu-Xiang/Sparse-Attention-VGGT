"""Optional monotone repair for grossly underfilled Lagrange allocations.

The original sixteen-round dual solver can return a feasible but heavily
underfilled discrete allocation.  This operator only upgrades individual
clusters to a higher-cost state when that strictly decreases their *existing*
loss.  No loss formula, state definition, or probability estimate is changed.

This is a greedy feasible improvement, NOT an exact knapsack solver.
GPU sort cost must be included in end-to-end timing.
"""
import torch


@torch.no_grad()
def repair_underfilled(errors: torch.Tensor, costs: torch.Tensor,
                       selected: torch.Tensor, budget: int,
                       passes: int = 2,
                       second_pass_topk: int = 0) -> torch.Tensor:
    """Greedily spend unused budget on positive loss reductions.

    errors, costs: [group, cluster, state]; selected: [group, cluster].
    Candidates are chosen by loss reduction per additional key; each pass
    upgrades at most one state per cluster. The prefix rule is conservative,
    so a few unused keys may remain. The returned selection is never more
    expensive and never has a higher sum of per-cluster losses than input.
    """
    if errors.shape != costs.shape or errors.ndim != 3:
        raise ValueError("errors and costs must have shape [G,F,S]")
    if selected.shape != errors.shape[:2]:
        raise ValueError("selected must have shape [G,F]")
    if budget < 0 or passes < 1 or second_pass_topk < 0:
        raise ValueError("budget and second_pass_topk must be nonnegative; passes >= 1")

    out = selected.clone()
    for step in range(passes):
        old_cost = costs.gather(2, out.unsqueeze(-1)).squeeze(-1)
        old_loss = errors.gather(2, out.unsqueeze(-1)).squeeze(-1)
        available = (budget - old_cost.sum(1)).clamp_min(0)

        extra = costs - old_cost.unsqueeze(-1)
        gain = old_loss.unsqueeze(-1) - errors
        eligible = (extra > 0) & (gain > 0) & torch.isfinite(gain) & torch.isfinite(errors)
        ratio = torch.where(eligible, gain / extra.clamp_min(1), -torch.inf)
        best_score, best_state = ratio.max(2)
        best_cost = extra.gather(2, best_state.unsqueeze(-1)).squeeze(-1)

        # Sort per query group; invalid candidates contribute zero to prefix.
        # The first pass may use many keys, so retain the full stable sort.
        # After it, only the remaining budget can be upgraded. Since every
        # upgrade costs at least one additional key, when slack <= k only
        # the first k ranked candidates can affect the selected states.
        if step > 0 and second_pass_topk > 0 and best_score.dtype == torch.float32:
            k = min(second_pass_topk, best_score.shape[1])
            if bool((available > k).any()):
                # Necessary for arbitrary inputs with large remaining budgets.
                order = best_score.argsort(dim=1, descending=True, stable=True)
            else:
                # Scores are strictly positive finite ratios or -inf.
                # Positive float32 bit patterns preserve numerical order;
                # adding lower-cluster-index tie bits reproduces stable sort.
                f = best_score.shape[1]
                bits = best_score.contiguous().view(torch.int32).to(torch.int64)
                tie = f - 1 - torch.arange(f, device=best_score.device, dtype=torch.int64)
                packed = (bits << 32) + tie[None, :]
                order = packed.topk(k, dim=1, largest=True, sorted=True).indices
        else:
            order = best_score.argsort(dim=1, descending=True, stable=True)
        ordered_score = best_score.gather(1, order)
        ordered_cost = best_cost.gather(1, order)
        ordered_cost = torch.where(torch.isfinite(ordered_score), ordered_cost, 0)
        accepted = (
            torch.isfinite(ordered_score)
            & (ordered_score > 0)
            & (ordered_cost.cumsum(1) <= available[:, None])
        )
        ordered_old = out.gather(1, order)
        ordered_new = best_state.gather(1, order).to(out.dtype)
        out.scatter_(1, order, torch.where(accepted, ordered_new, ordered_old))
    return out
