"""Shared Value-distortion statistics with query-specific sparse probability residuals.

For a cluster C and grid r, A[C,r] = sum ||V_j||^2,
B[C,r] = sum ||V_j-V_rep(C)||^2. All groups reuse these statistics.
Only observed historical edges contribute per-group probability corrections.
History preserves eligible edges without nearest-Value owners or mass transfer.
"""
import torch
import triton
import triton.language as tl


def prepare_geometry(z, labels, reps, population, min_width=1):
    """Compute only the Value scalars required by the new loss, not O(N*width) pair distances."""
    width = max(min_width, triton.next_power_of_2(int(population.max())))
    if width > 512:
        raise ValueError("Shared Value loss currently supports at most 512 members per cluster")
    v_rep = z[reps[labels]]
    return dict(
        width=width,
        norm2=(z*z).sum(-1).contiguous(),
        rep_distance=((z-v_rep)*(z-v_rep)).sum(-1).contiguous(),
    )


def prepare(z, labels, reps, grid, regions, geometry):
    """Once per layer, pack sparse Cluster x Grid statistics into existing width."""
    f = len(reps)
    width = geometry["width"]
    a = geometry["norm2"]
    b = geometry["rep_distance"]
    sparse = torch.sparse_coo_tensor(
        torch.stack((labels.long(), grid.long())),
        torch.stack((a, b), dim=-1),
        size=(f, regions, 2),
        device=z.device,
    ).coalesce()
    cluster, region = sparse.indices()
    values = sparse.values()
    counts = torch.bincount(cluster, minlength=f)
    starts = counts.cumsum(0) - counts
    local = torch.arange(cluster.numel(), device=z.device) - starts[cluster]
    # Number of occupied grids cannot exceed the number of cluster members.
    cells = torch.full((f, width), -1, device=z.device, dtype=torch.int32)
    drop = torch.zeros((f, width), device=z.device, dtype=torch.float32)
    rep = torch.zeros_like(drop)
    cells[cluster, local] = region.to(torch.int32)
    drop[cluster, local] = values[:, 0]
    rep[cluster, local] = values[:, 1]
    geometry.update(
        shared_value_loss=True,
        shared_cells=cells,
        shared_drop=drop,
        shared_rep=rep,
        shared_grid=grid.int().contiguous(),
        shared_token_drop=a.contiguous(),
        shared_token_rep=b.contiguous(),
        shared_regions=regions,
    )


@triton.jit
def _corrections(IDS, PRIOR, UNIFORM, GRID, LABEL, TOKEN_D, TOKEN_R, QUAL,
                 DELTA_D, DELTA_R, KEPT_D,
                 N:tl.constexpr, F:tl.constexpr, R:tl.constexpr,
                 A:tl.constexpr, B:tl.constexpr):
    group = tl.program_id(0)
    col = tl.program_id(1)*B + tl.arange(0, B)
    key = tl.load(IDS + group*A + col, col<A, -1)
    valid = (col<A) & (key>=0)
    safe = tl.maximum(key, 0)
    cluster = tl.load(LABEL + safe, valid, 0)
    region = tl.load(GRID + safe, valid, 0)
    probability = tl.load(PRIOR + group*N + safe, valid, 0.)
    uniform = tl.load(UNIFORM + group*R + region, valid, 0.)
    delta = probability - uniform
    a = tl.load(TOKEN_D + safe, valid, 0.)
    b = tl.load(TOKEN_R + safe, valid, 0.)
    qualified = tl.load(QUAL + group*N + safe, valid, False)
    tl.atomic_add(DELTA_D + group*F + cluster, delta*a,
                  valid & (delta != 0.), sem="relaxed")
    tl.atomic_add(DELTA_R + group*F + cluster, delta*b,
                  valid & (delta != 0.), sem="relaxed")
    tl.atomic_add(KEPT_D + group*F + cluster, probability*a,
                  valid & qualified, sem="relaxed")


def corrections(prior, old_ids, uniform, labels, geometry, context):
    g, n = prior.shape
    f = len(geometry["shared_cells"])
    r = geometry["shared_regions"]
    d = torch.zeros((g, f), device=prior.device, dtype=torch.float32)
    rep = torch.zeros_like(d)
    kept = torch.zeros_like(d)
    _corrections[(g, triton.cdiv(old_ids.shape[1], 256))](
        old_ids, prior, uniform, geometry["shared_grid"], labels,
        geometry["shared_token_drop"], geometry["shared_token_rep"],
        context["qualified"], d, rep, kept, n, f, r, old_ids.shape[1], 256,
        num_warps=4, enable_fp_fusion=False,
    )
    return d, rep, kept


@triton.jit(do_not_specialize=["COUNT"])
def _states(UNIFORM, CELL, CELL_D, CELL_R, CORR_D, CORR_R, KEPT_D,
            REMOVE, AMOUNT, ALLOW, POP, CLUSTERS, ERROR, COST, OBJECTIVE,
            F:tl.constexpr, R:tl.constexpr, W:tl.constexpr,
            COUNT, C:tl.constexpr, M:tl.constexpr, COUNT_PTR:tl.constexpr):
    if COUNT_PTR:
        COUNT = tl.load(COUNT)
    group = tl.program_id(0)
    index = tl.program_id(1)*C + tl.arange(0, C)
    cluster = tl.load(CLUSTERS + index, index<COUNT, 0)
    m = tl.arange(0, M)
    cell = tl.load(CELL + cluster[:, None]*W + m[None, :],
                   index[:, None]<COUNT, -1)
    valid = (index[:, None]<COUNT) & (cell>=0)
    u = tl.load(UNIFORM + group*R + tl.maximum(cell, 0), valid, 0.)
    d = tl.load(CELL_D + cluster[:, None]*W + m[None, :], valid, 0.)
    r = tl.load(CELL_R + cluster[:, None]*W + m[None, :], valid, 0.)
    drop = tl.maximum(0., tl.sum(u*d, 1) +
                      tl.load(CORR_D + group*F + cluster, index<COUNT, 0.))
    rep = tl.maximum(0., tl.sum(u*r, 1) +
                     tl.load(CORR_R + group*F + cluster, index<COUNT, 0.))
    kept = tl.load(KEPT_D + group*F + cluster, index<COUNT, 0.)
    history = tl.maximum(0., drop-kept)
    remove = tl.load(REMOVE + group*F + cluster, index<COUNT, False)
    amount = tl.load(AMOUNT + group*F + cluster, index<COUNT, 0)
    allowed = tl.load(ALLOW + group*F + cluster, index<COUNT, False)
    pop = tl.load(POP + cluster, index<COUNT, 0.)
    rep_error = tl.where(remove, float("inf"), rep)
    hist_error = tl.where(amount>0, history, float("inf"))
    exact_error = tl.where(allowed, 0., float("inf"))
    offset = (group*F + cluster)*4
    tl.store(ERROR + offset, drop, index<COUNT)
    tl.store(ERROR + offset+1, rep_error, index<COUNT)
    tl.store(ERROR + offset+2, hist_error, index<COUNT)
    tl.store(ERROR + offset+3, exact_error, index<COUNT)
    tl.store(COST + offset, 0., index<COUNT)
    tl.store(COST + offset+1, 1., index<COUNT)
    tl.store(COST + offset+2, amount.to(tl.float32), index<COUNT)
    tl.store(COST + offset+3, pop, index<COUNT)
    tl.store(OBJECTIVE + offset, drop-rep, index<COUNT)
    tl.store(OBJECTIVE + offset+1, rep_error-rep, index<COUNT)
    tl.store(OBJECTIVE + offset+2, hist_error-rep, index<COUNT)
    tl.store(OBJECTIVE + offset+3, exact_error-rep, index<COUNT)


def states(uniform, allowed, population, geometry, context, residuals):
    g = len(uniform)
    f = len(population)
    width = geometry["width"]
    errors = torch.empty((g, f, 4), device=uniform.device)
    costs = torch.empty_like(errors)
    objectives = torch.empty_like(errors)
    delta_d, delta_r, kept_d = residuals
    for index, (size, clusters) in enumerate(geometry["drop_bins"]):
        c = min(16, max(1, 128//size))
        use_ptr = "drop_bin_counts" in geometry
        count = geometry["drop_bin_counts"][index] if use_ptr else len(clusters)
        _states[(g, triton.cdiv(len(clusters), c))](
            uniform, geometry["shared_cells"], geometry["shared_drop"],
            geometry["shared_rep"], delta_d, delta_r, kept_d,
            context["remove_rep"], context["eligible_count"], allowed,
            population, clusters, errors, costs, objectives, f,
            geometry["shared_regions"], width, count, c, size, use_ptr,
            num_warps=4, enable_fp_fusion=False,
        )
    geometry["objective"] = objectives
    return errors, costs
