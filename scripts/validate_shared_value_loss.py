"""Check shared Cluster-Grid distortion against direct tokenwise losses.

Run: python scripts/validate_shared_value_loss.py
Without CUDA this validates the mathematical decomposition only.
With CUDA it additionally checks production Triton gate/State Loss kernels and
the no-compensation CUDA-Graph selector on the same synthetic inputs.
"""
import torch


def example(device):
    torch.manual_seed(20261104)
    n, f, r, g, dim = 128, 24, 18, 5, 12
    labels = torch.arange(n, device=device) % f
    grid = torch.randint(r, (n,), device=device)
    z = torch.randn(n, dim, device=device)
    reps = torch.stack([torch.nonzero(labels == c)[0, 0] for c in range(f)])
    population = torch.bincount(labels, minlength=f).float()
    a = z.square().sum(-1)
    b = (z-z[reps[labels]]).square().sum(-1)
    sizes = torch.bincount(grid, minlength=r).clamp_min(1).float()
    uniform = torch.rand(g, r, device=device) / sizes
    prior = uniform[:, grid].clone()
    old_ids = torch.stack([torch.randperm(n, device=device)[:n//5] for _ in range(g)])
    for group in range(g):
        ix = old_ids[group]
        prior[group, ix] *= 0.5 + torch.rand(len(ix), device=device)
    pi = torch.zeros(g, f, device=device).scatter_add_(1, labels[None].expand(g, -1), prior)
    return z, labels, grid, reps, population, uniform, prior, old_ids, pi, a, b


def check_math(v, qualified):
    z, labels, grid, reps, population, uniform, prior, old_ids, pi, a, b = v
    g, n = prior.shape
    f = len(reps)
    r = uniform.shape[1]
    stats = torch.sparse_coo_tensor(
        torch.stack((labels, grid)), torch.stack((a, b), -1), (f, r, 2)
    ).coalesce()
    cluster, region = stats.indices()
    vals = stats.values()
    for group in range(g):
        drop = torch.zeros(f, device=prior.device)
        rep = torch.zeros_like(drop)
        drop.index_add_(0, cluster, uniform[group, region]*vals[:, 0])
        rep.index_add_(0, cluster, uniform[group, region]*vals[:, 1])
        kept = torch.zeros_like(drop)
        for key in old_ids[group].tolist():
            c = int(labels[key])
            delta = prior[group, key]-uniform[group, grid[key]]
            drop[c] += delta*a[key]
            rep[c] += delta*b[key]
            if qualified[group, key]:
                kept[c] += prior[group, key]*a[key]
        for c in range(f):
            mask = labels == c
            expected = torch.stack((
                (prior[group, mask]*a[mask]).sum(),
                (prior[group, mask]*b[mask]).sum(),
                (prior[group, mask]*a[mask]*(~qualified[group, mask])).sum(),
            ))
            obtained = torch.stack((drop[c], rep[c], drop[c]-kept[c]))
            torch.testing.assert_close(obtained, expected, atol=1e-4, rtol=2e-5)


def check_gpu(v):
    from saf3r.patch.reuse_inference_ops import gate
    from saf3r.patch.reuse_window_graph import Window
    from saf3r.triton.reuse_drop_moment import prepare as prepare_bins
    from saf3r.triton import reuse_shared_value_states as shared

    z, labels, grid, reps, population, uniform, prior, old_ids, pi, a, b = v
    context = gate(prior, pi, population, old_ids, labels, reps)
    check_math(v, context["qualified"])
    geom = shared.prepare_geometry(z, labels, reps, population)
    geom["drop_bins"] = prepare_bins(population)
    geom["fuse_state_kernel"] = True
    shared.prepare(z, labels, reps, grid, uniform.shape[1], geom)
    residuals = shared.corrections(prior, old_ids, uniform, labels, geom, context)
    allowed = torch.rand(len(prior), len(reps), device=prior.device) > 0.5
    errors, costs = shared.states(uniform, allowed, population, geom, context, residuals)

    for group in range(len(prior)):
        for c in range(len(reps)):
            mask = labels == c
            drop = (prior[group, mask]*a[mask]).sum()
            rep = (prior[group, mask]*b[mask]).sum()
            hist = (prior[group, mask]*a[mask]*(~context["qualified"][group, mask])).sum()
            expected = torch.stack((
                drop,
                torch.full_like(drop, float("inf")) if context["remove_rep"][group, c] else rep,
                torch.full_like(hist, float("inf")) if context["eligible_count"][group, c] == 0 else hist,
                torch.full_like(drop, float("inf")) if not allowed[group, c] else torch.zeros_like(drop),
            ))
            torch.testing.assert_close(errors[group, c], expected, atol=1e-4, rtol=2e-5)
    graph = Window(prior, pi, old_ids, allowed, z, reps, labels, population,
                   geom, cap=24, iterations=16, drop_multiplier=1., uniform=uniform)
    ids, real, weights, record = graph(prior, pi, old_ids, allowed, uniform=uniform)
    assert bool((weights[real] == 1).all())
    assert bool((weights[~real] == 0).all())
    assert bool((record["counts"] <= 24).all())
    print("PASS: CUDA Triton State Loss and zero-bias CUDA Graph selection")


if __name__ == "__main__":
    cpu = example("cpu")
    qualified = torch.zeros_like(cpu[6], dtype=torch.bool)
    qualified.scatter_(1, cpu[7][:, ::3], True)
    check_math(cpu, qualified)
    print("PASS: CPU shared Cluster-Grid/History loss algebra")
    if torch.cuda.is_available():
        check_gpu(example("cuda"))
    else:
        print("SKIP: CUDA production kernel/Graph test (no CUDA device)")
