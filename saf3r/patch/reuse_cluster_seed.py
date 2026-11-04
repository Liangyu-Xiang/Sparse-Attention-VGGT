"""Budget-preserving first-layer seeds when cluster count is below budget."""
import torch


def prepare_seed(reps, labels, population, budget):
    """All clusters retain a representative; extra real members have unit mass.

    The old F>=budget path is unchanged. For F<budget choose deterministic,
    uniformly spaced ranks among non-representative patch IDs. Subtract their
    unit masses from their representatives, preserving every cluster's mass.
    """
    clusters = len(reps)
    if not 0 < budget <= len(labels):
        raise ValueError('Initial patch budget must be in [1, patch count]')
    if clusters >= budget:
        seed = torch.arange(budget, device=reps.device) * clusters // budget
        ids = reps[seed]
        return ids, population[seed], torch.zeros_like(ids, dtype=torch.bool)
    nonrep = torch.ones(len(labels), device=reps.device, dtype=torch.bool)
    nonrep[reps] = False
    available = nonrep.nonzero(as_tuple=True)[0]
    count = budget - clusters
    ranks = torch.arange(count, device=reps.device) * len(available) // count
    extra = available[ranks]
    deducted = torch.bincount(labels[extra], minlength=clusters).to(population.dtype)
    ids = torch.cat((reps, extra))
    mass = torch.cat((population - deducted, population.new_ones(count)))
    real = torch.arange(budget, device=reps.device) >= clusters
    return ids, mass, real
