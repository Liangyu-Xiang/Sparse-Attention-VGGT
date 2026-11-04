"""Budget-preserving first-layer seeds when Cluster count is below Key budget."""
import torch

def prepare_seed(reps,labels,population,budget):
    """Keep one Rep per Cluster, then distinct non-Rep Keys until the Key budget is filled.

    F>=budget is bitwise identical to the original seed selection.
    Extra selected Keys carry unit population; mass is subtracted from their
    original Cluster Rep for consistency with the upstream seed accounting.
    """
    clusters=len(reps)
    if not 0<budget<=len(labels):
        raise ValueError('Initial patch budget must be in [1, patch count]')
    if clusters>=budget:
        seed=torch.arange(budget,device=reps.device)*clusters//budget
        ids=reps[seed]
        return ids,population[seed],torch.zeros_like(ids,dtype=torch.bool)
    nonrep=torch.ones(len(labels),device=reps.device,dtype=torch.bool)
    nonrep[reps]=False
    available=nonrep.nonzero(as_tuple=True)[0]
    count=budget-clusters
    ranks=torch.arange(count,device=reps.device)*len(available)//count
    extra=available[ranks]
    deducted=torch.bincount(labels[extra],minlength=clusters).to(population.dtype)
    ids=torch.cat((reps,extra))
    mass=torch.cat((population-deducted,population.new_ones(count)))
    real=torch.arange(budget,device=reps.device)>=clusters
    return ids,mass,real
