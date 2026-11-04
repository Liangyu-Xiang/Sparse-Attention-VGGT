"""QK-free reference losses for four alternative sparse support operators.

History and Exact always have unit execution weights. Rep is explicitly unit
or population-weighted. All losses are functions of retained members/weights,
not state names. Torch reference for diagnosis; not a speed-optimized kernel.
"""
import math
import torch


def output_sketch(v,module,patches,dim=8):
    generator=torch.Generator(device='cpu').manual_seed(20261105)
    projection=torch.randn(module.proj.out_features,dim,generator=generator)
    projection=torch.linalg.qr(projection,mode='reduced').Q.to(v.device)
    coefficient=module.proj.weight.float().T@projection
    values=v.permute(1,0,2).reshape(v.shape[1],-1).float()@coefficient
    scale=values[patches].square().sum(-1).mean().clamp_min(1e-12).sqrt()
    return (values/scale).contiguous()


def reduce_cluster(values,labels,count):
    if values.ndim==2:
        return torch.zeros((len(values),count),device=values.device).scatter_add_(1,labels[None].expand_as(values),values)
    return torch.zeros((len(values),count,values.shape[-1]),device=values.device).scatter_add_(1,
        labels[None,:,None].expand_as(values),values)


def uniform_unobserved(prior,pi,old_ids,labels,population,sampled):
    """Preserve sampled historical probabilities; uniform remainder per cluster."""
    known=torch.zeros_like(prior,dtype=torch.int32).scatter_add_(1,old_ids.long().clamp_min(0),
        ((old_ids>=0)&sampled[:,None]).int())>0
    mass=reduce_cluster(prior*known,labels,len(population))
    amount=reduce_cluster(known.float(),labels,len(population))
    uniform=(pi-mass).clamp_min(0)/(population[None]-amount).clamp_min(1.)
    return torch.where(known,prior,uniform[:,labels])


def transport_losses(prior,pi,y,special_y,special_p,labels,reps,population,qualified,allowed):
    """H+R: ordinary History members represent themselves; only Rep covers rest.

    Rep weights are assigned residual probability / own estimated probability.
    No historical member except the explicit Rep receives others' mass.
    """
    g,n=prior.shape;f=len(reps)
    rep_mask=torch.zeros(n,device=prior.device,dtype=torch.bool);rep_mask[reps]=True
    other=qualified&~rep_mask[None]
    hp=prior*other
    amount=reduce_cluster(other.float(),labels,f)
    hmass=reduce_cluster(hp,labels,f)
    moment=reduce_cluster(prior[:,:,None]*y[None],labels,f)
    hmoment=reduce_cluster(hp[:,:,None]*y[None],labels,f)
    total=prior.sum(1)+special_p.sum(1)
    mu=(prior@y+special_p@special_y)/total[:,None].clamp_min(1e-30)
    yr=y[reps][None]
    residual=(y[None]-y[reps[labels]][None]).square().sum(-1)
    variance=((prior*(y[None]-mu[:,None]).square().sum(-1)).sum(1)+
        (special_p*(special_y[None]-mu[:,None]).square().sum(-1)).sum(1))/total.clamp_min(1e-30)
    costs=torch.stack((torch.zeros_like(pi),torch.ones_like(pi),amount+1,population[None].expand_as(pi)),-1)
    history_available=reduce_cluster(qualified.float(),labels,f)>0
    def pack(drop,rep,hist):
        result=torch.stack((drop.clamp_min(0),rep.clamp_min(0),hist.clamp_min(0),torch.zeros_like(pi)),-1)
        result[:,:,2].masked_fill_(~history_available,float('inf'))
        result[:,:,3].masked_fill_(~allowed,float('inf'))
        return result
    energy=reduce_cluster(prior*residual,labels,f)
    hist_energy=(energy-reduce_cluster(hp*residual,labels,f)).clamp_min(0)
    squared_energy=reduce_cluster(prior.square()*residual,labels,f)
    squared_hist=(squared_energy-reduce_cluster(prior.square()*other*residual,labels,f)).clamp_min(0)
    drop_energy=reduce_cluster(prior*y.square().sum(-1)[None],labels,f)
    centered_drop=reduce_cluster(prior*(y[None]-mu[:,None]).square().sum(-1),labels,f)
    rd=moment-pi[:,:,None]*mu[:,None]
    rr=moment-pi[:,:,None]*yr
    rh=moment-((pi-hmass)[:,:,None]*yr+hmoment)
    dv=rd.square().sum(-1);rv=rr.square().sum(-1);hv=rh.square().sum(-1)
    definitions=dict(residual_energy=pack(drop_energy,energy,hist_energy),
        centered_residual_energy=pack(centered_drop,energy,hist_energy),
        transport_moment=pack(dv,rv,hv),
        transport_mass=pack(dv+variance[:,None]*pi.square(),rv,hv),
        transport_variance=pack(dv+.1*centered_drop,rv+.1*squared_energy,hv+.1*squared_hist))
    return definitions,costs,other,hmass


def group_history_losses(prior,pi,y,special_y,special_p,labels,reps,population,qualified,allowed):
    """Historical collection represents the cluster with explicit mass calibration.

    Each retained History member gets pi / retained_prior_mass. The residual
    representation loss is only on unselected members; full support is exact.
    """
    f=len(reps);hp=prior*qualified
    hm=reduce_cluster(hp,labels,f);hc=reduce_cluster(qualified.float(),labels,f)
    m=reduce_cluster(prior[:,:,None]*y[None],labels,f)
    h=reduce_cluster(hp[:,:,None]*y[None],labels,f)
    a=reduce_cluster(prior*y.square().sum(-1)[None],labels,f)
    ah=reduce_cluster(hp*y.square().sum(-1)[None],labels,f)
    total=prior.sum(1)+special_p.sum(1)
    mu=(prior@y+special_p@special_y)/total[:,None].clamp_min(1e-30)
    yr=y[reps][None];hy=h/hm[:,:,None].clamp_min(1e-30)
    remainder=(pi-hm).clamp_min(0)
    rep_energy=(a-2*(yr*m).sum(-1)+pi*yr.square().sum(-1)).clamp_min(0)
    h_centroid=(a-ah-2*(hy*(m-h)).sum(-1)+remainder*hy.square().sum(-1)).clamp_min(0)
    h_mixture=(a-ah-2*(hy*(m-h)).sum(-1)+remainder*ah/hm.clamp_min(1e-30)).clamp_min(0)
    centered_drop=(a-2*(mu[:,None]*m).sum(-1)+pi*mu.square().sum(-1)[:,None]).clamp_min(0)
    dv=(m-pi[:,:,None]*mu[:,None]).square().sum(-1)
    rv=(m-pi[:,:,None]*yr).square().sum(-1)
    hv=(m-pi[:,:,None]*hy).square().sum(-1)
    costs=torch.stack((torch.zeros_like(pi),torch.ones_like(pi),hc,population[None].expand_as(pi)),-1)
    def pack(drop,rep,hist):
        result=torch.stack((drop,rep,hist,torch.zeros_like(pi)),-1).clamp_min(0)
        result[:,:,2].masked_fill_((hc==0)|(hm<=1e-30),float('inf'));result[:,:,3].masked_fill_(~allowed,float('inf'))
        return result
    definitions=dict(group_centroid=pack(a,rep_energy,h_centroid),
        group_centered=pack(centered_drop,rep_energy,h_centroid),
        group_mixture=pack(centered_drop,rep_energy,h_mixture),
        group_moment=pack(dv,rv,hv),
        group_moment_variance=pack(dv+.1*pi*centered_drop,rv+.1*pi*rep_energy,hv+.1*pi*h_centroid))
    return definitions,costs,hm


def losses(prior,pi,y,special_y,special_p,labels,reps,population,qualified,allowed,policy):
    """Return losses/costs and target from probabilities estimated without QK."""
    g,n=prior.shape;f=len(reps);d=y.shape[-1]
    total=prior.sum(1)+special_p.sum(1)
    mu=(prior@y+special_p@special_y)/total[:,None].clamp_min(1e-30)
    p_rep=prior[:,reps];y_rep=y[reps][None]
    rep_weight=population[None] if policy=='rep_population' else torch.ones_like(p_rep)
    hprob=prior*qualified
    hmass=reduce_cluster(hprob,labels,f)
    amount=reduce_cluster(qualified.float(),labels,f)
    moment=reduce_cluster(prior[:,:,None]*y[None],labels,f)
    hmoment=reduce_cluster(hprob[:,:,None]*y[None],labels,f)
    variance=((prior*(y[None]-mu[:,None]).square().sum(-1)).sum(1)+
        (special_p*(special_y[None]-mu[:,None]).square().sum(-1)).sum(1))/total.clamp_min(1e-30)
    rep_mass=rep_weight*p_rep
    costs=torch.stack((torch.zeros_like(pi),torch.ones_like(pi),amount,population[None].expand_as(pi)),-1)
    def pack(drop,rep,hist):
        out=torch.stack((drop.clamp_min(0),rep.clamp_min(0),hist.clamp_min(0),torch.zeros_like(drop)),-1)
        out[:,:,2].masked_fill_(amount==0,float('inf'));out[:,:,3].masked_fill_(~allowed,float('inf'))
        return out
    candidates={}
    for name,power,center in [('self_energy',1,False),('centered_energy',1,True),('centered_squared_energy',2,True)]:
        anchor=mu if center else torch.zeros_like(mu)
        energy=(y[None]-anchor[:,None]).square().sum(-1)*prior.pow(power)
        drop=reduce_cluster(energy,labels,f)
        rep=drop+((1-rep_weight).square()-1)*energy[:,reps]
        hist=drop-reduce_cluster(energy*qualified,labels,f)
        candidates[name]=pack(drop,rep,hist)
    drop_vector=moment-pi[:,:,None]*mu[:,None]
    rep_vector=moment-rep_mass[:,:,None]*y_rep-(pi-rep_mass)[:,:,None]*mu[:,None]
    hist_vector=moment-hmoment-(pi-hmass)[:,:,None]*mu[:,None]
    dv=drop_vector.square().sum(-1);rv=rep_vector.square().sum(-1);hv=hist_vector.square().sum(-1)
    for name,gamma in [('centered_moment',0.),('moment_mass01',.1),('moment_mass1',1.)]:
        factor=gamma*variance[:,None]
        candidates[name]=pack(dv+factor*pi.square(),rv+factor*(pi-rep_mass).square(),hv+factor*(pi-hmass).square())
    mean=moment/pi[:,:,None].clamp_min(1e-30)
    conditional_rep=pi.square()*(mean-y_rep).square().sum(-1)+variance[:,None]*(pi-rep_mass).square()
    conditional_h=pi.square()*(mean-hmoment/hmass[:,:,None].clamp_min(1e-30)).square().sum(-1)+variance[:,None]*(pi-hmass).square()
    candidates['conditional_mass']=pack(dv+variance[:,None]*pi.square(),conditional_rep,conditional_h)
    return candidates,costs,mu
