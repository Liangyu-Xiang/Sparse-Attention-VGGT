"""Parent-local joint fine states with exact or calibrated Value moments."""
import itertools,math
import torch
import torch.nn.functional as F

def templates(device):
 # -1 means keep that child's current baseline state. Candidate0 preserves it.
 rows=[(-1,)*4]
 for mask in range(15):rows.append(tuple(-1 if mask>>j&1 else 0 for j in range(4)))
 for mask in range(1,16):rows.append(tuple(1 if mask>>j&1 else 0 for j in range(4)))
 for j in range(4):
  for state in range(2,11):rows.append(tuple(state if k==j else 0 for k in range(4)))
 for j in range(4):
  for state in range(1,11):rows.append(tuple(state if k==j else -1 for k in range(4)))
 rows.append((10,)*4)
 return torch.tensor(rows,device=device,dtype=torch.int32)

def balanced_parents(z,reps,arity=4):
 if arity not in (2,4):raise ValueError('Parent arity must be2 or4')
 x=z[reps].float();f,d=x.shape
 centered=x-x.mean(0);_,vectors=torch.linalg.eigh(centered.T@centered)
 pc=vectors[:,-min(4,d):];sign=pc[pc.abs().argmax(0),torch.arange(pc.shape[1],device=x.device)].sign();pc=pc*torch.where(sign==0,1,sign)
 projection=centered@pc;bins=torch.empty_like(projection,dtype=torch.long)
 for axis in range(projection.shape[1]):
  order=projection[:,axis].argsort(stable=True);bins[order,axis]=torch.arange(f,device=x.device)*32//f
 code=torch.zeros(f,device=x.device,dtype=torch.long)
 for bit in range(5):
  for axis in range(projection.shape[1]):code|=((bins[:,axis]>>bit)&1)<<(bit*projection.shape[1]+axis)
 order=code.argsort(stable=True)
 combinations=torch.tensor([(0,)+c for c in itertools.combinations(range(1,2*arity),arity-1)],device=x.device)
 mask=torch.zeros((len(combinations),2*arity),device=x.device).scatter_(1,combinations,1.)
 for offset in (0,arity):
  prefix=order[:offset];work=order[offset:];length=len(work)
  if not length:continue
  padded=F.pad(work,(0,(-length)%(2*arity)),value=-1).reshape(-1,2*arity);valid=padded>=0
  values=x[padded.clamp_min(0)]*valid[:,:,None]
  sums=torch.einsum('pnd,cn->pcd',values,mask);total=values.sum(1)[:,None];count=valid.float()@mask.T;remaining=valid.sum(1)[:,None]-count
  target=valid.sum(1).clamp_max(arity)[:,None]
  score=sums.square().sum(-1)/count.clamp_min(1)+(total-sums).square().sum(-1)/remaining.clamp_min(1)
  score=score.masked_fill(count!=target,-torch.inf);best=score.argmax(1)
  pick=mask[best].bool();position=torch.arange(2*arity,device=x.device)[None].expand_as(padded)
  idx=torch.cat((position.masked_select(pick).reshape(-1,arity),position.masked_select(~pick).reshape(-1,arity)),1)
  work=padded.gather(1,idx).flatten();order=torch.cat((prefix,work[work>=0]))
 groups=F.pad(order,(0,(-f)%arity),value=-1).reshape(-1,arity)
 return F.pad(groups,(0,4-arity),value=-1).int().contiguous()

def prepare(z,reps,labels,geometry,arity):
 parents=balanced_parents(z,reps,arity)
 distance=geometry['rep_distance'] if geometry is not None else (1-(z*z[reps[labels]]).sum(-1)).clamp(0,2)
 maximum=torch.full((len(reps),),-torch.inf,device=z.device).scatter_reduce_(0,labels,distance,reduce='amax')
 candidate=torch.where(distance==maximum[labels],torch.arange(len(z),device=z.device),len(z))
 extreme=torch.full((len(reps),),len(z),device=z.device,dtype=torch.long).scatter_reduce_(0,labels,candidate,reduce='amin')
 return dict(parents=parents,extreme=extreme.int(),template=templates(z.device),labels=labels)

def compressed_moment(pi,rep_error,z,reps,hids,prior):
 valid=hids>=0;safe=hids.clamp_min(0).long();ph=prior.gather(1,safe.flatten(1)).reshape_as(hids).masked_fill(~valid,0)
 zr=z[reps];zh=z[safe];dh=(1-(zh*zr[None,:,None]).sum(-1)).clamp(0,2)
 residual=(pi-ph.sum(-1)).clamp_min(0)
 intrinsic=(rep_error-(ph*dh).sum(-1)).clamp_min(0);intrinsic=torch.minimum(intrinsic,2*residual)
 moment=(residual-intrinsic)[:,:,None]*zr[None]+(ph[:,:,:,None]*zh).sum(2)
 return moment

def reference_cost(states,errors,costs,pi,moment,z,reps,hids,extreme,prior=None):
 """Tiny-parent CPU oracle; actual selected Exact keys proxied by Rep+extreme."""
 g,f=states.shape;out=[];routes=[]
 for group in range(g):
  keys=[]
  for child in range(f):
   state=int(states[group,child]);h=hids.shape[-1]
   if state==1 or h+2<=state<=2*h+1 or state==2*h+2:keys.append(int(reps[child]))
   if 2<=state<=h+1:keys+=hids[group,child,:state-1].tolist()
   if h+2<=state<=2*h+1:keys+=hids[group,child,:state-h-1].tolist()
   if state==2*h+2:keys.append(int(extreme[child]))
  keys=sorted(set(k for k in keys if k>=0));loss=torch.tensor(0.,device=z.device);owners=[]
  for child in range(f):
   state=int(states[group,child]);owner=-1
   if state:loss+=errors[group,child,state]
   elif keys:
    valid=hids[group,child]>=0;observed=hids[group,child,valid].long()
    ph=prior[group,observed] if prior is not None else torch.zeros(0,device=z.device)
    residual_mass=pi[group,child]-ph.sum()
    residual_m=moment[group,child]-(ph[:,None]*z[observed]).sum(0) if len(observed) else moment[group,child]
    atom_m=torch.cat((residual_m[None],ph[:,None]*z[observed]),0) if len(observed) else residual_m[None]
    atom_mass=torch.cat((residual_mass[None],ph))
    matrix=(atom_mass[:,None]-atom_m@z[keys].T).clamp_min(0)
    best=matrix.min(1).values
    if best.sum()<errors[group,child,0]:loss+=best.sum();owner=keys[int(matrix[0].argmin())]
    else:loss+=errors[group,child,0]
   else:loss+=errors[group,child,0]
   owners.append(owner)
  out.append(loss);routes.append(owners)
 return torch.stack(out),costs.gather(2,states[:,:,None]).sum((1,2)),torch.tensor(routes,device=z.device)

def allocate(errors,costs,pi,z,reps,hids,prior,baseline,prepared,cap,iterations,passes=2,geometry=None,moment_mode="exact"):
 from saf3r.triton.hierarchical_reuse import bank,choose,route,moments
 state=baseline;records=[];owners=None
 exact= moments(z,prior,prepared["labels"],reps,geometry) if moment_mode=="exact" else None
 for _ in range(passes):
  e,c=bank(errors,costs,pi,z,reps,hids,prior,state,prepared,exact)
  selected=choose(e,c,cap,iterations)
  loss=e.gather(2,selected[:,:,None]).sum((1,2));baseline_loss=e[:,:,0].sum(1)
  count=c.gather(2,selected[:,:,None]).sum((1,2))
  fallback=(count>cap)|(loss>baseline_loss+1e-6*pi.sum(1))|~torch.isfinite(loss)
  selected=torch.where(fallback[:,None],0,selected)
  state,owners=route(errors,costs,pi,z,reps,hids,prior,state,prepared,selected,exact)
  records.append(dict(fallback=fallback,proxy_loss=e.gather(2,selected[:,:,None]).sum((1,2)),baseline_proxy_loss=baseline_loss))
 return state,owners,records

def transfer(pops,packed,prior,pi,owners,hids):
 g,p=prior.shape;valid=hids>=0
 ph=prior.gather(1,hids.long().clamp_min(0).flatten(1)).reshape_as(hids).masked_fill(~valid,0)
 residual=(pi-ph.sum(-1)).clamp_min(0)
 atom_mass=torch.cat((residual[:,:,None],ph),-1)
 delegated=owners>=0
 received=torch.zeros((g,p+1),device=prior.device)
 received.scatter_add_(1,owners.long().masked_fill(~delegated,p).flatten(1),atom_mass.masked_fill(~delegated,0).flatten(1))
 safe=packed.long().clamp_min(0);own=prior.gather(1,safe)
 added=received[:,:p].gather(1,safe).masked_fill(packed<0,0)
 weights=pops+added/own.clamp_min(1e-30)
 delivered=added.sum(1);expected=atom_mass.masked_fill(~delegated,0).sum((1,2))
 if bool(((delivered-expected).abs()>1e-5).any()):raise RuntimeError('Delegation owner missing or duplicated in actual support')
 return weights,dict(delegated_clusters=delegated.any(-1).sum(1),delegated_probability_mass=expected,mass_transfer_error=(delivered-expected).abs())
