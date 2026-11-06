"""Same-QKV fast4 teacher matching,no geometric test truth or model reruns."""
import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def block_sum(x,grid,r):
 import torch
 return torch.zeros((len(x),r),device=x.device).scatter_add_(1,grid[None].expand(len(x),-1),x)

def support(data,key,device):
 import torch
 chunks=data['windows'];g=len(data['groups']);width=chunks[0][key].shape[1];out=torch.empty((g,width),dtype=chunks[0][key].dtype,device=device)
 for row in chunks:out[row['group_rows'].to(device)]=row[key].to(device)
 return out

def observe(data,block,probes,samples,device):
 import torch
 patch_ids=data['patches'].to(device);specials=data['specials'].to(device);k=data['k'].to(device);idx=[0] if probes==1 else [0,2] if probes==2 else [0,1,2,3];q=data['q'][:,:,idx].to(device).float();h,g,p,d=q.shape;n=len(patch_ids);r=(n+block-1)//block;grid=torch.arange(n,device=device)//block;sizes=torch.bincount(grid,minlength=r).float()
 sp=torch.einsum('hgpd,hsd->hgps',q,k[:,specials].float())*data['scale'];patch=k[:,patch_ids].float()
 if samples:
  m=samples;effective=sizes.clamp_max(m).long();slot=torch.arange(m,device=device);offset=((2*slot[None]+1)*sizes.long()[:,None]//(2*effective[:,None]));valid=slot[None]<effective[:,None];sample_ids=(torch.arange(r,device=device)[:,None]*block+offset).clamp_max(n-1)
  kk=patch[:,sample_ids.flatten()];raw=torch.einsum('hgpd,hkd->hgpk',q,kk).reshape(h,g,p,r,m)*data['scale'];raw=raw.masked_fill(~valid[None,None,None],-torch.inf);score=raw.logsumexp(-1)-effective.float().log()[None,None,None]
 else:
  pooled=torch.zeros(h,r,d,device=device).index_add_(1,grid,patch)/sizes[None,:,None];score=torch.einsum('hgpd,hrd->hgpr',q,pooled)*data['scale'];raw=None;sample_ids=None;valid=None
 den=torch.logaddexp((score+sizes.log()[None,None,None]).logsumexp(-1),sp.logsumexp(-1))
 base=(score-den[:,:,:,None]).exp().mean((0,2));coarse=base*sizes
 result=dict(q=q,k=k,patches=patch_ids,specials=specials,grid=grid,sizes=sizes,regions=r,base=base,coarse=coarse,score=score,sample_raw=raw,sample_ids=sample_ids,sample_valid=valid,block=block)
 if samples:
  pooled=torch.zeros(h,r,d,device=device).index_add_(1,grid,patch)/sizes[None,:,None];mean_score=torch.einsum('hgpd,hrd->hgpr',q,pooled)*data['scale'];mean_den=torch.logaddexp((mean_score+sizes.log()[None,None,None]).logsumexp(-1),sp.logsumexp(-1));mean_base=(mean_score-mean_den[:,:,:,None]).exp().mean((0,2))
  result.update(base_mean=mean_base,coarse_mean=mean_base*sizes,sample_conditional=raw.softmax(-1).mean((0,2)),normalizer_score=mean_score)
 return result

def assignment(data,obs,style):
 import torch
 grid=obs['grid'];sample=obs['sample_ids'];valid=obs['sample_valid'];n=len(grid);r=obs['regions'];block=obs['block']
 if style=='position':distance=(torch.arange(n,device=grid.device)[:,None]-sample[grid]).abs().float()
 else:
  from saf3r.patch.head_value_sketch import projection,features
  if '_key_features_cuda' not in data:
   k=obs['k'][:,obs['patches']];data['_key_features_cuda']=features(k,projection(k.shape[0],k.shape[-1],4,k.device))
  f=data['_key_features_cuda'];anchors=f[sample];all_ids=torch.arange(r*block,device=f.device).clamp_max(n-1);x=f[all_ids].reshape(r,block,-1)
  distance=(x.square().sum(-1)[:,:,None]+anchors.square().sum(-1)[:,None]-2*torch.bmm(x,anchors.transpose(1,2))).reshape(r*block,-1)[:n]
 distance=distance.masked_fill(~valid[grid],torch.inf)
 return distance.argmin(-1)

def predict(old,obs,weight,bootstrap,teacher,anchor=None,gamma=.5):
 import torch
 grid=obs['grid'];sizes=obs['sizes'];cur=obs['coarse'];base=obs['base'][:,grid]
 if old is None:return teacher.clone() if bootstrap else base.clone()
 previous=old['coarse'];full=old['block_mass'];patch=full*cur/previous.clamp_min(1e-30);special=(1-full.sum(1)).clamp_min(0)*(1-cur.sum(1)).clamp_min(0)/(1-previous.sum(1)).clamp_min(1e-30);target=patch/(patch.sum(1)+special).clamp_min(1e-30)[:,None]
 beta=(2*(cur*previous).clamp_min(0).sqrt()/(cur+previous).clamp_min(1e-30)).clamp(0,1)
 conditional=old['phi']/full[:,grid].clamp_min(1e-30)
 if anchor is not None:
  delta=(obs['sample_conditional'].clamp_min(1e-30).log()-old['sample_conditional'].clamp_min(1e-30).log())
  change=delta[:,grid,obs['assignment']]
  tilted=conditional*(gamma*change).clamp(-20,20).exp();normalizer=block_sum(tilted,grid,len(sizes));conditional=tilted/normalizer[:,grid].clamp_min(1e-30)
 historical=target[:,grid]*(beta[:,grid]*conditional+(1-beta[:,grid])/sizes[grid])
 return weight*historical+(1-weight)*base

def update(data,obs,ids,pred,mode,eta=1.):
 import torch
 g,a=ids.shape;valid=ids>=0;safe=ids.clamp_min(0).long();grid=obs['grid'];r=obs['regions'];sizes=obs['sizes'];regions=grid[safe];q=obs['q'];k=obs['k'];h,_,p,d=q.shape
 keys=torch.cat((obs['specials'][None].expand(g,-1),obs['patches'][safe]),1);s=len(obs['specials']);raw=torch.einsum('hgpd,hgad->hgpa',q,k[:,keys].float())*data['scale'];raw[:,:,:,s:]=raw[:,:,:,s:].masked_fill(~valid[None,:,None],-torch.inf)
 count=torch.zeros(g,r,device=q.device).scatter_add_(1,regions,valid.float());remaining=sizes-count
 if mode.startswith('conditional'):
  mass=(1-pred.sum(1)+(pred.gather(1,safe)*valid).sum(1)).clamp(1e-7,1-1e-7)
  if mode=='conditional_odds':
   score=obs.get('normalizer_score',obs['score']);miss=(score+remaining.clamp_min(1).log()[None,:,None]).masked_fill(remaining[None,:,None]<=0,-torch.inf).logsumexp(-1);odds=raw.logsumexp(-1)-miss;lo=torch.full_like(mass,-40.);hi=torch.full_like(mass,40.)
   for _ in range(16):
    mid=(lo+hi)*.5;m=(odds+mid[None,:,None]).sigmoid().mean((0,2));lo=torch.where(m<mass,mid,lo);hi=torch.where(m<mass,hi,mid)
   head_mass=(odds+((lo+hi)*.5)[None,:,None]).sigmoid();observed=(raw.softmax(-1)*head_mass[:,:,:,None]).mean((0,2))[:,s:]*valid
  else:observed=raw.softmax(-1).mean((0,2))[:,s:]*valid*mass[:,None]
  out=pred.clone();current=pred.gather(1,safe);new=(current.clamp_min(1e-30).log()*(1-eta)+observed.clamp_min(1e-30).log()*eta).exp()
  for j in range(g):out[j,safe[j,valid[j]]]=new[j,valid[j]]
  target=block_sum(pred,grid,r);actual=block_sum(out,grid,r);out*= (target/actual.clamp_min(1e-30))[:,grid]
 else:
  score=obs['score']
  if obs['sample_raw'] is not None:
   selected=torch.zeros(g,len(grid)+1,device=q.device,dtype=torch.bool);selected.scatter_(1,ids.long().masked_fill(~valid,len(grid)),valid)
   sample_selected=selected[:,:len(grid)][:,obs['sample_ids'].flatten()].reshape(g,r,-1);unselected=obs['sample_valid'][None]&~sample_selected;available=unselected.sum(-1)
   missing_score=obs['sample_raw'].masked_fill(~unselected[None,:,None],-torch.inf).logsumexp(-1)-available.clamp_min(1).float().log()[None,:,None]
   score=torch.where(available[None,:,None]>0,missing_score,score)
  miss=(score+remaining.clamp_min(1).log()[None,:,None]).masked_fill(remaining[None,:,None]<=0,-torch.inf).logsumexp(-1);den=torch.logaddexp(raw.logsumexp(-1),miss)
  selected_p=(raw-den[:,:,:,None]).exp().mean((0,2));uniform=(score-den[:,:,:,None]).exp().mean((0,2));miss_mass=uniform*remaining
  if mode=='compressed':out=uniform[:,grid].clone()
  else:
   out=pred.clone()
   for j in range(g):out[j,safe[j,valid[j]]]=0.
   old_mass=block_sum(out,grid,r);out*= (miss_mass/old_mass.clamp_min(1e-30))[:,grid]
  for j in range(g):out[j,safe[j,valid[j]]]=selected_p[j,s:][valid[j]]
  norm=(out.sum(1)+selected_p[:,:s].sum(1)).clamp_min(1e-30);out/=norm[:,None]
  if eta!=1:out=eta*out+(1-eta)*pred
 result=dict(phi=out,coarse=obs['coarse'],block_mass=block_sum(out,grid,r))
 if obs['sample_raw'] is not None:result['sample_conditional']=obs['sample_conditional']
 return result

def metrics(pred,true,ids,cap):
 import torch
 eps=1e-30;p=pred/pred.sum(1,keepdim=True).clamp_min(eps);q=true/true.sum(1,keepdim=True).clamp_min(eps);m=(p+q)*.5
 result=dict(relative_l1=float((pred-true).abs().sum()/true.sum().clamp_min(eps)),relative_l2=float((pred-true).square().sum().sqrt()/true.square().sum().sqrt().clamp_min(eps)),js=float((.5*(q*(q.clamp_min(eps).log()-m.clamp_min(eps).log())).sum(1)+.5*(p*(p.clamp_min(eps).log()-m.clamp_min(eps).log())).sum(1)).mean()),cosine=float(torch.nn.functional.cosine_similarity(pred,true,dim=1).mean()))
 valid=ids>=0;safe=ids.clamp_min(0).long();a=pred.gather(1,safe)*valid;b=true.gather(1,safe)*valid;result['history_edges']=int(valid.sum());result['history_relative_l1']=float((a-b).abs().sum()/b.sum().clamp_min(eps))
 top=pred.topk(min(cap,pred.shape[1]),1).indices;oracle=true.topk(min(cap,pred.shape[1]),1).values.sum(1);result['top_budget_mass_ratio']=float((true.gather(1,top).sum(1)/oracle.clamp_min(eps)).mean());return result

def candidates():
 rows=[]
 for probes in [1,2,4]:
  for samples in [0,4,8]:
   rows.append(dict(name=f'compressed_p{probes}_s{samples}',probes=probes,samples=samples,mode='compressed',bootstrap=False,weight=.5,eta=1.,block=64))
 for probes in [2,4]:
  for samples in [0,4,8]:
   for mode in ['persistent','conditional']:
    rows.append(dict(name=f'{mode}_p{probes}_s{samples}_boot',probes=probes,samples=samples,mode=mode,bootstrap=True,weight=.75,eta=1.,block=64))
 for samples in [4,8]:
  for anchor in ['key','position']:
   for gamma in [.5,1.]:rows.append(dict(name=f'anchor_{anchor}_s{samples}_g{gamma}',probes=4,samples=samples,mode='conditional',bootstrap=True,weight=.75,eta=1.,block=64,anchor=anchor,gamma=gamma))
 for probes in [2,4]:
  for gamma in [.5,1.]:rows.append(dict(name=f'odds_p{probes}_s8_g{gamma}',probes=probes,samples=8,mode='conditional_odds',bootstrap=True,weight=.75,eta=1.,block=64,anchor='key',gamma=gamma))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--refine',type=Path);p.add_argument('--odds-only',action='store_true');a=p.parse_args()
 import torch
 from scripts.quality_metrics import quality_record
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 manifest=json.loads((a.replay/'manifest.json').read_text());assert manifest['complete'];configs=candidates()
 if a.odds_only:configs=[c for c in configs if c['mode']=='conditional_odds']
 if a.refine:
  originals=json.loads(a.refine.read_text());configs=[]
  for c in originals:
   for weight in [.25,.5,.75,1.]:
    for eta in [.5,1.]:configs.append(dict(c,name=c['name']+f'_w{weight}_u{eta}',weight=weight,eta=eta))
 old={c['name']:None for c in configs};rows=[];intrinsic=[];previous=None
 for record in manifest['layers']:
  layer=record['layer'];data=torch.load(a.replay/record['path'],map_location='cpu',weights_only=True);patches=data['patches'].cuda();teacher=data['teacher4'].cuda()[:,patches];ids=support(data,'current_ids','cuda');history=support(data,'old_ids','cuda');baseline=support(data,'prior','cuda');observations={}
  intr=dict(layer=layer,baseline=metrics(baseline,teacher,history,data['patch_cap']))
  if previous is not None:intr['perfect_previous_fast4']=metrics(previous,teacher,history,data['patch_cap'])
  previous=teacher.clone();intrinsic.append(intr)
  for c in configs:
   key=(c['block'],c['probes'],c['samples'])
   if key not in observations:observations[key]=observe(data,*key,'cuda')
   obs=observations[key]
   if c.get('anchor'):
    assignment_key=(key,c['anchor'])
    if assignment_key not in observations:observations[assignment_key]=assignment(data,obs,c['anchor'])
    obs=dict(obs,base=obs['base_mean'],coarse=obs['coarse_mean'],assignment=observations[assignment_key])
   pred=predict(old[c['name']],obs,c['weight'],c['bootstrap'],teacher,c.get('anchor'),c.get('gamma',.5));row=dict(name=c['name'],layer=layer,**metrics(pred,teacher,history,data['patch_cap']));rows.append(row)
   if layer==0 and c['bootstrap']:
    old[c['name']]=dict(phi=teacher.clone(),coarse=obs['coarse'],block_mass=block_sum(teacher,obs['grid'],obs['regions']))
    if obs['sample_raw'] is not None:old[c['name']]['sample_conditional']=obs['sample_conditional']
   else:old[c['name']]=update(data,obs,ids,pred,c['mode'],c['eta'])
  print('LAYER',manifest['sequence'],layer,flush=True);del observations,data;torch.cuda.empty_cache()
 summary=[]
 for c in configs:
  rr=[r for r in rows if r['name']==c['name']];summary.append(dict(config=c,post_bootstrap_relative_l1=statistics.mean(r['relative_l1'] for r in rr if r['layer']>0),**{key:statistics.mean(r[key] for r in rr if key!='history_relative_l1' or r['history_edges']>0) for key in ['relative_l1','relative_l2','js','cosine','history_relative_l1','top_budget_mass_ratio']}))
 summary.sort(key=lambda r:(r['relative_l1'],-r['top_budget_mass_ratio']))
 save(a.output,dict(quality=quality_record({},'Offline same-QKV fast4 probability matching,not geometric quality'),sequence=manifest['sequence'],summary=summary,rows=rows,intrinsic=intrinsic,selection_uses_test_gt=False,speedup_vs_dense=None,speedup_reason='Offline replay,not full-forward timing',scope='Fixed sparse source trajectory and fixed previous support;candidate prediction caches evolve on that common QKV,not a full neural rollout'))
 print(json.dumps(summary[:5],indent=2))
if __name__=='__main__':main()
