"""Frozen-QKV mask and AV comparison against fast4-derived state decisions."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save
from scripts.replay_history_probability import observe,predict,update,assignment,support,metrics

def geometry(data,device):
 import torch,triton
 z=data['z'].to(device);labels=data['labels'].to(device);reps=data['reps'].to(device);pop=data['population'].to(device);n=len(labels);f=len(pop);width=triton.next_power_of_2(int(pop.max()));order=labels.argsort(stable=True);starts=pop.long().cumsum(0)-pop.long();rank=torch.arange(n,device=device)-starts[labels[order]];members=torch.full((f,width),-1,device=device,dtype=torch.int32);members[labels[order],rank]=order.int()
 geo=dict(probe_member_width=width,members=members,norm2=z.square().sum(-1).contiguous(),rep_distance=(z-z[reps[labels]]).square().sum(-1).contiguous(),rep_history_execution='prior_residual')
 return z,labels,reps,pop,geo

def attention(data,obs,ids,weights):
 import torch
 k=obs['k'];v=data['v'].cuda();q=data['q'].cuda().float();g=len(ids);out=[]
 for j in range(g):
  valid=ids[j]>=0;bank=torch.cat((obs['specials'],obs['patches'][ids[j,valid].long()]));w=torch.cat((torch.ones(len(obs['specials']),device=k.device),weights[j,valid])).log();p=(q[:,j]@k[:,bank].float().transpose(1,2)*data['scale']+w).softmax(-1);out.append(p@v[:,bank].float())
 result=torch.stack(out,1);special_query=torch.isin(data['positions'].cuda(),obs['specials']);dense=data['teacher_output'].cuda();result=torch.where(special_query[None,:,:,None],dense,result);return result

def error(data,a,b):
 import torch
 h,g,p,d=a.shape;weight=data['projection_weight'].cuda().float();aa=(a-b).permute(1,2,0,3).reshape(g*p,h*d);bb=b.permute(1,2,0,3).reshape(g*p,h*d);da=torch.nn.functional.linear(aa,weight);db=torch.nn.functional.linear(bb,weight);return float(da.square().sum()/db.square().sum().clamp_min(1e-20))

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--configs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from saf3r.patch.current_probe_states import select
 from scripts.quality_metrics import quality_record
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 manifest=json.loads((a.replay/'manifest.json').read_text());configs=json.loads(a.configs.read_text());old={c['name']:None for c in configs};old_ids={c['name']:None for c in configs};rows=[]
 for entry in manifest['layers']:
  data=torch.load(a.replay/entry['path'],map_location='cpu',weights_only=True);teacher=data['teacher4'].cuda()[:,data['patches'].cuda()];g=len(teacher);z,labels,reps,pop,geo=geometry(data,'cuda');f=len(pop);observations={};dense=data['teacher_output'].cuda()
  for c in configs:
   key=(c['block'],c['probes'],c['samples'])
   if key not in observations:observations[key]=observe(data,*key,'cuda')
   obs=observations[key]
   if c.get('anchor'):obs=dict(obs,base=obs['base_mean'],coarse=obs['coarse_mean'],assignment=assignment(data,obs,c['anchor']))
   pred=predict(old[c['name']],obs,c['weight'],c['bootstrap'],teacher,c.get('anchor'),c.get('gamma',.5));history=old_ids[c['name']]
   if history is None:history=torch.full((g,1),-1,device='cuda',dtype=torch.int32)
   pi=torch.empty(g,f,device='cuda');allow=torch.ones(g,1,device='cuda',dtype=torch.bool);ids,_,weights,record=select(pred,pi,history,allow,z,reps,labels,pop,geo,data['patch_cap'],16)
   reference,_,ref_weights,ref_record=select(teacher,torch.empty_like(pi),history,allow,z,reps,labels,pop,geo,data['patch_cap'],16)
   assert not bool(record['implementation_validation_error'].any())
   out=attention(data,obs,ids,weights);target=attention(data,obs,reference,ref_weights)
   source_ids=support(data,'current_ids','cuda');source_weights=support(data,'current_weights','cuda');baseline=attention(data,obs,source_ids,source_weights)
   n=teacher.shape[1];actual_set=torch.zeros(g,n+1,device='cuda',dtype=torch.bool).scatter_(1,ids.long().masked_fill(ids<0,n),ids>=0)[:,:n];ref_set=torch.zeros(g,n+1,device='cuda',dtype=torch.bool).scatter_(1,reference.long().masked_fill(reference<0,n),reference>=0)[:,:n];inter=(actual_set&ref_set).sum(1).float();union=(actual_set|ref_set).sum(1).float();captured=(teacher*actual_set).sum(1);ref_captured=(teacher*ref_set).sum(1)
   rows.append(dict(name=c['name'],layer=entry['layer'],**metrics(pred,teacher,history,data['patch_cap']),mask_jaccard=float((inter/union.clamp_min(1)).mean()),captured_mass_vs_reference=float((captured/ref_captured.clamp_min(1e-20)).mean()),projected_error_vs_teacher_mask=error(data,out,target),projected_error_vs_dense=error(data,out,dense),teacher_mask_error_vs_dense=error(data,target,dense),baseline_recorded_mask_error_vs_dense=error(data,baseline,dense)))
   old_ids[c['name']]=ids
   if entry['layer']==0 and c['bootstrap']:
    from scripts.replay_history_probability import block_sum
    old[c['name']]=dict(phi=teacher.clone(),coarse=obs['coarse'],block_mass=block_sum(teacher,obs['grid'],obs['regions']),sample_conditional=obs.get('sample_conditional'))
   else:old[c['name']]=update(data,obs,ids,pred,c['mode'],c['eta'])
  print('DECISION',manifest['sequence'],entry['layer'],flush=True)
 summary=[]
 for c in configs:
  rr=[r for r in rows if r['name']==c['name']];summary.append(dict(config=c,**{k:statistics.mean(r[k] for r in rr) for k in ['relative_l1','relative_l2','js','cosine','top_budget_mass_ratio','mask_jaccard','captured_mass_vs_reference','projected_error_vs_teacher_mask','projected_error_vs_dense','teacher_mask_error_vs_dense','baseline_recorded_mask_error_vs_dense']}))
 summary.sort(key=lambda r:r['projected_error_vs_teacher_mask']);save(a.output,dict(quality=quality_record({},'Same-QKV mask/cache rollout and sampled Attention diagnostic only;no geometry'),sequence=manifest['sequence'],summary=summary,rows=rows,speedup_vs_dense=None,speedup_reason='Offline replay,not full model runtime',scope='Candidate masks and probability caches roll across recorded source QKV;teacher mask uses exact fast4 probability with the same current historical IDs;not full neural rollout'));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
