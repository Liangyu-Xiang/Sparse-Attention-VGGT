"""Teacher-forced same-QKV output errors for A/B; no Dense oracle in inference."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_loss_20261105/common_qkv'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 import torch,yaml
 import torch.nn.functional as F
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration,ReuseResidualExploration
 from saf3r.patch import reuse_window_graph as window
 from saf3r.patch.rep_history_loss import select as ab_select
 from saf3r.triton import reuse_direct_sampled_probability as kernel
 from scripts.quality_metrics import quality_record,quality_markdown
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 quality=quality_record({},'Teacher-forced same-Q/K/V attention diagnostic,32 ordinary Queries per sampled layer; pose/recon_unposed and production speed not evaluated')
 save(OUT/'quality.json',quality);(OUT/'quality.md').write_text(quality_markdown(quality))
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');inp=json.loads((ROOT/'outputs/eth3d_layout_fixed_full_20261105/runs/pipes/input.json').read_text());images=load_images(model,'vggt',inp['image_paths']).cuda();save(OUT/'input.json',inp)
 config=json.loads((ROOT/'outputs/rep_history_loss_20261105/calibration/configs/baseline.json').read_text());state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config))
 target={6,11,13,19};seen=set();pending={};current={};rows=[]
 original_forward=ReuseResidualExploration.forward;original_attention=kernel.direct_sparse_attention_sampled;original_select=window.select
 def forward(module,*args,**kw):
  current['module']=module;return original_forward(module,*args,**kw)
 def selector(s,layer,prior,pi,old,allowed,z,reps,labels,pop,geometry,uniform=None,special_p=None):
  result=window.eager(prior,pi,old,allowed,z,reps,labels,pop,geometry,s.budget,s.config.dual_iterations,s.config.drop_multiplier,uniform=uniform,special_p=special_p)
  if layer in target and layer not in seen:
   gi=min(1,len(prior)-1);sl=slice(gi,gi+1);masks={'baseline':(result[0][sl].clone(),result[2][sl].clone())}
   for method in ['A','B']:
    geo=dict(geometry,rep_history_loss=method,rep_history_diagnostics=False)
    ids,_,weights,_=ab_select(prior[sl],pi[sl],old[sl],allowed[sl],z,reps,labels,pop,geo,s.budget,s.config.dual_iterations,uniform[sl])
    masks[method]=(ids.clone(),weights.clone())
   pending.update(layer=layer,group=gi,masks=masks);seen.add(layer)
  return result
 def attention(q,k,v,ids,specials,scale,*args,**kw):
  result,evidence=original_attention(q,k,v,ids,specials,scale,*args,**kw)
  if pending:
   layer=pending['layer'];gi=pending['group'];lo=gi*config['query_group_size'];hi=min(lo+config['query_group_size'],q.shape[1])
   eligible=torch.arange(lo,hi,device=q.device);eligible=eligible[eligible.remainder(state.height*state.width+5)>=5]
   take=eligible[torch.linspace(0,len(eligible)-1,32,device=q.device).long()];qq=q[:,take]
   dense=F.scaled_dot_product_attention(qq[None],k[None],v[None],dropout_p=0.,scale=scale)[0].float()
   original=current['module'].original
   def projected(x):return F.linear(x.transpose(0,1).reshape(len(take),-1).float(),original.proj.weight.float(),None if original.proj.bias is None else original.proj.bias.float())
   def metrics(a,b):
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),cosine=float(F.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1)).item()))
   for method,(patch_ids,weight) in pending['masks'].items():
    valid=patch_ids[0]>=0;actual=state.patches[patch_ids[0,valid].long()]
    keys=torch.cat((specials,actual));bias=torch.cat((torch.zeros(len(specials),device=q.device),weight[0,valid].log()))
    sparse=F.scaled_dot_product_attention(qq[None],k[:,keys][None],v[:,keys][None],attn_mask=bias[None,None,None,:],dropout_p=0.,scale=scale)[0].float()
    row=dict(layer=layer,method=method,group=gi,queries=len(take),all_heads=q.shape[0],patch_key_count=len(actual),special_key_count=len(specials),pre_projection=metrics(sparse,dense),post_projection=metrics(projected(sparse),projected(dense)),patch_keys=patch_ids[0,valid].cpu().tolist(),weights=weight[0,valid].cpu().tolist())
    if method=='baseline':row['actual_kernel_vs_weighted_reference']=metrics(result[:,take].float(),sparse)
    rows.append(row)
   pending.clear();save(OUT/'result.json',dict(quality=quality,rows=rows,scope='pipes,14 frames; layers6/11/13/19, first window group1,32 fixed ordinary Queries,all heads/full Key bank; same current QKV, M5c priors, clusters and history from baseline trajectory. Dense used only after mask selection for diagnostic errors; not full-model quality, not global optimality, not deployment QK.'))
  return result,evidence
 ReuseResidualExploration.forward=forward;window.select=selector;kernel.direct_sparse_attention_sampled=attention
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 finally:ReuseResidualExploration.forward=original_forward;window.select=original_select;kernel.direct_sparse_attention_sampled=original_attention
 assert len(rows)==12 and seen==target
 print('SAME_QKV_DIAGNOSTIC_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
