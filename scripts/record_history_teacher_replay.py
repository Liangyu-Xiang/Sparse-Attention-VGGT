"""RGB-only sparse-trajectory replay with exact local fast4 probability targets.

Stores diagnostic features locally,not model weights or geometric test truth.
No full Dense/fast/SAF3R control is evaluated and no timing is claimed.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['scannet','tum'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from saf3r.patch import reuse_residual_exploration as mod,reuse_window_graph as window
 from scripts.quality_metrics import quality_record
 plan=json.loads((ROOT/'outputs/quality_speed_search_20261106/phase5/plan.json').read_text());entry=plan['calibration'][a.sequence];out=a.output;out.mkdir(parents=True,exist_ok=False);quality=quality_record({},'RGB sparse-trajectory feature replay and sampled fast4 Dense probability targets only;no geometric GT or quality evaluation')
 save(out/'quality.json',quality);torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 config=json.loads((ROOT/'configs/evaluation/streamed_history_block64_calibrated_20261106.json').read_text());cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint'];model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',entry['image_paths']).cuda();state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')}
 original_prepare=mod.prepare_qkv;original_select=window.select;records=[];active={};handles=[]
 def prepare(original,x,pos):
  q,k,v=original_prepare(original,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous();s=state;layer=s.previous_layer+1;groups=torch.linspace(0,s.groups-1,8,device=q.device).long().unique();positions=(groups[:,None]*256+torch.tensor([32,96,160,224],device=q.device)).clamp_max(s.n-1)
  with torch.autocast('cuda',enabled=False):
   raw=q[:,positions.flatten()].float()@k.float().transpose(1,2)*original.scale
   p=raw.softmax(-1).mean(0).reshape(len(groups),4,s.n)
   dense=(raw.softmax(-1)@v.float()).reshape(len(q),len(groups),4,v.shape[-1])
  active.clear();active.update(layer=layer,groups=groups.cpu(),positions=positions.cpu(),q=q[:,positions].cpu(),k=k.cpu(),v=v.cpu(),teacher_per_probe=p.cpu(),teacher4=p.mean(1).cpu(),teacher_output=dense.cpu(),patches=s.patches.cpu(),specials=s.specials.cpu(),scale=original.scale,projection_weight=original.proj.weight.detach().cpu(),config=config,old_cache=None if s.streamed_history_cache is None else {key:value[groups].cpu() for key,value in s.streamed_history_cache.items()},windows=[],start=0)
  return q,k,v
 def select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,**kwargs):
  result=original_select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,**kwargs)
  # This wrapper is outside graph capture;never copies features from inside a graph.
  if active['start']==0:active.update(z=z.cpu(),reps=reps.cpu(),labels=labels.cpu(),population=population.cpu(),patch_cap=s.budget)
  start=active['start'];g=len(prior);groups=active['groups'];pick=(groups>=start)&(groups<start+g);local=(groups[pick]-start).to(prior.device)
  if len(local):active['windows'].append(dict(group_rows=pick.nonzero().flatten(),prior=prior[local].cpu(),old_ids=old_ids[local].cpu(),current_ids=result[0][local].cpu(),current_weights=result[2][local].cpu()))
  active['start']+=g;return result
 def finish(module,args):
  layer=active['layer'];groups=active['groups'].to('cuda');active['new_cache']={key:value[groups].cpu() for key,value in state.streamed_history_cache.items()}
  data=dict(active);data.pop('start');path=out/f'layer{layer:02}.pt';torch.save(data,path);records.append(dict(layer=layer,path=path.name,bytes=path.stat().st_size));print('REPLAY',a.sequence,layer,records[-1]['bytes'],flush=True)
 mod.prepare_qkv=prepare;window.select=select
 for block in model.aggregator.global_blocks:handles.append(block.attn.original.proj.register_forward_pre_hook(finish))
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 finally:
  mod.prepare_qkv=original_prepare;window.select=original_select
  for h in handles:h.remove()
 assert len(records)==24 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in source.items())
 save(out/'manifest.json',dict(complete=True,sequence=a.sequence,input=entry,checkpoint_sha256=plan['checkpoint_sha256'],config=config,source_sha256=source,layers=records,quality=quality,speedup_vs_dense=None,speedup_reason='Instrumented RGB replay only;not a runtime/geometry run',teacher='Head/probe mean of four current actual Query full-Key softmax probabilities on the SAME sparse trajectory QKV',query_groups_per_layer=8,test_gt_read=False))
if __name__=='__main__':main()
