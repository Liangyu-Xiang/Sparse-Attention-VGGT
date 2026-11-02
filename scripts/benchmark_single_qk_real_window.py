"""Matched old/new attention wrappers on one identical real layer/window."""
import json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FORK=ROOT/'outputs/headmean_single_qk_quality_20261102/source';sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
import saf3r.patch.reuse_residual_exploration as impl
import saf3r.triton.reuse_single_qk_grid as new
from saf3r.triton.reuse_explore_av_ready import weighted_attention as old
OUT=ROOT/'outputs/single_qk_real_window_causal_v2_20261102'
def difference(a,b):
 d=a.float()-b.float();return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.float().norm().clamp_min(1e-30)))
def main():
 if OUT.exists():raise ValueError('Fresh output required')
 OUT.mkdir();write_quality(OUT,quality_record({},'Kernel-only matched causal diagnostic; no pose/reconstruction evaluation'));save(OUT/'status.json',dict(stage='loading'))
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((FORK/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(FORK/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 inp=json.loads((ROOT/'outputs/headmean_single_qk_quality_20261102/7scenes/heads/headmean_single_qk/input.json').read_text())
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inp['image_paths']).cuda();state=impl.install_reuse_residual_exploration(model,impl.ReuseExploreConfig(**json.loads((FORK/'configs/evaluation/reuse_zdrop_history_states.json').read_text())))
 original=new.weighted_attention;done=[]
 def wrapper(q,k,v,ids,bias,special,scale,**kw):
  if state.previous_layer==11 and q.shape[1]==2048 and not done:
   done.append(True)
   def previous():
    output,edge,probe=old(q,k,v,ids,bias,special,scale)
    summary=torch.zeros((len(ids),2,kw['regions']),device=q.device)
    summary[:,0].scatter_add_(1,kw['key_grid'],edge)
    summary[:,1].scatter_add_(1,kw['key_grid'],edge/bias.exp().clamp_min(1e-30))
    return output,summary,probe
   functions={'two_qk':previous,'one_qk':lambda:original(q,k,v,ids,bias,special,scale,**kw)};samples={key:[] for key in functions}
   for i in range(5):
    for name,fn in functions.items():
     torch.cuda.synchronize();begin=time.perf_counter();value=fn();torch.cuda.synchronize()
     if i>=2:samples[name].append(time.perf_counter()-begin)
     del value
   x,target,rawold=functions['two_qk']();y,grid,rawnew=functions['one_qk']();regions=kw['regions'];key_grid=kw['key_grid']
   count=torch.zeros((len(ids),regions),device=q.device).scatter_add_(1,key_grid,(ids>=0).float());oldtiles=len(ids)*((ids.shape[-1]+63)//64);newtiles=int(((count+63)//64).sum())
   finite=torch.isfinite(rawold);result=dict(scope='Same real heads128 L12 first64-group window; same Q,K,V,IDs,bias and CUDA device; both wrappers return output, two unnormalized grid summaries and probe logits. Old grid aggregation is included in timing. Two warmups and three measured wrapper calls each, alternating order. Not complete-forward speedup.',shape=dict(query=list(q.shape),key=list(k.shape),support=list(ids.shape),regions=regions),samples=samples,median_seconds={name:statistics.median(s) for name,s in samples.items()},single_qk_speed_ratio=statistics.median(samples['two_qk'])/statistics.median(samples['one_qk']),old_patch_key_tiles=oldtiles,new_patch_key_tiles=newtiles,padded_patch_tile_ratio=newtiles/oldtiles,output_difference=difference(x,y),grid_difference=difference(target,grid),raw_probe_difference=difference(rawold.masked_fill(~finite,0),rawnew.masked_fill(~finite,0)),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel-only causal diagnostic, no GT quality',dense_speedup=None,dense_reason='No unchanged Dense rerun')
   save(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)
  return original(q,k,v,ids,bias,special,scale,**kw)
 new.weighted_attention=wrapper;save(OUT/'status.json',dict(stage='forward'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();assert done;save(OUT/'status.json',dict(stage='diagnostic_complete'))
if __name__=='__main__':main()
