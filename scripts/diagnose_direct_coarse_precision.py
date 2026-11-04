"""No GT tuning: isolate shared FP32 versus legacy BF16 coarse QK on identical inputs."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_direct_coarse_observation as coarse,reuse_residual_exploration as impl
from saf3r.triton.reuse_coarse_observation import patch
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(ROOT/'configs/evaluation/reuse_v8_direct_sparse.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();original_prepare=impl.ReuseExploreState.prepare;original_observe=coarse.observe;current={};rows=[]
def prepare(s,k,features):
 out=original_prepare(s,k,features);current.update(pooled_bf16=out[2],overlap=out[3]);return out
def observe(s,q,k,start,g,scale):
 obs=original_observe(s,q,k,start,g,scale)
 if start==0 and s.previous_layer>=0:
  with torch.autocast('cuda',dtype=torch.bfloat16):dot=obs['probe_q']@current['pooled_bf16'][:,None].transpose(-1,-2)
  old_prob,old_demand=patch(dot,s.grid_log_sizes,s.previous_grid[:g],s.previous_real_grid[:g],scale)
  new_prob=obs['patch_probability'];previous=s.previous_grid[:g].float();confidence=torch.where(previous>1e-12,(new_prob/previous.clamp_min(1e-12)).clamp_max(1),0);new_demand=(new_prob-confidence*s.previous_real_grid[:g].float()).clamp_min(0)
  old_residual=torch.sparse.mm(current['overlap'],old_demand.T).T;new_residual=torch.sparse.mm(current['overlap'],new_demand.T).T;count=max(1,math.ceil(s.fine.base_count*s.config.candidate_fraction));oa=old_residual.topk(count,dim=1).indices;na=new_residual.topk(count,dim=1).indices
  mask=torch.zeros_like(old_residual,dtype=torch.bool).scatter_(1,oa,True);common=mask.gather(1,na).sum(1).float();fraction=1-common/count
  old_logits=dot.float().permute(1,0,2,3)*scale;error=(old_logits-obs['grid_logits']).abs()
  rows.append(dict(layer=s.previous_layer+1,groups=g,logit_max_abs=float(error.max()),patch_probability_l1_mean=float((old_prob-new_prob).abs().sum(-1).mean()),exact_candidate_replaced_fraction_mean=float(fraction.mean()),exact_candidate_replaced_fraction_max=float(fraction.max()),candidate_count=count))
 return obs
impl.ReuseExploreState.prepare=prepare;coarse.observe=observe;begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize();assert len(rows)==23
result=dict(complete=True,rows=rows,scope='heads128 Direct trajectory,first64-group window/L1–23;same Q/K/pooled means and previous Grid statistics;alternative coarse outputs not fed into model',conclusion='This isolates candidate sensitivity to the precision change,not its geometric causal contribution. Full-dataset intervention not run,no quality rerun/tuning.',diagnostic_wall_seconds=time.perf_counter()-begin,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Matched numerical candidate sensitivity diagnostic;no geometry evaluation',dense_speedup=None,dense_reason='Extra oracle computations;not performance benchmark')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
