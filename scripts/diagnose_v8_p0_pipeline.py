"""All23 changed global layers: compare captured P0 routing with original eager pipeline."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_v8_p0 import build
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_window_graph as graph
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(ROOT/'configs/evaluation/reuse_v8_p0.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();original=graph.select;seen=set();checks=[]
def select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,pop,geo):
 out=original(s,layer,prior,pi,old_ids,allowed,z,reps,labels,pop,geo)
 if layer not in seen:
  seen.add(layer);reference_geo=dict(geo);reference_geo['p0_loss_pack']=False
  expected=graph.eager(prior,pi,old_ids,allowed,z,reps,labels,pop,reference_geo,s.budget,s.config.dual_iterations,s.config.drop_multiplier)
  mismatch=int((out[0]!=expected[0]).sum());print('LAYER',layer,'key mismatch',mismatch,flush=True)
  assert torch.equal(out[1],expected[1]),('padding',layer)
  error=float((out[2]-expected[2]).abs().max());rel=float((out[2]-expected[2]).norm()/expected[2].norm().clamp_min(1e-30));pass # record failed-candidate weight difference without hiding key mismatches
  for name in ['state_0_clusters','state_1_clusters','state_5_clusters','state_10_clusters','counts','reuse_edges','representatives','exact_edges','dropped_clusters']:
   pass # record state differences below
  checks.append(dict(layer=layer,groups=len(prior),key_order_exact=mismatch==0,key_mismatch=mismatch,selected_state_counts_exact=all(torch.equal(out[3][k],expected[3][k]) for k in ['state_0_clusters','state_1_clusters','state_5_clusters','state_10_clusters']),padding_exact=True,weights_max_abs=error,weights_relative_l2=rel))
 return out
graph.select=select;begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize();assert len(checks)==23
result=dict(complete=True,equivalence_passed=all(x['key_order_exact'] for x in checks),checks=checks,scope='First64-group routing window of every changed layer,L1–23;L0 seed unchanged',diagnostic_wall_seconds=time.perf_counter()-begin,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Captured-vs-eager routing diagnostic; no geometric quality evaluation',dense_speedup=None,dense_reason='Extra reference execution; diagnostic wall is not benchmark')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
