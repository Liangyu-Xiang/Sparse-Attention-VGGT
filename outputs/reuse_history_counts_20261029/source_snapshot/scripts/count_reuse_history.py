"""Replay original supports and count per-cluster history; quality not evaluated."""
import os,sys,json,time,traceback,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
OUT=ROOT/'outputs/reuse_history_counts_20261029';REFERENCE=ROOT/'outputs/reuse_single_change_7scenes128_20261029'
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))
def scene_run(scene):
 import torch,numpy as np
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 import saf3r.patch.reuse_residual_exploration as module
 from scripts.quality_metrics import quality_record
 from scripts.evaluate_attention_first_residual_value import write_quality
 out=OUT/scene
 out.mkdir(parents=True,exist_ok=True);write_quality(out,quality_record({},'History-count diagnostic replay only; no pose/recon_unposed quality or paired warmed Dense timing'))
 save(out/'status.json',dict(stage='loading'));torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inp=json.loads((REFERENCE/scene/'original/input.json').read_text());cfg=Dict(json.loads((REFERENCE/scene/'original/config.json').read_text())['model'])
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inp['image_paths']).cuda();assert list(images.shape)==inp['shape']
 state=module.install_reuse_residual_exploration(model,module.ReuseExploreConfig())
 original=module.pack_support;histograms={};windows={};current={}
 def counted(selected,hids,reps,labels,population,budget):
  layer=state.previous_layer+1;g,f=selected.shape;limit=hids.shape[-1]
  t=torch.where((selected>=2)&(selected<=limit+1),selected-1,torch.where((selected>=limit+2)&(selected<=2*limit+1),selected-limit-1,0))
  n=((torch.arange(limit,device=selected.device)[None,None]<t[:,:,None])&(hids>=0)).sum(-1)
  if layer not in histograms:
   histograms[layer]=torch.zeros((f,5),device=selected.device,dtype=torch.int64);windows[layer]=0
   current[layer]=dict(population=population.detach().clone(),representatives=reps.detach().clone())
  cluster=torch.arange(f,device=selected.device)[None].expand(g,-1)
  histograms[layer].view(-1).scatter_add_(0,(cluster*5+n).reshape(-1),torch.ones(g*f,device=selected.device,dtype=torch.int64))
  windows[layer]+=g
  return original(selected,hids,reps,labels,population,budget)
 module.pack_support=counted
 save(out/'input.json',inp);save(out/'status.json',dict(stage='counting'))
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();elapsed=time.perf_counter()-begin
 arrays={};rows=[]
 # L0 has no history selector and emits only population representatives.
 f=state.fine.base_count;g=state.groups
 arrays['layer00_histogram']=np.stack((np.full(f,g,dtype=np.int64),*[np.zeros(f,dtype=np.int64) for _ in range(4)]),1)
 for layer,hist in histograms.items():
  arrays[f'layer{layer:02d}_histogram']=hist.cpu().numpy()
  arrays[f'layer{layer:02d}_population']=current[layer]['population'].cpu().numpy()
  arrays[f'layer{layer:02d}_representative_patch_id']=current[layer]['representatives'].cpu().numpy()
 old=json.loads((REFERENCE/scene/'original/layers.json').read_text())
 for layer in range(24):
  h=arrays[f'layer{layer:02d}_histogram'];tot=h.sum(0);count=int(tot.sum());mean=float(tot@np.arange(5)/count)
  assert (h.sum(1)==g).all()
  rows.append(dict(layer=layer,clusters=f,query_groups=g,cluster_query_pairs=count,histogram=tot.tolist(),fractions=(tot/count).tolist(),mean_reused_per_cluster_query_group=mean,reused_total=int(tot@np.arange(5)),mean_reused_per_query_group=float(tot@np.arange(5)/g),saved_run_mean_per_cluster=old[layer]['reuse_edges_mean']/old[layer]['fine_clusters']))
 np.savez_compressed(out/'per_cluster_counts.npz',**arrays)
 save(out/'counts.json',dict(scene=scene,layer_rows=rows,scope='fresh original replay; histogram row is one current-layer fine cluster, columns0..4 are counts across query groups; exact-expansion edges are not counted as reused history',runtime_seconds=elapsed,timing_scope='cold diagnostic forward with counters; not warmed speed comparison',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality_available=False,speedup_vs_dense=None,speedup_unavailable='No matched warmed Dense timings for counting diagnostic',config=json.loads((REFERENCE/scene/'original/config.json').read_text()),hardware=dict(gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))))
 save(out/'status.json',dict(stage='diagnostic_complete'))
def summarize():
 import numpy as np
 results=[]
 for scene in SCENES:
  p=OUT/scene/'counts.json'
  if p.exists():results.append(json.loads(p.read_text()))
 save(OUT/'summary.json',dict(scenes_complete=len(results),required_scenes=7,per_scene=results))
 if len(results)<7:return
 lines=['# Original reuse: history counts','', '7Scenes all seven scenes,128frames. Counting replays only; no new quality or speed comparison. Each cluster/query-group has0..4 reused historical Key IDs; full exact-expansion members are separate. Layer IDs0..23. Percentages are equal-scene macro of cluster/query-group frequencies.','', '| Layer | 0 history % | 1 % | 2 % | 3 % | 4 % | Mean history / cluster / query group |','|---|---:|---:|---:|---:|---:|---:|']
 for layer in range(24):
  frac=np.mean([r['layer_rows'][layer]['fractions'] for r in results],0);mean=float(frac@np.arange(5));lines.append('| '+str(layer)+' | '+' | '.join(f'{100*x:.6f}' for x in frac)+f' | {mean:.6f} |')
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
def main():
 OUT.mkdir(exist_ok=True)
 procs=[]
 for gpu,scene in enumerate(SCENES):
  log=(OUT/f'{scene}.log').open('w');p=subprocess.Popen([sys.executable,__file__,'--scene',scene],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
 while any(p.poll() is None for p,_ in procs):summarize();time.sleep(15)
 for p,log in procs:log.close()
 summarize();save(OUT/'status.json',dict(all_terminal=True,exit_codes=[p.returncode for p,_ in procs]))
if __name__=='__main__':
 if '--scene' in sys.argv:
  scene=sys.argv[sys.argv.index('--scene')+1]
  try:scene_run(scene)
  except Exception as exc:
   save(OUT/scene/'status.json',dict(stage='failed',error=str(exc),traceback=traceback.format_exc()));raise
 else:main()
