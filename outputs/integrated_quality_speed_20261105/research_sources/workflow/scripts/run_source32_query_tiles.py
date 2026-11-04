"""Three fixed Query tiles, identical stages/warps/method, full timing + six metrics."""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source32_query_tiles_20261105'
base=json.loads((ROOT/'configs/evaluation/reuse_shared_value_loss_q256.json').read_text())
if not (OUT/'initial_default.json').exists():(OUT/'initial_default.json').write_text(json.dumps(base,indent=2)+'\n')
configs={}
for bm in (32,64,128):
 c=dict(base,attention_block_queries=bm,attention_stages=2,attention_warps=4)
 path=ROOT/'configs/evaluation'/f'reuse_shared_value_loss_q256_source32_bq{bm}_s2.json';path.write_text(json.dumps(c,indent=2)+'\n');configs[bm]=path
variants=[(f'bq{bm}',configs[bm]) for bm in (32,64,128)]
probe=json.loads((OUT/'resource_probe/result.json').read_text());assert probe['passed']
for bm in (64,128):
 winner=probe['winners'][str(bm)]
 c=dict(base,attention_block_queries=bm,attention_stages=winner['num_stages'],attention_warps=winner['num_warps'])
 path=ROOT/'configs/evaluation'/f'reuse_shared_value_loss_q256_source32_bq{bm}_resource.json';path.write_text(json.dumps(c,indent=2)+'\n')
 variants.append((f'bq{bm}_resource',path))
(OUT/'plan.json').write_text(json.dumps(dict(scenes=['heads','chess','stairs'],frames=128,tiles=[32,64,128],fixed_ablation_num_stages=2,fixed_ablation_num_warps=4,history_split_export=False,
 query_group_size=256,source_width=32,clustering='source_local_unanchored/sequential',retention=.2,local_iterations=2,
 variants=[dict(name=n,config=str(p)) for n,p in variants],resource_probe_selection='One candidate per larger tile, selected on fixed-input kernel time and tolerance checks only; config differs in stages/warps and is reported separately from fixed4warps/stages2 ablation',normal_timing='1 warmup + 3 full forwards',attention_timing='3 additional lightly-instrumented full forwards; choose whole-wall median pass and keep its complete partition',
 quality='pose + recon_unposed on last NORMAL output; all six metrics',
 selection='Fastest three-scene full-forward mean among predeclared tile sizes passing fixed-input operator tolerances and no execution failures. No tuning against 7Scenes test truth. Report all quality outcomes.',
 controls='Dense/SAF3R unchanged, not rerun. Dense speedup N/A without current paired timing.',
 pipeline_note='Legacy BM32 ignores config.attention_stages and uses 2. All three variants explicitly set config stages2, yielding actual stages2 for every tile.'),indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=['heads','chess','stairs']);p.add_argument('--gpus',nargs='+',type=int,default=[1,3,6]);p.add_argument('--variants',nargs='+');a=p.parse_args()
assert len(a.scenes)==len(a.gpus)
if a.variants:variants=[v for v in variants if v[0] in a.variants]
operators=json.loads((OUT/'operators/result.json').read_text());assert operators['passed']
monitor=json.loads((OUT/'operators.monitor.json').read_text());assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
def worker(scene,gpu):
 rows=[]
 for label,configuration in variants:
  bm=json.loads(configuration.read_text())['attention_block_queries']
  target=OUT/scene/label;prefix=OUT/f'{scene}_{label}'
  if (target/'result.json').exists():rows.append(dict(scene=scene,variant=label,tile=bm,existing=True));continue
  if target.exists():raise RuntimeError('Preserve incomplete attempt '+str(target))
  cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(prefix)+'.monitor.json','--log',str(prefix)+'.log','--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','7scenes','--scene',scene,'--frames','128','--method','zdrop_history_states','--config',str(configuration),'--output-dir',str(target),'--warmup','1','--repeats','3','--attention-timing','3']
  print('START',scene,label,gpu,flush=True);proc=subprocess.run(cmd,cwd=ROOT);row=dict(scene=scene,variant=label,tile=bm,gpu=gpu,returncode=proc.returncode);rows.append(row);print('END',row,flush=True)
  (OUT/f'{scene}_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
  if proc.returncode:break
 return rows
with ThreadPoolExecutor(max_workers=len(a.scenes)) as pool:rows=list(pool.map(lambda pair:worker(*pair),zip(a.scenes,a.gpus)))
(OUT/('coordinator_'+'_'.join(a.scenes)+('_'+'_'.join(a.variants) if a.variants else '')+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
if any(r.get('returncode',0) for group in rows for r in group):sys.exit(1)
