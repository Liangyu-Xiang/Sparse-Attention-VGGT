"""Fresh seven-scene Dense/DINO64/lazy4 measurement and comparison extension."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_record,quality_cells
from scripts.evaluate_attention_first_residual_value import save,write_quality
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
METHODS=('dense','saf3r_dino64','saf3r_lazy4')
OUT=ROOT/'outputs/saf3r_matched_reuse_comparison_20261029'
def scene_worker(gpu,scene):
 for method in METHODS:
  dest=OUT/scene/method
  if (dest/'status.json').exists():continue
  cmd=[sys.executable,'scripts/evaluate_saf3r_matched.py','--scene',scene,'--method',method,'--frames','128','--output-dir',str(dest)]
  baseline=OUT/scene/'dense/forward.json'
  if method!='dense' and baseline.exists():cmd+=['--baseline',str(baseline)]
  with (OUT/'logs'/f'{scene}_{method}.log').open('w') as log:
   rc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT).returncode
  if not (dest/'status.json').exists():
   dest.mkdir(parents=True,exist_ok=True);write_quality(dest,quality_record({},f'Process exited {rc} before status initialization; see log'));save(dest/'status.json',dict(stage='failed',error=f'exit={rc}'))
def report(terminal=False):
 rows=[];complete={};lines=['# SAF3R addition to reuse comparison','', 'Local 7Scenes all seven scenes, uniform128 identical frames/checkpoint/preprocessing and pose/recon_unposed protocol. Fresh paired Dense for this cohort. Warmup1 +3 complete forwards; per-forward routing reset and DINO Top-K construction included. Distances metres, F1 threshold0.05m, downsample0.0078125m, CD=(ACC+Comp)/2 unsquared. DINO64 is a declared paper-mechanism configuration, not a recovered unpublished scene-to-K mapping. ETH3D calibrated head profile. lazy4 is separately labelled. No published reference rows.','', '|Scene|Method|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Median s|Peak GiB|Speedup|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for scene in SCENES:
  for method in METHODS:
   dest=OUT/scene/method;result=dest/'result.json';r=json.loads(result.read_text()) if result.exists() else None
   q=r['quality'] if r else (json.loads((dest/'quality.json').read_text()) if (dest/'quality.json').exists() else quality_record({},'Queued; not complete'))
   if r and q['complete']:complete[scene,method]=r
   rows.append(dict(scene=scene,method=method,quality=q,source=str(dest)))
   perf=['N/A']*3 if not r else [f"{r['timing']['median_seconds']:.4f}",f"{max(t['peak_allocated_gib'] for t in r['timing']['samples']):.4f}",'N/A' if q.get('speedup_vs_vggt') is None else f"{q['speedup_vs_vggt']:.4f}x"]
   lines.append('| '+' | '.join([scene,method]+quality_cells(q)+perf)+' |')
 macro=[]
 for method in METHODS:
  rs=[complete[s,method] for s in SCENES if (s,method) in complete];item=dict(method=method,quality_scenes_complete=len(rs),required_scenes=7,values=None)
  if len(rs)==7:item.update(values={k:float(np.mean([r['quality']['values'][k] for r in rs])) for k in rs[0]['quality']['values']},mean_scene_median_seconds=float(np.mean([r['timing']['median_seconds'] for r in rs])),mean_scene_peak_allocated_gib=float(np.mean([np.median([t['peak_allocated_gib'] for t in r['timing']['samples']]) for r in rs])))
  macro.append(item)
 dense=macro[0]
 lines+=['','|Method macro|Scenes|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Median s|Peak GiB|Speedup|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for item in macro:
  q=quality_record({},'All seven complete quality runs required');perf=['N/A']*3
  if item['values'] is not None:
   q['values']=item['values'];speed=None if dense['values'] is None else dense['mean_scene_median_seconds']/item['mean_scene_median_seconds'];item['speedup_ratio_of_mean_scene_medians']=speed
   perf=[f"{item['mean_scene_median_seconds']:.4f}",f"{item['mean_scene_peak_allocated_gib']:.4f}",'N/A' if speed is None else f'{speed:.4f}x']
  lines.append('| '+' | '.join([item['method'],f"{item['quality_scenes_complete']}/7"]+quality_cells(q)+perf)+' |')
 save(OUT/'summary.json',dict(macro=macro,per_run=rows,terminal=terminal));(OUT/'report.md').write_text('\n'.join(lines)+'\n')
 jobs=[]
 for row in rows:
  p=Path(row['source'])/'status.json';jobs.append(dict(scene=row['scene'],method=row['method'],state=json.loads(p.read_text())['stage'] if p.exists() else 'queued'))
 save(OUT/'queue_status.json',dict(jobs=jobs,all_terminal=terminal))
def consolidate():
 dest=ROOT/'outputs/reuse_variants_consolidated_20261029';original=json.loads((dest/'summary.json').read_text());extra=json.loads((OUT/'summary.json').read_text())
 original['saf3r_comparison']=extra;save(dest/'summary.json',original)
 text=(dest/'report.md').read_text().split('\n## SAF3R addition')[0]
 text+='\n## SAF3R addition (fresh paired timing cohort)\n\nDINO64: final DINO cosine similarity per-frame Top-64, max_topk=8192, no truncation at128frames; lazy4: first-correspondence head Top-4. Both use ETH3D calibrated profile. Fresh paired Dense timing is used only for these rows. Earlier reuse rows retain their own paired Dense timings; timing cohorts are not silently rebased. All inputs/protocols are audited against original reuse campaign.\n\n| Method | Scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean median s | Mean peak GiB | Speedup vs paired Dense |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for r in extra['macro']:
  q=quality_record({},'Incomplete');perf=['N/A']*3
  if r['values'] is not None:
   q['values']=r['values'];perf=[f"{r['mean_scene_median_seconds']:.6f}",f"{r['mean_scene_peak_allocated_gib']:.6f}",f"{r['speedup_ratio_of_mean_scene_medians']:.6f}x"]
  text+='| '+' | '.join([('Fresh paired Dense' if r['method']=='dense' else r['method']),f"{r['quality_scenes_complete']}/7"]+quality_cells(q)+perf)+' |\n'
 (dest/'report.md').write_text(text)
def audit():
 checks=[]
 for scene in SCENES:
  reference=json.loads((ROOT/'outputs/reuse_single_change_7scenes128_20261029'/scene/'dense/forward.json').read_text())
  for method in METHODS:
   p=OUT/scene/method/'forward.json'
   if not p.exists():continue
   r=json.loads(p.read_text())
   for key in ('input','hardware','quality_protocol'):assert r[key]==reference[key],f'{scene}/{method} {key} mismatch'
   checks.append(dict(scene=scene,method=method,exact_pairing_passed=True))
 save(OUT/'pairing_audit.json',checks)
def main():
 OUT.mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 save(OUT/'manifest.json',dict(scenes=SCENES,methods=METHODS,frames=128,warmup=1,repeats=3,tasks=['pose','recon_unposed'],gpus=list(range(7)),timing_cohort='fresh paired Dense/DINO64/lazy4',source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/evaluate_saf3r_matched.py','scripts/run_saf3r_comparison.py','configs/evaluation/eval_saf3r_vggt.yaml','configs/evaluation/scannet50_paper_dino64.yaml','saf3r/patch/patch_vggt.py','saf3r/patch/utils.py']}))
 with ThreadPoolExecutor(max_workers=7) as pool:
  pending={pool.submit(scene_worker,g,s) for g,s in enumerate(SCENES)}
  while pending:
   done,pending=wait(pending,timeout=15)
   for future in done:future.result()
   report(not pending)
 audit();consolidate()
 print('COMPARISON_FINISHED',flush=True)
if __name__=='__main__':main()
