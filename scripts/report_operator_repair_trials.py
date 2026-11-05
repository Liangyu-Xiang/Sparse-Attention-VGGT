"""Audit and report both frozen operator-repair calibration batches, without score tuning."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_record,quality_markdown
KEYS=['auc03','auc30','acc','comp','cd','f1'];SCENES=['pipes','courtyard','playground']
PHASES=['operator_repair_20261105','operator_repair_phase2_20261105']
OUT=ROOT/'outputs/operator_repair_phase2_20261105'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 baseline=ROOT/'outputs/eth3d_layout_fixed_full_20261105';rows=[];macros=[];audits=[];runtime=[]
 groups=[('layout_fixed',baseline,None)]
 for phase in PHASES:
  folder=ROOT/'outputs'/phase;manifest=read(folder/'frozen_manifest.json')
  groups.extend((v,folder/'quality'/v,manifest) for v in manifest['variants'])
 for variant,folder,manifest in groups:
  group=[]
  for scene in SCENES:
   p=folder/'runs'/scene/'result.json';d=read(p);b=read(baseline/'runs'/scene/'result.json')
   assert d['quality']['complete'],(variant,scene)
   checks={k:d['input'][k]==b['input'][k] for k in ['image_paths','frame_indices','shape','checkpoint_sha256']}
   checks['quality_protocol']=d['quality_protocol']==b['quality_protocol']
   if manifest:checks['frozen_model_and_evaluation']=all(d['source_sha256'].get(k)==v for k,v in manifest['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
   assert all(checks.values()),(variant,scene,checks)
   audits.append(dict(variant=variant,scene=scene,checks=checks,result=str(p.relative_to(ROOT))))
   row=dict(variant=variant,scene=scene,**d['quality']['values']);rows.append(row);group.append(row)
   layers=read(p.parent/'layers.json')
   runtime.append(dict(variant=variant,scene=scene,frames=d['frames'],timing=d['timing'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),actual_unique_density_mean=statistics.mean(l['actual_unique_density'] for l in layers),max_native_cost_count_error=max(l.get('native_cost_count_error_mean',0) for l in layers),extra_fine_probe_qk_per_head=sum(l.get('fine_probe_qk_per_head',0) for l in layers),scope='Quality forward cold/JIT included; not a warmed speed comparison'))
  macros.append(dict(variant=variant,scenes=3,**{k:statistics.mean(r[k] for r in group) for k in KEYS}))
 speeds=[];dense=read(OUT/'controls/dense/timing_result.json');saf=read(OUT/'controls/saf3r/timing_result.json')
 assert dense['input']==saf['input'] and dense['hardware']==saf['hardware']
 for d in [dense,saf]:
  assert d['timing']['warmup']==1 and d['timing']['repeats']==3 and not d['timing']['capture']
 assert saf['saf3r']['topk']==4 and saf['saf3r']['lazy_dino_topk']
 for phase in PHASES:
  folder=ROOT/'outputs'/phase
  speed_path=folder/'speed_heads128/summary.json'
  for r in read(speed_path)['rows']:
   same_gpu=r['hardware']['uuid']==dense['hardware']['gpu_uuid']
   paired=phase==PHASES[1] and same_gpu
   speeds.append(dict(phase=phase,variant=r['variant'],seconds=r['median_seconds'],speedup_vs_dense=dense['timing']['median_seconds']/r['median_seconds'] if paired else None,speedup_unavailable=None if paired else 'Earlier phase1 batch; no contemporaneous Dense timing in that batch',same_gpu=same_gpu,peak_allocated_gib=max(s['peak_allocated_gib'] for s in r['samples']),graph_creations=[s['graph_creations'] for s in r['samples']],graph_evictions=[s['graph_evictions'] for s in r['samples']],samples=r['samples'],within_variant_repeat=r['within_variant_repeat']))
 base_time=read(ROOT/'outputs/operator_repair_20261105/baseline_speed/summary.json')['rows'][0]
 assert base_time['hardware']['uuid']==dense['hardware']['gpu_uuid']
 for name,seconds,peak in [('layout_fixed',base_time['median_seconds'],max(s['peak_allocated_gib'] for s in base_time['samples'])),('dense',dense['timing']['median_seconds'],max(s['peak_allocated_gib'] for s in dense['timing']['samples'])),('saf3r_top4_lazy_correspondence',saf['timing']['median_seconds'],max(s['peak_allocated_gib'] for s in saf['timing']['samples']))]:
  speeds.append(dict(phase='phase2_controls',variant=name,seconds=seconds,speedup_vs_dense=dense['timing']['median_seconds']/seconds,speedup_unavailable=None,peak_allocated_gib=peak))
 for row in speeds:
  row['relative_time_to_layout_fixed']=row['seconds']/base_time['median_seconds'];row['relative_time_to_saf3r']=row['seconds']/saf['timing']['median_seconds']
 for path in [OUT/'speed_gpu.json',OUT/'controls/gpu.json',ROOT/'outputs/operator_repair_20261105/baseline_speed_gpu.json']:
  monitor=read(path);assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected'],path
 for phase in PHASES:assert not read(ROOT/'outputs'/phase/'execution.json')['failures']
 save(OUT/'summary.json',dict(scope='Three ETH3D calibration scenes, not full ETH3D11; speed only7Scenes heads128, not seven-scene macro',macros=macros,per_scene=rows,paired_heads128_speed=speeds,input_protocol_audits=audits,quality_forward_runtime=runtime,adopted=False,baseline_full_eth3d=str(baseline.relative_to(ROOT)),baseline_repeat_drift=base_time['within_variant_repeat'],reason='No candidate satisfies AUC, reconstruction and no-speed-sacrifice jointly; production layout-fixed method retained'))
 with (OUT/'per_scene.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['variant','scene']+KEYS);writer.writeheader();writer.writerows(rows)
 lines=['# Operator-loss and partial-Key repair experiments','',
 'Research only. The production layout-fixed reuse method is retained. None of the eight frozen candidates has passed AUC, reconstruction and no-speed-sacrifice jointly. This is not an improved full-dataset result.','',
 '## Scope and controls','',
 'Quality: pipes, courtyard, playground ETH3D calibration scenes, same checkpoint, exact frozen RGB frame lists and preprocessing; pose plus recon_unposed. Equal-scene means below. Full11-scene layout-fixed reference remains unchanged. No7Scenes truth used to select parameters.','',
 'Reconstruction: ACC/Comp/CD in metres, CD=(ACC+Comp)/2 unsquared. F1 threshold0.25m, downsampling/TSDF voxel0.0390625m, truncation0.2m,1M mesh points, existing Umeyama/RANSAC alignment and masking. AUC/F1 in percentages in tables; raw JSON stores fractions.','',
 'All candidates preserve the original per-layer patch budgets:0.10 for layers0–5/18–23,0.22 for layers6–17 (mean cap0.16); this is a cap, not measured effective density. Actual unique densities and budget errors are separately stored in summary.json. No extra full Dense QK/oracle at inference.','',
 'SAF3R timing control: original repository eth3d-train-fltr-calib_cmpmse.json profile, Top-4 lazy first-correspondence head, not paper DINO Top-K. No new Dense/SAF3R quality evaluation.','',
 '## What was changed','',
 '1. normalized_moment scores each executable support using its weighted probability mass and projected output moment, including the softmax denominator and protected Keys. Equal support+weights gives equal loss, resolving Rep versus singleton History contradiction under estimated probabilities. It remains a local Dense-background approximation, not the true global all-cluster objective.','',
 '2. Add Alternate and Pair states. Alternate is a real cluster member farthest from its current representative in a deterministic head-aware Key sketch; Pair retains representative+alternate. Drop/Rep/History/Exact remain available. The shared global budget chooses among all clusters: no one-Key-per-cluster quota. This expands support coverage but cannot express every arbitrary subset.','',
 '3. Phase2 calibrated weights multiply population compensation by clipped estimated mass ratio [0.25,4]. The loss uses the same weight that the sparse kernel executes. This avoids scoring one operator and executing another, but depends on probability estimation accuracy.','',
 '4. Phase2 fixes CUDA Graph topology churn through fixed-capacity population bins. Zero warmed creations/evictions are verified, separately from prediction quality.','',
 '## Calibration macro quality','',
 '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 def quality_line(r,label):return '| '+label+' | '+' | '.join(f"{r[k]*100:.3f}" if k in ['auc03','auc30','f1'] else f"{r[k]:.6f}" for k in KEYS)+' |'
 lines += [quality_line(r,r['variant']) for r in macros]
 lines += ['', '## 7Scenes heads128 warmed full-forward speed','',
 'GPU6 RTX4090,1 warmup+3 repeats, median seconds. Includes reset, clustering, routing, history, reordering and graph work. Excludes file/model loading. Controls explicitly authorized for timing only. All speed-only quality fields are N/A; no geometric quality was requested for these runs. This single scene is not a7Scenes macro.','',
 '| Method/batch | Seconds | Dense speedup | Time/layout-fixed | Time/SAF3R | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
 for r in speeds:
  speed=f"{r['speedup_vs_dense']:.3f}x" if r['speedup_vs_dense'] is not None else 'N/A (earlier batch)'
  lines.append(f"| {r['variant']} / {r['phase']} | {r['seconds']:.6f} | {speed} | {r['relative_time_to_layout_fixed']:.4f} | {r['relative_time_to_saf3r']:.4f} | {r['peak_allocated_gib']:.3f} |")
 lines += ['', '## Per-scene quality','', '| Method / scene | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 lines += [quality_line(r,r['variant']+' / '+r['scene']) for r in rows]
 lines += ['', '## Interpretation','',
 'Fixing the loss formula alone did not stabilize AUC. Current probabilities are coarse-grid estimates plus historical one-Probe evidence, shared across heads/query groups; a4D Value/output sketch and separable cluster objective leave head/query-specific probability error and cross-cluster softmax interactions. Better loss consistency does not establish better true output error. The oracle used true current attention and a broader mask search, which these deployable candidates do not have.','',
 'Repeatability limitation: warmed repeated forwards are not bitwise identical even for the unchanged layout-fixed baseline (pose-encoding relative L2 about0.0053); normalized-loss candidates are around0.02–0.027. The existing reductions/clustering plus discrete state decisions amplify numerical differences. Small single-run AUC improvements are exploratory, not a statistically established gain. Full repeat drift is retained in summary.json and the timing artifacts. Fixed population-bin reductions can also alter rounding versus the phase1 padded layout; states_only_fast is separately evaluated, not declared output-identical.','',
 'Partial states sometimes improve pose while damaging unposed reconstruction; this is not a successful repair. Mass calibration and additional states likewise must pass measured gates before adoption. No test-set parameter selection or hidden density increase was used. Failed candidates remain documented rather than replacing the production method.','',
 'Diagnostics: unit tests cover identical physical-support losses, normalized predicted error, binned statistics and emitted execution weights. Earlier Triton compile failure is retained as a failed diagnostic with all six quality fields N/A. No candidate was promoted to full ETH3D/7Scenes evaluation after failing calibration gates.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 # Every diagnostic (including the retained initial compile failure) has explicit six-field unavailability.
 for directory,name,reason in [(ROOT/'outputs/operator_repair_20261105/validation','failed_compile','Initial Triton compile failure; no completed forward or pose/recon_unposed'),(OUT,'validation','CUDA algebra/kernel diagnostics only; not pose or recon_unposed evaluation')]:
  q=quality_record({},reason);save(directory/f'{name}_quality.json',q);(directory/f'{name}_quality.md').write_text(quality_markdown(q))
 print(json.dumps(dict(macros=macros,speeds=speeds),indent=2))
if __name__=='__main__':main()
