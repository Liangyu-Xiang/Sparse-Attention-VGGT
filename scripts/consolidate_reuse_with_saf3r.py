"""Rebuild the combined local comparison, retaining distinct paired timing cohorts."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/reuse_variants_consolidated_20261029';NEW=ROOT/'outputs/saf3r_matched_reuse_comparison_20261029'
def main():
 data=json.loads((OUT/'summary.json').read_text());fresh=json.loads((NEW/'summary.json').read_text());assert fresh['terminal']
 controls=[];additional=[]
 for row in fresh['per_run']:
  source=Path(row['source']);r=json.loads((source/'result.json').read_text());assert r['quality']['complete'];v=r['quality']['values'];assert all(math.isfinite(x) for x in v.values());assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-9
  item=dict(scene=row['scene'],method=row['method'],source=str(source),result=r)
  (controls if row['method']=='dense' else additional).append(item)
 assert len(controls)==7 and len(additional)==14
 for macro in fresh['macro']:
  if macro['method']=='dense':
   assert macro['values']==data['macro']['dense']['values'];data['fresh_saf3r_paired_dense']=macro
  else:data['macro'][macro['method']]=macro
 data['per_run']=[r for r in data['per_run'] if r['method'] not in ('saf3r_dino64','saf3r_lazy4')]+additional
 data.update(unique_combinations=63,successful=56,failed_oom=7,additional_dense_timing_control_runs=7,dense_timing_controls=controls,saf3r_comparison=fresh)
 order=['dense','saf3r_dino64','saf3r_lazy4','original','shift8','max_error','single_qk','grid4','grid8']
 names={'dense':'VGGT Dense','saf3r_dino64':'SAF3R DINO per-frame Top-64','saf3r_lazy4':'SAF3R lazy first-correspondence Top-4','original':'Reuse original 2x2','shift8':'Query shift8 (OOM)','max_error':'Maximize state error','single_qk':'Single QK saved logits','grid4':'Canonical grid 4x4','grid8':'Canonical grid 8x8'}
 fields=('auc03','auc30','acc','comp','cd','f1')
 text=['# Reuse variants and SAF3R: consolidated local measurements','', '7Scenes all seven scenes, 128 uniform frames. Same exact input lists/checkpoint/preprocessing/hardware/pose and recon_unposed settings; pairing audit for new cohort passed for all21 runs. All jobs terminal: 56 successful +7 OOM out of63 primary method-scene combinations, plus7 fresh paired Dense timing controls. Diagnostics and superseded noisy timing runs excluded.','', 'Quality is equal-scene macro. AUC/F1 percentages; distances metres. F1 threshold0.05m, downsampling0.0078125m; CD=(ACC+Comp)/2 unsquared distances. Runtime includes all complete model forward costs, with1 warmup +3 measurements. Peak GiB is equal-scene average of per-run median measured peak allocated GiB. Speedup is ratio of equal-scene mean median full-forward seconds for paired Dense/method; below1x means slower.','', 'Timing cohorts: reuse methods retain their original matched Dense (12.282440s). SAF3R methods use freshly paired Dense (12.294450s); no historical timing rebasing. Fresh Dense quality equals original Dense quality exactly. Raw samples and hardware/protocol checks are retained.','', 'SAF3R DINO64 uses final DINO feature cosine similarity, 64keys per frame, max_topk8192 (128*64; no cap beyond requested per-frame support), no guaranteed dense first frame. Its declared fixed K is a paper-mechanism reproduction setting; unpublished per-scene paper K is not claimed recovered. Lazy4 uses the first correspondence head. Both use configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json. These are local measured rows, with no published-reference rows or H100 timing baseline.','', '| Method | Scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean median s | Mean peak GiB | Speedup vs paired Dense |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in order:
  r=data['macro'][method];v=r['values'];cells=['N/A']*6 if v is None else [f'{v[k]*(100 if k in ("auc03","auc30","f1") else 1):.6f}' for k in fields]
  perf=['N/A']*3 if v is None else [f'{r["mean_scene_median_seconds"]:.6f}',f'{r["mean_scene_peak_allocated_gib"]:.6f}',f'{r["speedup_ratio_of_mean_scene_medians"]:.6f}x']
  text.append('| '+' | '.join([names[method],str(r['quality_scenes_complete'])+'/7']+cells+perf)+' |')
 text+=['','shift8: all7 scenes OOM in all-group historical support merging/sorting/coalescing. All six metrics and paired speedup remain N/A with per-run reasons. It is not a completed quality evaluation.','', 'Per-scene and raw source reports:','', '- reuse_single_change_7scenes128_20261029/report.md','- reuse_canonical_grid_7scenes128_20261029/report.md','- saf3r_matched_reuse_comparison_20261029/report.md','- saf3r_matched_reuse_comparison_20261029/pairing_audit.json']
 (OUT/'summary.json').write_text(json.dumps(data,indent=2));(OUT/'report.md').write_text('\n'.join(text)+'\n')
 print('Consolidated',len(data['per_run']),'primary rows; audit',len(json.loads((NEW/'pairing_audit.json').read_text())))
if __name__=='__main__':main()
