"""Aggregate kernel-only runs without inventing any geometric quality scores."""
import csv,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/direct_inline_probability_summary_20261104';FINAL=ROOT/'outputs/direct_inline_branch_all_windows_v2_20261104/benchmark/result.json'
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def csvwrite(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 from scripts.quality_metrics import quality_record,quality_markdown
 OUT.mkdir(parents=True,exist_ok=True);d=json.loads(FINAL.read_text());assert d['complete'] and len(d['windows'])==1584 and len(d['layers'])==24
 validation=json.loads((ROOT/'outputs/direct_inline_branch_validation_20261104/numerical/result.json').read_text());assert validation['complete']
 for name,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 monitors=[];runs=[]
 for folder in sorted((ROOT/'outputs').glob('direct_inline_*_20261104')):
  if folder==OUT:continue
  mon=folder/'monitor.json'
  if mon.exists():
   m=json.loads(mon.read_text());assert not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']<=1,mon
   monitors.append(dict(run=folder.name,returncode=m['returncode'],gpu=m['gpu'],exclusive=True))
  for p in folder.glob('*/result.json'):
   r=json.loads(p.read_text());q=r.get('quality',{});reason=r.get('quality_reason') or '; '.join(dict.fromkeys(q.get('unavailable',{}).values())) or 'Kernel diagnostic, no pose/reconstruction evaluation'
   if 'values' in q:record=q
   else:record=quality_record({},reason)
   if not (p.parent/'quality.md').exists():(p.parent/'quality.md').write_text(quality_markdown(record))
   for method in r.get('totals',{'diagnostic':None}):runs.append(dict(run=folder.name,method=method,diagnostic_complete=r.get('complete',False),**{k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason=reason,dense_speedup=None))
 base=d['totals']['av_only_seconds'];totals=[]
 for method,seconds in d['totals'].items():
  spacing=32 if method in ['probe_probability_seconds','branch_inline_1_seconds'] else 256 if method=='branch_inline_8_seconds' else 512 if method=='branch_inline_16_seconds' else None
  totals.append(dict(method=method,query_spacing=spacing,seconds=seconds,overhead_percent=100*(seconds/base-1),extra_seconds=seconds-base))
 errors={key:max(row[key] for row in d['windows']) for key in d['windows'][0] if key.endswith(('max_abs','relative_l2'))}
 source_names=list(d['source_sha256'])+['scripts/validate_sampled_probability.py','scripts/report_sampled_probability.py']
 hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names};save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in source_names:archive.add(ROOT/name,arcname=name,recursive=False)
 final_mon=json.loads((FINAL.parent.parent/'monitor.json').read_text());assert final_mon['returncode']==0
 quality=quality_record({},'Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation')
 summary=dict(complete=True,base_commit='d10fd3e6',sampling=d['sampling'],windows=len(d['windows']),layers=24,totals=totals,errors=errors,numerical_cases=len(validation['checks']),cuda_graph_cases=len(validation['cuda_graphs']),monitors=monitors,quality=quality,dense_speedup=None,dense_reason=d['dense_reason'],kernel_source_unchanged=True,history_sampling_changed=False,full_forward_seconds=None)
 save(OUT/'summary.json',summary);csvwrite(OUT/'timings.csv',totals);csvwrite(OUT/'per_window.csv',d['windows']);csvwrite(OUT/'per_layer.csv',d['layers']);csvwrite(OUT/'six_metrics_unavailable.csv',runs);(OUT/'quality.md').write_text(quality_markdown(quality))
 table=['| Output | Query interval | 24-layer total s | Extra cost |','|---|---:|---:|---:|']+[f"| {r['method']} | {r['query_spacing'] or '-'} | {r['seconds']:.9f} | {r['overhead_percent']:+.3f}% |" for r in totals]
 text=['# Direct Sparse selected-Key probability export','',
 'Final matched actual-data kernel benchmark:all1584 windows across24 global layers,heads128,all16Heads,same Q/K/V/support/bias,three samples per variant/window. Sums of window medians,NOT a complete forward or single-layer time. Alternate outputs never feed the original current Direct-probability trajectory. GPU7 exclusive,no Dense/SAF3R model reruns.','',*table,'',
 'The new branch path exports fixed-interval Query probabilities on every selected Key. Normalize each Head using its final compensated sparse LSE,then average Heads. Specials are retained in the denominator. AV covers all Queries. Final-probability reduction and all export launches are included in these timings.','',
 'Main QK is computed once per Query Group. Sampled Groups export FP16 online exponents and FP32 Tile reference maxima before AV;ordinary Groups use the usual AV body in the same launch. The small finalizer rescales with final LSE and averages Heads. There is no auxiliary probability QK,no Grid routing,no full QxK mask,and no FP32 raw-logit bank. Dense four-Probe output remains materially more expensive;the near-free claim only applies to the measured sparse Query sampling.','',
 'Scratch for one64-Group16-Head window:every256Queries6929152bytes(~6.61MiB),every512Queries3464576bytes(~3.30MiB). No current history sampling/default changed.','',
 '## Numerical validation','',f"Numerical cases={len(validation['checks'])},fresh CUDA Graph cases={len(validation['cuda_graphs'])}; FP16/BF16,pure/compensated,zero weights,bias60,empty patch support with Specials,tail/duplicate Probes,one/four Probes,strides1/2/4/8/16 and non-contiguous Head windows. Real-data maximum errors: {json.dumps(errors)}.",'',
 'FP16 export storage is an explicit auxiliary probability quantization;do not claim bitwise probability equality. AV numerical equality in the actual all-window measurements is recorded as zero relative L2 for every window;this is not a bytewise dump comparison. Default debiased_log=False returns actual compensated sparse P;debiased_log=True is a separate log-score export with different numeric errors and is not the timed near-free output.','',
 '## Geometry and end-to-end limits','',quality_markdown(quality),'',
 'No new full-model forward median is measured in this kernel-only extension. The preceding model15.641640s belongs to direct_probability_quality_20261104 and is not a new timing or geometry result. New Query sampling has not been fed into history,so existing six-metric quality cannot validate such a modeling change. Dense end-to-end speedup N/A,not an operator ratio.','',
 '## Earlier attempts','',
 'Original Tile-normalized inline,Gather exponent reuse,masked stores and splitting AV into two launches are retained with their raw benchmarks. The first attempted one-Probe pointer store failed compilation and was fixed by explicit pointer broadcasting. One branch benchmark failed because instrumentation wrappers were imported as a device function;import order was corrected without changing kernel arithmetic. Failed runs preserve all six scores N/A and their logs. Synthetic preflight and first-window24-layer timings are separate from final full1584-window evidence.','',
 f"Compiler metadata: {json.dumps(d['compiler_resources'])}. This is not measured occupancy. Verified exclusive monitors: {len(monitors)};kernel source hashes unchanged. All raw run six-metric statuses are in six_metrics_unavailable.csv;no GT parameter tuning or quality-gate changes."]
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
