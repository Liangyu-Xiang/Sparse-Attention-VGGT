"""Annotate authorized matched seven-scene full-forward timings; no quality edits."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_markdown

def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--report-dir',type=Path,required=True);a=p.parse_args()
    plan=read(a.plan);assert plan['user_approved_control_timing'];name=next(iter(plan['methods']));rows=[]
    for scene in plan['datasets']['7scenes']:
        directory=a.runs/'7scenes'/scene
        paths={m:directory/m/('result.json' if m==name else 'timing_result.json') for m in ['dense','saf3r',name]}
        runs={m:read(p) for m,p in paths.items()};dense=runs['dense'];saf=runs['saf3r']
        assert saf['saf3r']['topk']==4 and saf['saf3r']['lazy_dino_topk']
        for method,d in runs.items():
            assert d['input']==dense['input'] and d['hardware']==dense['hardware'],(scene,method,'pair mismatch')
            for k in ['warmup','repeats','capture']:assert d['timing'][k]==dense['timing'][k],(scene,method,k)
            assert d['timing']['warmup']==1 and d['timing']['repeats']==3 and not d['timing']['capture']
            monitor=read(a.runs/f'7scenes_{scene}_{method}.monitor.json')
            assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
            assert all(d['source_sha256'].get(k)==v for k,v in dense['source_sha256'].items() if k.startswith(('saf3r/','evaluation/','scripts/'))),(scene,method,'source mismatch')
            seconds=d['timing']['median_seconds'];speed=dense['timing']['median_seconds']/seconds
            before=dict(d['quality']['values'])
            d['quality'].update(speedup_vs_vggt=speed,speedup_unavailable=None)
            d['paired_timing']=dict(dense_result=str(paths['dense'].resolve()),saf3r_result=str(paths['saf3r'].resolve()),same_inputs_hardware_protocol=True,quality_values_unchanged=True)
            assert d['quality']['values']==before
            save(paths[method],d);save(paths[method].parent/'quality.json',d['quality']);(paths[method].parent/'quality.md').write_text(quality_markdown(d['quality']))
            rows.append(dict(scene=scene,method=method,seconds=seconds,speedup_vs_dense=speed,ratio_to_saf3r=seconds/saf['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),hardware=d['hardware'],quality_complete=d['quality']['complete']))
    macro={m:dict(mean_scene_median_seconds=statistics.mean(r['seconds'] for r in rows if r['method']==m),max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows if r['method']==m)) for m in ['dense','saf3r',name]}
    for m,d in macro.items():d['speedup_vs_dense']=macro['dense']['mean_scene_median_seconds']/d['mean_scene_median_seconds']
    ratio=macro[name]['mean_scene_median_seconds']/macro['saf3r']['mean_scene_median_seconds']
    gates=dict(no_slowdown_macro=ratio<=1.,within_original_10percent_macro=ratio<=1.1,within_10percent_every_scene=all(r['ratio_to_saf3r']<=1.1 for r in rows if r['method']==name))
    result=dict(dataset='7scenes',scenes=7,method=name,macro=macro,method_time_over_saf3r=ratio,gates=gates,per_scene=rows,aggregation='Ratio of equal-scene means of warmed per-scene medians; per-scene ratios also reported',scope='1 warmup +3 complete forwards; includes reset/clustering/routing/history/reordering and SAF3R Top-K recomputation; validation synchronizations retained',saf3r=dict(topk=4,source='lazy first-correspondence head; not paper DINO feature Top-K',profile=saf['saf3r']['profile'],profile_sha256=saf['source_sha256']['configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']))
    a.report_dir.mkdir(parents=True,exist_ok=True);save(a.report_dir/'paired_speed.json',result)
    text=['# Paired 7Scenes speed','',result['scope']+'.','',result['aggregation']+'.','',
          '| Method | Mean scene median (s) | Speedup vs original Dense | Maximum allocated GiB |','|---|---:|---:|---:|']
    for m,d in macro.items():text.append(f"| {m} | {d['mean_scene_median_seconds']:.6f} | {d['speedup_vs_dense']:.3f}x | {d['max_peak_allocated_gib']:.6f} |")
    text+=['',f'Method / SAF3R time: {ratio:.6f}. Gates: {gates}.','',
           'SAF3R uses repository profile eth3d-train-fltr-calib_cmpmse.json and lazy first-correspondence-head Top-4 per frame (512 selected indices at128frames), not paper DINO Top-K. Control runs have all six quality metrics N/A because only timing was requested; no control quality scores were invented. ETH3D speed not evaluated.','',
           '| Scene | Method | Seconds | Dense speedup | Method / SAF3R time |','|---|---|---:|---:|---:|']
    for r in rows:text.append(f"| {r['scene']} | {r['method']} | {r['seconds']:.6f} | {r['speedup_vs_dense']:.3f}x | {r['ratio_to_saf3r']:.6f} |")
    (a.report_dir/'speed.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(macro=macro,gates=gates),indent=2))

if __name__=='__main__':main()
