"""Three128-frame scene pairs: explicitly requested Dense/SAF3R, matched group128."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dense_saf3r_query128_20261104'
SCENES=('heads','chess','stairs')
METHODS=('dense','saf3r','shared_value_q128')
FIELDS=('auc03','auc30','acc','comp','cd','f1')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def report():
    per_run=[];macro={};failures=[]
    lines=['# Matched Dense / SAF3R lazy Top4 / shared Value Query-group128','',
        'Local RTX4090, three128-frame sequences only:heads/chess/stairs. NOT the complete7Scenes macro.',
        'Same exact checkpoint/input frames/preprocessing, BF16, TF32 off. One warmup + three full forwards.',
        'Each scene: all methods serially on the same exclusive physical GPU; all model, initialization, clustering, routing, history and output costs included.',
        'SAF3R is repository lazy first-correspondence-head Top4, NOT paper DINO feature Top-K.',
        'Top-K flags reset inside EVERY timed SAF3R forward, including warmup. No warmup Top-K reuse; reset/selection overhead included.',
        'Earlier cached-Top-K attempts are preserved under excluded_attempts and excluded from every final row. Dense timings remain valid; only the evaluator SAF3R conditional was corrected.',
        'Profile:configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json.',
        'Shared Value:20% maximum Key retention, nominal20% Cluster count, query_group_size128, history_query_interval256.',
        'Pose plus recon_unposed; AUC/F1 percentages, reconstruction distances metres. CD=(ACC+Comp)/2, unsquared.',
        'F1 threshold0.05m, reconstruction downsample0.0078125m. No truth-based parameter tuning.','',
        '| Scene | Method | Forward s | vs Dense | Peak allocated GiB | Peak reserved GiB | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def cells(values):return ['N/A' if values[k] is None else f"{values[k]*(100 if k in ('auc03','auc30','f1') else 1):.5f}" for k in FIELDS]
    items={m:[] for m in METHODS}
    for scene in SCENES:
        group=[]
        for name in METHODS:
            path=OUT/scene/name/'result.json'
            if not path.exists():failures.append(dict(scene=scene,method=name,reason='Missing result'));continue
            d=json.loads(path.read_text());d['comparison_name']=name
            monitor=json.loads((OUT/'monitors'/f'{scene}_{name}.json').read_text())
            assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected'] and not monitor['foreign_gpu_pids']
            d['exclusive_monitor']=monitor
            assert len(d['timing']['samples'])==3 and d['frames']==128
            if not d['quality']['complete']:failures.append(dict(scene=scene,method=name,reason=d['quality_errors']))
            else:
                v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
            per_run.append(d);group.append(d);items[name].append(d)
            t=d['timing'];q=d['quality']
            lines.append('| '+' | '.join([scene,name,f"{t['median_seconds']:.5f}",f"{q['speedup_vs_vggt']:.4f}x",
                f"{max(p['peak_allocated_gib'] for p in t['samples']):.4f}",f"{max(p['peak_reserved_gib'] for p in t['samples']):.4f}"]+cells(q['values']))+' |')
        if len(group)==3:
            baseline=group[0]
            for d in group[1:]:
                for k in ('input','hardware','quality_protocol'):assert d[k]==baseline[k],(scene,k)
                # Only the evaluator's SAF3R reset conditional changed; Dense
                # source/model/preprocessing and its measurements remain intact.
                for k in baseline['source_sha256']:
                    if k.startswith('saf3r/') or k.startswith('configs/'):
                        assert d['source_sha256'][k]==baseline['source_sha256'][k],(scene,k)
                if d['comparison_name']=='saf3r':assert d['timing']['saf3r_topk_reset_each_forward']
                assert d['exclusive_monitor']['gpu_uuid']==baseline['exclusive_monitor']['gpu_uuid']
                assert d['quality']['speedup_vs_vggt']==baseline['timing']['median_seconds']/d['timing']['median_seconds']
    if not failures:
        for name,runs in items.items():
            macro[name]=dict(values={k:sum(d['quality']['values'][k] for d in runs)/len(runs) for k in FIELDS},
                mean_scene_median_seconds=sum(d['timing']['median_seconds'] for d in runs)/len(runs),
                peak_allocated_gib=max(p['peak_allocated_gib'] for d in runs for p in d['timing']['samples']),
                peak_reserved_gib=max(p['peak_reserved_gib'] for d in runs for p in d['timing']['samples']))
        for name in METHODS:macro[name]['speedup_vs_dense']=macro['dense']['mean_scene_median_seconds']/macro[name]['mean_scene_median_seconds']
        lines+=['','## Three-scene equal averages (not full7Scenes)','',
            '| Method | Mean scene median s | vs Dense | Peak allocated GiB | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name in METHODS:
            d=macro[name]
            lines.append('| '+' | '.join([name,f"{d['mean_scene_median_seconds']:.5f}",f"{d['speedup_vs_dense']:.4f}x",f"{d['peak_allocated_gib']:.4f}"]+cells(d['values']))+' |')
        lines+=['','## Quality difference versus local matched Dense','',
            '| Method | Δ AUC@3 pp | Δ AUC@30 pp | Δ ACC m | Δ Comp m | Δ CD m | Δ F1 pp |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name in METHODS[1:]:
            delta={k:macro[name]['values'][k]-macro['dense']['values'][k] for k in FIELDS}
            lines.append('| '+name+' | '+' | '.join(f"{delta[k]*(100 if k in ('auc03','auc30','f1') else 1):+.5f}" for k in FIELDS)+' |')
    if failures:lines+=['','Incomplete: '+json.dumps(failures)]
    lines+=['','All rows are local measurements. Prior batches, published baselines and their timings are not mixed into these speedups.',
        'Raw per-run quality.json/quality.md, raw task metrics, timing samples, inputs/config, source tar/hashes, layers and exclusive GPU monitors are preserved.','']
    summary=dict(per_run=per_run,subset_average=macro,complete=len(per_run)==9 and not failures,incomplete=failures,
        scope='Three7Scenes sequences,128 frames each; not full dataset macro')
    save(OUT/'summary.json',summary);(OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    if a.report_only:report();return
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'manifest.json',dict(scenes=SCENES,methods=METHODS,frames=128,gpus=[3,4,5],warmup=1,repeats=3,
        authorization='User explicitly requested original Dense and SAF3R128-frame speed; group128 remeasured for contemporaneous paired speedup',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        config='configs/evaluation/reuse_shared_value_loss_q128.json'))
    def worker(scene,gpu):
        order=('dense','shared_value_q128','saf3r') if scene=='chess' else METHODS
        for name in order:
            output=OUT/scene/name
            if (output/'result.json').exists():continue
            if output.exists():raise RuntimeError('Incomplete existing directory preserved:'+str(output))
            method='zdrop_history_states' if name=='shared_value_q128' else name
            command=[sys.executable,'scripts/evaluate_zdrop_history_states.py','--dataset','7scenes','--scene',scene,
                '--frames','128','--method',method,'--config','configs/evaluation/reuse_shared_value_loss_q128.json',
                '--output-dir',str(output),'--warmup','1','--repeats','3']
            if name!='dense':command+=['--baseline',str(OUT/scene/'dense/forward.json')]
            call=[sys.executable,'scripts/run_gpu_lease.py','--strict-contexts','--gpu',str(gpu),
                '--monitor',str(OUT/'monitors'/f'{scene}_{name}.json'),'--log',str(OUT/'logs'/f'{scene}_{name}.log'),'--',*command]
            print('START',scene,name,gpu,flush=True);proc=subprocess.run(call,cwd=ROOT)
            print('END',scene,name,proc.returncode,flush=True)
            if proc.returncode:raise RuntimeError(f'{scene}/{name} failed')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(worker,s,g) for s,g in zip(SCENES,(3,4,5))]
        status=[]
        for job in jobs:
            try:job.result();status.append(dict(complete=True))
            except Exception as exc:status.append(dict(complete=False,error=str(exc)))
    save(OUT/'status.json',status)
    summary=report();print('COMPLETE',summary['complete'],OUT/'report.md',flush=True)

if __name__=='__main__':main()
