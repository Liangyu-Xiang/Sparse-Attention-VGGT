"""Seven-scene paired128-frame evaluation; no tuning against test truth."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record,quality_cells

SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
METHODS=('dense','saf3r_dino64','original','cosine_distortion','rep_combined')


def report(root):
    results={};forwards={};rows=[]
    lines=['# Rep-baseline complete combined revision: matched 7Scenes 128-frame evaluation','',
        'Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.',
        'AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.',
        'Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.',
        'Only the complete combined revision is scheduled, per user correction. Superseded staged runs are excluded. Original and previous cosine quality results are reused from the earlier completed cohort; their timings are historical. Fresh Dense/SAF3R controls already completed before the user requested avoiding redundant controls; these are retained. See reference_provenance.json.',
        'Combined changes: Drop=pi and Rep-relative gains; all executed patch edges provide history; fused compact raw-logit residual history calibrated within current canonical grids; estimated mass transfer retained as previously requested. See docs/experiments/reuse_rep_revision.md. Second-QK mass summaries are retained and counted; the combined revision adds no extra QK pass and stores no full logit tensor.',
        'Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.',
        '', '| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Speedup vs Dense |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        for method in METHODS:
            out=root/scene/method;result=out/'result.json';forward=out/'forward.json'
            if forward.exists():forwards[scene,method]=json.loads(forward.read_text())
            if result.exists():
                data=json.loads(result.read_text());results[scene,method]=data;q=data['quality']
            else:
                quality=out/'quality.json'
                q=json.loads(quality.read_text()) if quality.exists() else quality_record({},'Queued; evaluation not complete')
            f=forwards.get((scene,method))
            seconds='N/A' if f is None else f"{f['timing']['median_seconds']:.3f}"
            peak='N/A' if f is None else f"{np.median([s['peak_allocated_gib'] for s in f['timing']['samples']]):.3f}"
            speed=q.get('speedup_vs_vggt');speed='N/A' if speed is None else f'{speed:.3f}x'
            lines.append('| '+' | '.join([scene,method]+quality_cells(q)+[seconds,peak,speed])+' |')
            rows.append(dict(scene=scene,method=method,quality=q,forward_complete=f is not None,
                             quality_complete=q['complete']))
    macro=[]
    for method in METHODS:
        complete=[results[s,method] for s in SCENES if (s,method) in results and results[s,method]['quality']['complete']]
        item=dict(method=method,quality_scenes_complete=len(complete),required_scenes=7,values=None)
        if len(complete)==7:
            item['values']={k:float(np.mean([r['quality']['values'][k] for r in complete])) for k in complete[0]['quality']['values']}
            item['mean_scene_median_seconds']=float(np.mean([r['timing']['median_seconds'] for r in complete]))
            item['mean_scene_peak_allocated_gib']=float(np.mean([np.median([t['peak_allocated_gib'] for t in r['timing']['samples']]) for r in complete]))
        macro.append(item)
    dense=next(x for x in macro if x['method']=='dense')
    lines+=['','## Equal-scene macro averages (only when all seven quality evaluations finish)','',
        '| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in macro:
        q=quality_record({},'All seven quality evaluations required for macro')
        if item['values'] is not None:q['values']=item['values']
        speed=None
        if item['values'] is not None and dense['values'] is not None:
            speed=None if item['method'] in ('original','cosine_distortion') else dense['mean_scene_median_seconds']/item['mean_scene_median_seconds']
            if speed is None:
                item['speedup_unavailable']='Reused historical control timing; not a contemporaneous paired speed comparison'
                lines.append('| '+' | '.join([item['method'],f"{item['quality_scenes_complete']}/7"]+quality_cells(q)+['N/A (historical)'])+' |')
                continue
            item['speedup_ratio_of_mean_scene_medians']=speed
            item['delta_auc30_pp']=100*(item['values']['auc30']-dense['values']['auc30'])
            item['delta_f1_pp']=100*(item['values']['f1']-dense['values']['f1'])
            item['auc30_f1_gates_pass']=item['delta_auc30_pp']>=-.2 and item['delta_f1_pp']>=-.5
            item['twofold_speed_gate_pass']=speed>=2
            item['memory_gate_pass']=all(np.median([r['peak_allocated_gib'] for r in results[s,item['method']]['timing']['samples']])
                <=np.median([r['peak_allocated_gib'] for r in results[s,'dense']['timing']['samples']]) for s in SCENES)
        lines.append('| '+' | '.join([item['method'],f"{item['quality_scenes_complete']}/7"]+quality_cells(q)+['N/A' if speed is None else f'{speed:.3f}x'])+' |')
    lines+=['','¹ Ratio of equal-scene mean warmed median full-forward times. Below1x is slower.',
        'Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.',
        'Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.',
        'Per-method input/config/source snapshots and raw metrics are stored under scene/method/.']
    comparisons=[]
    reference=next(x for x in macro if x['method']=='original')
    lines+=['','## Quality differences against reused original results','',
        'AUC/F1 differences in percentage points; distance differences in metres. Runtime ratio = original / variant; >1 means faster.',
        '', '| Variant | dAUC@3 pp | dAUC@30 pp | dACC m | dComp m | dCD m | dF1 pp | Original/variant time |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in macro:
        if item['method'] in ('dense','original'):continue
        entry=dict(method=item['method'],reference='original',delta_values=None,runtime_ratio=None)
        if item['values'] is not None and reference['values'] is not None:
            delta={k:item['values'][k]-reference['values'][k] for k in item['values']}
            ratio=None # Original quality is reused; its timing is historical
            entry.update(delta_values=delta,runtime_ratio=ratio)
            values=[f"{delta[k]*(100 if k in ('auc03','auc30','f1') else 1):+.6f}" for k in ('auc03','auc30','acc','comp','cd','f1')]
            lines.append('| '+' | '.join([item['method']]+values+['N/A (historical reference)'])+' |')
        comparisons.append(entry)
    save(root/'summary.json',dict(per_run=rows,macro=macro,comparisons_to_original=comparisons,quality_scope='seven scenes128frames, pose+recon_unposed',
        published_reference=False,unavailable_depth='Not evaluated in this pose/reconstruction experiment'))
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    write_quality(root,quality_record({},'Dataset evaluation in progress; consult per-scene results and report.md'))
    jobs=[]
    for gpu,scene in enumerate(SCENES):
        for method in ('rep_combined',):
            jobs.append(dict(gpu=gpu,scene=scene,method=method,output=str(root/scene/method)))
    frozen={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py',
        'scripts/evaluate_reuse_cosine.py','configs/evaluation/reuse_residual_exploration.json',
        'configs/evaluation/reuse_ablation_original.json','configs/evaluation/reuse_ablation_shift8.json',
        'configs/evaluation/reuse_ablation_max_error.json','configs/evaluation/reuse_ablation_single_qk.json',
        'saf3r/patch/reuse_cosine_distortion.py','saf3r/triton/reuse_cosine_distortion.py',
        'scripts/run_reuse_cosine_eval.py','scripts/evaluate_saf3r_matched.py',
        'configs/evaluation/reuse_cosine_distortion.json','docs/experiments/reuse_cosine_distortion.md',
        'docs/experiments/reuse_cosine_distortion_reference.md',
        'docs/experiments/reuse_rep_revision.md','docs/experiments/reuse_rep_revision_reference.md',
        'scripts/run_reuse_rep_revision.py','tests/test_reuse_rep_revision.py','scripts/validate_reuse_rep_revision.py',
        'configs/evaluation/reuse_rep_combined.json',
        'configs/evaluation/reuse_rep_a.json','configs/evaluation/reuse_rep_b.json',
        'configs/evaluation/reuse_rep_c.json','configs/evaluation/reuse_rep_d.json')}
    save(root/'manifest.json',dict(jobs=jobs,source_sha256=frozen,frames=128,warmup=1,repeats=3,
        tasks=['pose','recon_unposed'],seed=0,retention_cap=.2,no_truth_tuning=True,
        baseline='contemporaneous per-scene Dense on same physical GPU',
        old='Prior original and cosine quality results reused via relative symlinks; no reruns. Fresh already-completed Dense timing pairs the four modified runs; historical original timing is not a contemporaneous comparison'))
    live={};done={};handles=[]
    while len(done)<len(jobs):
        for i,job in enumerate(jobs):
            if i in live or i in done:continue
            out=Path(job['output'])
            if (out/'status.json').exists():
                state=json.loads((out/'status.json').read_text())['stage']
                if state in ('complete','failed','quality_incomplete'):done[i]=state
                continue
            earlier=[j for j in range(i) if jobs[j]['gpu']==job['gpu'] and j not in done and j not in live]
            if earlier:continue
            peers=[j for j in live if jobs[j]['gpu']==job['gpu']]
            if len(peers)>=2:continue
            def gpu_busy(j):
                path=Path(jobs[j]['output'])/'status.json'
                return not path.exists() or json.loads(path.read_text())['stage'] not in ('quality','complete','quality_incomplete','failed')
            if any(gpu_busy(j) for j in peers):continue
            baseline=root/job['scene']/'dense/forward.json'
            if job['method']!='dense' and not baseline.exists():
                if any(jobs[j]['scene']==job['scene'] and jobs[j]['method']=='dense' and done.get(j)=='failed' for j in done):
                    out.mkdir(parents=True,exist_ok=True)
                    write_quality(out,quality_record({},'Paired Dense forward failed; no comparable run started'))
                    save(out/'status.json',dict(stage='failed',error='Paired Dense unavailable'));done[i]='failed'
                continue
            log=(root/'logs'/f"{job['scene']}_{job['method']}.log").open('w');handles.append(log)
            command=[sys.executable,'scripts/evaluate_reuse_cosine.py','--scene',job['scene'],
                '--method',job['method'],'--frames','128','--output-dir',job['output']]
            if job['method']=='saf3r_dino64':command[1]='scripts/evaluate_saf3r_matched.py'
            if job['method']!='dense':command+=['--baseline',str(baseline)]
            env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':str(job['gpu']),
                 'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
            live[i]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for i,p in list(live.items()):
            rc=p.poll()
            if rc is not None:
                path=Path(jobs[i]['output'])/'status.json'
                done[i]=json.loads(path.read_text())['stage'] if path.exists() else f'failed_exit_{rc}'
                del live[i]
        save(root/'queue_status.json',dict(jobs=[dict(job,state=done.get(i,'running' if i in live else 'queued'),
            pid=live[i].pid if i in live else None) for i,job in enumerate(jobs)],all_terminal=len(done)==len(jobs)))
        report(root)
        if len(done)<len(jobs):time.sleep(15)
    for h in handles:h.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--report-only',action='store_true');a=p.parse_args()
    if a.report_only:report(a.output_dir)
    else:run(a)
