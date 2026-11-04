"""True 128/256-query support groups; unchanged shared-Value method otherwise.

Only changed methods are run. Existing Dense is not rerun (AGENTS.md).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SCENES={'7scenes':('heads','chess','stairs')}
FIELDS=('auc03','auc30','acc','comp','cd','f1')

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def report(root):
    runs=[];pairs=[];aggregates={};failures=[]
    lines=['# Actual Query-group size 128 vs 256','',
        'Source: codex/shared-value-cluster-scaling-20261104, d7f535ca, plus archived group-size patch.',
        'Selected-Key retention 20%; original nominal 20% Cluster count; all other method settings frozen.',
        'Support selection, coarse probes and history state use actual 128/256-query groups.',
        'Four evenly spaced coarse probes per group; actual history sampling interval remains 256 queries.',
        'One warmup + three full forwards, BF16, TF32 off. Each scene pair uses the SAME exclusive physical GPU.',
        'Includes reset, clustering, history, selection, reordering, CUDA Graph handling and execution.',
        'AUC/F1 shown in percent; ACC/Comp/CD in metres; CD=(ACC+Comp)/2 unsquared.',
        'Reconstruction uses pose + recon_unposed and archived thresholds/downsampling; no test-truth tuning.',
        'Dense not rerun per branch AGENTS.md; no contemporaneous matched Dense timing, so speedup vs Dense=N/A.',
        'No new group32 runs. Historical RTX5090 group32 timings are not local RTX4090 baselines.','',
        '| Dataset | Scene | Query group | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Forward s | Peak allocated GiB | Peak reserved GiB | vs Dense |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def cells(q):
        return ['N/A' if q['values'][k] is None else f"{q['values'][k]*(100 if k in ('auc03','auc30','f1') else 1):.5f}" for k in FIELDS]
    for dataset,scenes in SCENES.items():
        per={128:[],256:[]}
        for scene in scenes:
            matched=[]
            for size in (128,256):
                out=root/dataset/scene/f'q{size}'
                result=out/'result.json'
                if not result.exists():
                    status=out/'status.json'
                    failures.append(dict(dataset=dataset,scene=scene,size=size,status=json.loads(status.read_text()) if status.exists() else 'not run'))
                    if (out/'quality.json').exists():
                        q=json.loads((out/'quality.json').read_text())
                        lines.append('| '+' | '.join([dataset,scene,str(size),'N/A']+cells(q)+['N/A']*4)+' |')
                    continue
                item=json.loads(result.read_text());item['query_group_size']=size
                monitor_path=root/'monitors'/f'{dataset}_{scene}_q{size}.json'
                monitor=json.loads(monitor_path.read_text()) if monitor_path.exists() else None
                item['exclusive_gpu_monitor']=monitor
                if not monitor or monitor['same_gpu_overlap_detected'] or monitor['returncode']:
                    failures.append(dict(dataset=dataset,scene=scene,size=size,error='Invalid or missing exclusive-GPU monitor'))
                if item['quality']['complete']:
                    v=item['quality']['values']
                    assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-9
                    per[size].append(item)
                else:failures.append(dict(dataset=dataset,scene=scene,size=size,error=item['quality_errors']))
                runs.append(item);matched.append(item)
                t=item['timing'];s=t['samples']
                lines.append('| '+' | '.join([dataset,scene,str(size),str(item['frames'])]+cells(item['quality'])+
                    [f"{t['median_seconds']:.5f}",f"{max(p['peak_allocated_gib'] for p in s):.4f}",f"{max(p['peak_reserved_gib'] for p in s):.4f}",'N/A'])+' |')
            if len(matched)==2:
                a,b=matched
                for k in ('input','quality_protocol','hardware'):assert a[k]==b[k],(dataset,scene,k)
                assert a['exclusive_gpu_monitor']['gpu_uuid']==b['exclusive_gpu_monitor']['gpu_uuid']
                assert {k:v for k,v in a['config'].items() if k!='query_group_size'}=={k:v for k,v in b['config'].items() if k!='query_group_size'}
                pairs.append(dict(dataset=dataset,scene=scene,speedup_256_over_128=a['timing']['median_seconds']/b['timing']['median_seconds'],
                    quality_delta_256_minus_128={k:None if a['quality']['values'][k] is None or b['quality']['values'][k] is None else b['quality']['values'][k]-a['quality']['values'][k] for k in FIELDS}))
        if all(len(per[s])==len(scenes) for s in (128,256)):
            aggregates[dataset]={}
            for size in (128,256):
                group=per[size]
                aggregates[dataset][str(size)]=dict(scenes=list(scenes),frames=[x['frames'] for x in group],
                    values={k:sum(x['quality']['values'][k] for x in group)/len(group) for k in FIELDS},
                    mean_scene_median_seconds=sum(x['timing']['median_seconds'] for x in group)/len(group),
                    peak_allocated_gib=max(p['peak_allocated_gib'] for x in group for p in x['timing']['samples']),
                    peak_reserved_gib=max(p['peak_reserved_gib'] for x in group for p in x['timing']['samples']),
                    speedup_vs_dense=None,speedup_unavailable='No contemporaneous paired Dense; unchanged control not rerun per AGENTS.md')
            v=aggregates[dataset]
            v['speedup_256_over_128']=v['128']['mean_scene_median_seconds']/v['256']['mean_scene_median_seconds']
    lines+=['','## Equal-scene subset averages','',
        'Only heads/chess/stairs at128 frames, as requested. These are three-scene equal averages, NOT the seven-scene 7Scenes macro.','',
        '| Scope | Query group | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean scene median s | Peak allocated GiB | vs Dense |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for dataset,data in aggregates.items():
        for size in ('128','256'):
            a=data[size]
            lines.append('| '+' | '.join([dataset+' (3-scene subset)',size]+cells({'values':a['values']})+[f"{a['mean_scene_median_seconds']:.5f}",f"{a['peak_allocated_gib']:.4f}",'N/A'])+' |')
        lines+=['',f"{dataset}: 256 vs 128 full-forward speedup = {data['speedup_256_over_128']:.4f}x.",'']
    lines+=['','## Per-scene 256 / 128 comparison','',
        '| Dataset | Scene | Speedup 256 over 128 | Δ AUC@3 pp | Δ AUC@30 pp | Δ ACC m | Δ Comp m | Δ CD m | Δ F1 pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        d=pair['quality_delta_256_minus_128']
        lines.append('| '+' | '.join([pair['dataset'],pair['scene'],f"{pair['speedup_256_over_128']:.4f}x"]+
            ['N/A' if d[k] is None else f"{d[k]*(100 if k in ('auc03','auc30','f1') else 1):+.5f}" for k in FIELDS])+' |')
    if failures:lines+=['','Incomplete entries: '+json.dumps(failures,ensure_ascii=False)]
    summary=dict(per_run=runs,paired_comparisons=pairs,subset_average=aggregates,incomplete=failures,
        complete=len(runs)==2*sum(len(v) for v in SCENES.values()) and not failures,speedup_vs_dense=None,
        speedup_unavailable='No contemporaneous paired Dense; unchanged control not rerun per AGENTS.md')
    save(root/'summary.json',summary);(root/'report.md').write_text('\n'.join(lines)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[3,4,5])
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/query_group_scaling_20261104')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();root=a.output_root
    if a.report_only:report(root);return
    root.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'configs/evaluation/reuse_shared_value_loss_256.json').read_text())
    for size in (128,256):save(ROOT/f'configs/evaluation/reuse_shared_value_loss_q{size}.json',dict(base,query_group_size=size))
    scenes=[(d,s) for d,ss in SCENES.items() for s in ss]
    manifest=dict(source_branch='codex/shared-value-cluster-scaling-20261104',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        groups=[128,256],scenes=SCENES,frames=128,warmup=1,repeats=3,gpus=a.gpus,
        group32_not_run=True,dense_not_run='Changed-method-only requirement in AGENTS.md',
        history_interval=256,coarse_probes=4,base_config=base)
    save(root/'manifest.json',manifest)
    (root/'group_size.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    def worker(gpu,jobs):
        completed=[]
        for job_index,(dataset,scene) in jobs:
            for size in ((128,256) if job_index%2==0 else (256,128)):
                output=root/dataset/scene/f'q{size}'
                monitor=root/'monitors'/f'{dataset}_{scene}_q{size}.json'
                if (output/'result.json').exists() and monitor.exists():continue
                if output.exists():raise RuntimeError(f'Existing incomplete result preserved: {output}')
                command=[sys.executable,'scripts/run_gpu_lease.py','--strict-contexts','--gpu',str(gpu),
                    '--monitor',str(monitor),'--log',str(root/'logs'/f'{dataset}_{scene}_q{size}.log'),'--',
                    sys.executable,'scripts/evaluate_zdrop_history_states.py','--dataset',dataset,'--scene',scene,
                    '--frames','128','--method','zdrop_history_states','--config',f'configs/evaluation/reuse_shared_value_loss_q{size}.json',
                    '--output-dir',str(output),'--warmup','1','--repeats','3']
                print('START',dataset,scene,size,'GPU',gpu,flush=True)
                proc=subprocess.run(command,cwd=ROOT)
                completed.append(dict(dataset=dataset,scene=scene,group=size,gpu=gpu,returncode=proc.returncode))
                save(root/f'worker_gpu{gpu}.json',completed)
                print('END',dataset,scene,size,proc.returncode,flush=True)
                if proc.returncode:raise RuntimeError(f'Worker failed; see {output} and log')
        return completed
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        jobs=[pool.submit(worker,g,[(i,x) for i,x in enumerate(scenes) if i%len(a.gpus)==j]) for j,g in enumerate(a.gpus)]
        statuses=[]
        for job in jobs:
            try:statuses.append(dict(complete=True,runs=job.result()))
            except Exception as exc:statuses.append(dict(complete=False,error=str(exc)))
    save(root/'status.json',statuses)
    summary=report(root)
    print('ALL_COMPLETE',summary['complete'],root/'report.md',flush=True)

if __name__=='__main__':main()
