"""75 quality-first jobs:3 reproduced sparse baselines +24 single-layer restorations per scene.

Only one GPU-forward job per device. A second process may overlap CPU-only
quality with that forward. Never interrupt/allocate over an unrelated GPU job.
Failures/retries retain separate attempt directories and all six N/A fields.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_full_global_output_comparison import SOURCE_FILES,save
from scripts.evaluate_single_dense_history80 import run_name
from scripts.quality_metrics import quality_record,quality_cells,LABELS,FIELDS


def reproduced(old,new,tolerance=1e-5):
    return all(old[k] is not None and new[k] is not None and abs(old[k]-new[k])<=tolerance for k in FIELDS)


def read_result(root,job):
    selected=job.get('selected_result')
    return json.loads(Path(selected).read_text()) if selected else None


def report(args,jobs,plots=False):
    rows=[];rankings={};summaries={}
    lines=['# Original population History80:single Dense-layer restoration','',
        'Selected diagnostic subset ONLY:heads,stairs,chess;not a complete7Scenes dataset evaluation.',
        '128 uniform frames,seed0,exact previous frame/checkpoint/preprocessing/pose/fusion protocol.',
        'Original middle A80/V20,shallow/deep A20/V80,L0 pureValue;history80%,population compensation ON.',
        'One restored global layer returns Dense output,while its shadow sparse operator still updates all state/history.',
        'Future masks regenerate naturally;no Dense-derived history,no frozen mask,and frame/local attention unchanged.',
        'AUC/F1 percent,ACC/Comp/CD metres,CD=(ACC+Comp)/2 unsquared;F1threshold0.05m,downsample0.0078125m.',
        'Cold diagnostic seconds and memory are recorded separately;no warmed speed comparisons. Sparse speedup N/A.',
        'Dense1.00x is reference identity;historical rows are reused quality,not contemporaneous timing baselines.',
        'Each restoration is a one-layer intervention;interactions mean scores need not improve monotonically.',
        'Labels and snapshots preserve any failed attempts and unchanged retries.']
    for scene in args.scenes:
        baseline_job=jobs[f'{scene}/sparse_baseline'];fresh=read_result(args.output_dir,baseline_job)
        old=json.loads((args.baseline/'7scenes'/scene/'history80/result.json').read_text())
        dense=json.loads((args.baseline/'7scenes'/scene/'dense/result.json').read_text())
        rows_for_scene=[]
        lines+=['',f'## {scene}','',f'Fresh baseline reproduction: {baseline_job.get("baseline_reproduced","pending")}',
            '', '| Method | '+' | '.join(LABELS)+' | Speedup |','|---|'+'---:|'*7]
        for name,ref in [('Dense (reused)',dense),('Original population History80 (reused)',old)]:
            lines.append('| '+name+' | '+' | '.join(quality_cells(ref['quality']))+
                (' | 1.00x reference |' if name.startswith('Dense') else ' | N/A |'))
            rows.append(dict(scene=scene,method=name,restored_layer=-2,status='reused quality',
                result_source=str(args.baseline/'7scenes'/scene/('dense' if name.startswith('Dense') else 'history80')/'result.json'),
                **ref['quality']['values'],auc03_delta_pp=None,auc30_delta_pp=None,f1_delta_pp=None,cd_delta_m=None,
                diagnostic_seconds=None,peak_allocated_gib=None,speedup_vs_vggt=1. if name.startswith('Dense') else None))
        for layer in [-1,*range(24)]:
            name=run_name(layer);job=jobs[f'{scene}/{name}'];result=read_result(args.output_dir,job)
            q=result['quality'] if result else quality_record({},f'{job["stage"]}:single-layer ablation pending/failed;see queue_status.json')
            vals=q['values'];deltas={k:None for k in ('auc03_delta_pp','auc30_delta_pp','f1_delta_pp','cd_delta_m')}
            if result and fresh and q['complete']:
                for k in ('auc03','auc30','f1'):deltas[k+'_delta_pp']=100*(vals[k]-fresh['quality']['values'][k])
                deltas['cd_delta_m']=vals['cd']-fresh['quality']['values']['cd']
            row=dict(scene=scene,method=name,restored_layer=layer,status=job['stage'],
                result_source=job.get('selected_result'),**vals,**deltas,
                diagnostic_seconds=None if result is None else result['diagnostic_seconds'],
                peak_allocated_gib=None if result is None else result['peak_allocated_gib'],speedup_vs_vggt=None)
            rows.append(row)
            if layer>=0 and result and q['complete']:rows_for_scene.append(row)
            lines.append('| '+name+' | '+' | '.join(quality_cells(q))+' | N/A |')
        rank={}
        if baseline_job.get('baseline_reproduced'):
            for metric in ('auc03_delta_pp','auc30_delta_pp','f1_delta_pp'):
                rank[metric]=sorted(rows_for_scene,key=lambda r:r[metric],reverse=True)
        rankings[scene]=rank
        summaries[scene]=dict(complete_layers=len(rows_for_scene),total_layers=24,
            baseline_reproduced=baseline_job.get('baseline_reproduced'),
            all_complete=len(rows_for_scene)==24 and bool(baseline_job.get('baseline_reproduced')))
        lines+=['','### Changes against fresh sparse baseline','',
            '| Restored layer | ΔAUC@3 pp | ΔAUC@30 pp | ΔF1 pp | ΔCD m |','|---|---:|---:|---:|---:|']
        for row in sorted(rows_for_scene,key=lambda r:r['restored_layer']):
            lines.append('| '+str(row['restored_layer'])+' | '+' | '.join(f'{row[k]:+.5f}' for k in
                ('auc03_delta_pp','auc30_delta_pp','f1_delta_pp','cd_delta_m'))+' |')
        if plots and rows_for_scene:plot(args.output_dir,scene,rows_for_scene)
    if rows:
        with (args.output_dir/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    save(args.output_dir/'summary.json',dict(complete=all(v['all_complete'] for v in summaries.values()),
        subset_not_dataset_macro=True,scenes=summaries,rankings=rankings))


def plot(root,scene,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(rows,key=lambda r:r['restored_layer'])
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    for ax,k,label in zip(axes,('auc03_delta_pp','auc30_delta_pp','f1_delta_pp'),('AUC@3','AUC@30','F1')):
        v=[r[k] for r in rows]
        ax.bar([r['restored_layer'] for r in rows],v,color=['tab:blue' if x>=0 else 'tab:red' for x in v])
        ax.axhline(0,color='black',linewidth=.7);ax.set_ylabel(f'Delta {label} (pp)');ax.grid(axis='y',alpha=.25)
    axes[-1].set_xticks(range(24));axes[-1].set_xlabel('Single restored global layer (zero based)')
    fig.suptitle(f'{scene},128 frames | Dense output at ONE layer,original sparse state preserved')
    fig.savefig(root/f'{scene}_restoration.png',dpi=170);plt.close(fig)


def campaign(args):
    if args.output_dir.exists():raise ValueError('Fresh campaign required')
    args.output_dir.mkdir(parents=True)
    names=(*SOURCE_FILES,'saf3r/patch/single_dense_av_ablation.py','scripts/evaluate_single_dense_history80.py',
        'scripts/run_single_dense_history80.py','configs/evaluation/parent_region_history80_population.json')
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    save(args.output_dir/'manifest.json',dict(scenes=args.scenes,gpus=args.gpus,frames=128,source_sha256=hashes,
        baseline=str(args.baseline.resolve()),single_restored_global_layers=list(range(24)),
        quality_tasks=['pose','recon_unposed'],shadow_sparse_state_preserved=True,max_processes_per_gpu=2,
        gpu_overlap='at most1 GPU-forward;second process only overlaps CPU quality',
        selection='heads:max7Scenes AUC3 loss;stairs:max AUC30/F1 loss;chess:normal control',
        not_parameter_search=True,diagnostic_not_speed_benchmark=True))
    priority=[*range(10,17),*range(17,24),*range(10)]
    specs=[(s,-1) for s in args.scenes]+[(s,l) for l in priority for s in args.scenes]
    jobs={f'{s}/{run_name(l)}':dict(scene=s,layer=l,stage='queued',attempts=[]) for s,l in specs}
    save(args.output_dir/'queue_status.json',jobs)
    uuid_to_index={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines():
        g,u=line.split(',');uuid_to_index[u.strip()]=int(g)
    active={};last_report=0
    while True:
        dirty=False
        for key,a in list(active.items()):
            if a['process'].poll() is None:continue
            dirty=True;job=jobs[key];attempt=job['attempts'][-1];attempt['exit_code']=a['process'].returncode
            a['log'].close();del active[key]
            p=Path(attempt['directory'])/'result.json'
            result=json.loads(p.read_text()) if p.exists() else None
            if a['process'].returncode==0 and result and result['quality']['complete']:
                job.update(stage='complete',selected_result=str(p))
                if job['layer']==-1:
                    old=json.loads((args.baseline/'7scenes'/job['scene']/'history80/result.json').read_text())
                    ok=(result['input']==old['input'] and result['quality_protocol']==old['quality_protocol']
                        and reproduced(old['quality']['values'],result['quality']['values']))
                    job['baseline_reproduced']=ok
                    if not ok:job['stage']='reproduction_failed'
            else:
                job['stage']='queued' if len(job['attempts'])<args.max_attempts else 'failed'
                attempt['error']='Worker failed/incomplete metrics;see attempt status/log'
            attempt['finished_unix']=time.time()
        for key,job in jobs.items():
            if job['layer']>=0 and job['stage']=='queued':
                b=jobs[f'{job["scene"]}/sparse_baseline']
                if b['stage'] in ('failed','reproduction_failed'):
                    job.update(stage='blocked',reason='Fresh baseline failed reproduction/quality');dirty=True
        # Check process ownership,not merely aggregate VRAM. A foreign task is never killed.
        foreign={g:False for g in args.gpus};owned={a['process'].pid for a in active.values()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'],text=True)
        for line in apps.splitlines():
            fields=[v.strip() for v in line.split(',')]
            if len(fields)>=2 and fields[0] in uuid_to_index:
                gpu=uuid_to_index[fields[0]]
                if gpu in foreign and fields[1].isdigit() and int(fields[1]) not in owned:foreign[gpu]=True
        for gpu in args.gpus:
            on_gpu=[a for a in active.values() if a['gpu']==gpu]
            if foreign[gpu] or len(on_gpu)>=2:continue
            heavy=False
            for a in on_gpu:
                status=Path(a['directory'])/'status.json'
                stage=json.loads(status.read_text()).get('stage') if status.exists() else 'initializing'
                if stage not in ('quality','complete','quality_incomplete'):heavy=True
            if heavy:continue
            key=next((k for k,j in jobs.items() if j['stage']=='queued' and
                (j['layer']==-1 or jobs[f'{j["scene"]}/sparse_baseline'].get('baseline_reproduced'))),None)
            if key is None:continue
            # Fail closed if sources changed during a campaign.
            for name,digest in hashes.items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed:'+name)
            job=jobs[key];number=len(job['attempts']);attempt_root=args.output_dir/f'attempt_{number:02d}'
            directory=attempt_root/job['scene']/run_name(job['layer'])
            logpath=args.output_dir/'logs'/f'{job["scene"]}_{run_name(job["layer"])}_attempt{number}.log'
            logpath.parent.mkdir(exist_ok=True);stream=logpath.open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/evaluate_single_dense_history80.py'),
                '--scene',job['scene'],'--dense-layer',str(job['layer']),'--frames','128',
                '--output-dir',str(attempt_root),'--baseline',str(args.baseline)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            attempt=dict(gpu=gpu,pid=proc.pid,directory=str(directory),log=str(logpath),started_unix=time.time())
            job['attempts'].append(attempt);job['stage']='running';dirty=True
            active[key]=dict(process=proc,gpu=gpu,directory=str(directory),log=stream)
        if dirty:save(args.output_dir/'queue_status.json',jobs)
        if dirty and time.time()-last_report>5:
            report(args,jobs);last_report=time.time()
        if not active and not any(j['stage']=='queued' for j in jobs.values()):break
        time.sleep(2)
    report(args,jobs,plots=True)
    complete=all(j['stage']=='complete' for j in jobs.values())
    save(args.output_dir/'status.json',dict(complete=complete,completed_jobs=sum(j['stage']=='complete' for j in jobs.values()),
        total_jobs=len(jobs),all_baselines_reproduced=all(jobs[f'{s}/sparse_baseline'].get('baseline_reproduced') for s in args.scenes)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',default=['heads','stairs','chess'],choices=['heads','stairs','chess'])
    p.add_argument('--gpus',nargs='+',type=int,default=list(range(7)))
    p.add_argument('--max-attempts',type=int,default=3)
    args=p.parse_args();campaign(args)
