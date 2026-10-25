"""Evaluate only the pinned unified-refactor branch; reuse four frozen baselines.

Never starts baseline reruns, silently changes parameters, or uses other GPUs.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_recovery_guard import query_inventory,bounded_command,FAULT_UUID
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record,quality_cells

SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
OLD_METHODS=('dense','saf3r','history80','reuse_explore')
LABELS={'dense':'VGGT Dense (historical)','saf3r':'SAF3R lazy Top-4 (historical)',
    'history80':'History80 + population (historical)','reuse_explore':'Original reuse + residual (historical)',
    'unified':'Unified refactor (new)'}
PIN='3afe65a038b01e34732493beddf9bf9eb63af76c'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def report(root,baseline):
    rows=[]
    for scene in SCENES:
        reference=json.loads((baseline/scene/'dense/result.json').read_text())
        for method in (*OLD_METHODS,'unified'):
            path=(root/'quality128'/scene/'result.json') if method=='unified' else baseline/scene/method/'result.json'
            if path.exists():
                data=json.loads(path.read_text())
                if data['input']!=reference['input'] or data['quality_protocol']!=reference['quality_protocol']:
                    raise ValueError(f'Historical matched-input/protocol mismatch: {scene}/{method}')
                q=data['quality'];timing=data['timing']
            else:
                data={};p=root/'quality128'/scene/'quality.json'
                q=json.loads(p.read_text()) if p.exists() else quality_record({},'Queued; this scene has not finished')
                timing=None
            if method=='unified':
                q=dict(q,speedup_vs_vggt=None,speedup_unavailable=
                    'User requested no Dense rerun; historical timings are not contemporaneous/same-UUID paired timings')
            rows.append(dict(scene=scene,method=method,source=str(path),quality=q,timing=timing,
                             saf3r=data.get('saf3r'),hardware=data.get('hardware'),
                             quality_protocol=data.get('quality_protocol',reference['quality_protocol']),
                             phase='new branch evaluation' if method=='unified' else 'historical frozen evaluation'))
    table=['# Unified branch vs four existing baselines: 7Scenes 128 frames','',
        f'Branch `refactor/unified-reuse-exploration`, pinned upstream `{PIN}`.',
        'Only the new branch is evaluated. Four old methods are read from the frozen seven-scene results, NOT rerun.',
        'Exact checkpoint, 128-frame lists, preprocessing, pose protocol and unposed reconstruction settings are checked.',
        'AUC/F1 percent; distances metres; CD=(ACC+Comp)/2 unsquared.',
        'F1 threshold .05m; downsample .0078125m. One warmup and three whole forwards, including all selector/history costs.',
        'SAF3R: lazy first-correspondence-head Top-4, ETH3D calibration profile; not paper DINO Top-K reproduction.',
        'Old timings/speedups belong to their original measurement phase; new runtime is shown separately.',
        'New vs Dense end-to-end speedup is N/A: no contemporaneous same-UUID Dense timing, per user no-rerun instruction.',
        'No parameter search against test truth; no Dense fallback or silent budget/frame reductions.',
        '', '| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Forward s | Peak allocated GiB | Speedup¹ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        t=r['timing'];speed=r['quality'].get('speedup_vs_vggt')
        peak=None if t is None else max(s['peak_allocated_gib'] for s in t['samples'])
        table.append('| '+' | '.join([r['scene'],LABELS[r['method']]]+quality_cells(r['quality'])+[
            'N/A' if t is None else f"{t['median_seconds']:.3f}",
            'N/A' if peak is None else f'{peak:.3f}', 'N/A' if speed is None else f'{speed:.3f}x'])+' |')
    macro=[]
    table+=['','## Equal-scene macro: only all seven complete scenes qualify','',
        '| Method | Complete | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean scene median s | Speedup¹ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in (*OLD_METHODS,'unified'):
        complete=[r for r in rows if r['method']==method and r['quality']['complete']]
        q=quality_record({},'All seven scenes required for macro')
        item=dict(method=method,complete=len(complete),required=7,quality=q,mean_scene_median_seconds=None)
        if len(complete)==7:
            q['values']={k:sum(r['quality']['values'][k] for r in complete)/7 for k in q['values']}
            q['complete']=True;q['unavailable']={}
            item['mean_scene_median_seconds']=sum(r['timing']['median_seconds'] for r in complete)/7
            if method!='unified':
                dense=[r for r in rows if r['method']=='dense']
                q['speedup_vs_vggt']=(sum(r['timing']['median_seconds'] for r in dense)/7)/item['mean_scene_median_seconds']
                q['speedup_unavailable']=None
            else:q['speedup_unavailable']='No fresh matched Dense per user instruction; historical timings not contemporaneous'
        if q['complete']:
            dense_values={k:sum(r['quality']['values'][k] for r in rows if r['method']=='dense')/7 for k in q['values']}
            item['delta_vs_local_dense']={k:(v-dense_values[k])*(100 if k in ('auc03','auc30','f1') else 1)
                                         for k,v in q['values'].items()}
        macro.append(item);speed=q.get('speedup_vs_vggt');seconds=item['mean_scene_median_seconds']
        table.append('| '+' | '.join([LABELS[method],f'{len(complete)}/7']+quality_cells(q)+[
            'N/A' if seconds is None else f'{seconds:.3f}','N/A' if speed is None else f'{speed:.3f}x'])+' |')
    table+=['','¹ Historical speedups are retained only within their original phase. Do not rank new/old runtime as a matched speed comparison.',
            'Published paper scores/H100 times are not mixed into this local RTX4090 comparison.']
    save(root/'comparison.json',dict(per_run=rows,macro=macro,pinned_upstream=PIN,all_seven_required=True))
    (root/'comparison.md').write_text('\n'.join(table)+'\n')


def run(a):
    branch=a.branch_dir.resolve();root=a.output_dir.resolve();baseline=a.baseline_dir.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    commit=bounded_command(['git','-C',str(branch),'rev-parse','HEAD']).strip()
    if commit!=PIN:raise ValueError('Branch checkout does not match frozen upstream commit')
    inventory=query_inventory();by={d['gpu']:d for d in inventory}
    devices=[by[g] for g in (4,5)]
    if any(d['uuid'].lower()==FAULT_UUID.lower() for d in devices):raise ValueError('Quarantined GPU requested')
    source_names=('saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py',
        'saf3r/patch/fused_shared_av.py','saf3r/patch/value_clustering_gpu.py',
        'scripts/evaluate_reuse_exploration.py','configs/evaluation/reuse_residual_exploration.json')
    hashes={n:digest(branch/n) for n in source_names}
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['source_sha256']!=hashes:raise ValueError('Experiment sources changed; do not resume under changed implementation')
    else:
        if not (root/'preflight/status.json').exists() or json.loads((root/'preflight/status.json').read_text())['stage']!='diagnostic_complete':
            raise ValueError('Successful branch GPU preflight required before dataset launch')
        originals=[]
        for scene in SCENES:
            dense=json.loads((baseline/scene/'dense/result.json').read_text())
            for m in OLD_METHODS:
                p=baseline/scene/m/'result.json';d=json.loads(p.read_text())
                if not d['quality']['complete'] or d['input']!=dense['input'] or d['quality_protocol']!=dense['quality_protocol']:
                    raise ValueError('Incomplete/mismatched historical baseline')
                originals.append(dict(scene=scene,method=m,path=str(p),sha256=digest(p)))
        save(manifest_path,dict(upstream_commit=PIN,branch='refactor/unified-reuse-exploration',
            branch_dir=str(branch),source_sha256=hashes,devices=devices,scenes=SCENES,frames=128,
            warmup=1,repeats=3,tasks=['pose','recon_unposed'],baseline_reruns=False,no_truth_tuning=True,
            baselines=originals,checkpoint=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'),
            speedup_unavailable='No new Dense timing; historical measurements belong to different phase'))
        (root/'evaluation_overlay.patch').write_text(bounded_command(['git','-C',str(branch),'diff','--',*source_names]))
    jobs=[dict(scene=s,output=str(root/'quality128'/s)) for s in SCENES]
    done={};live={};logs=[];wait_since={};last_health=0
    for i,j in enumerate(jobs):
        out=Path(j['output'])
        if (out/'result.json').exists() and json.loads((out/'result.json').read_text())['quality']['complete']:
            done[i]='complete'
        elif (out/'status.json').exists():
            state=json.loads((out/'status.json').read_text())['stage']
            if state in ('failed','quality_incomplete'):done[i]=state
            else:raise RuntimeError('Nonterminal existing run: inspect worker before resume, no duplicate launched')
    while len(done)<len(jobs):
        if time.monotonic()-last_health>=60:
            try:observed=query_inventory([d['uuid'] for d in devices])
            except Exception as e:
                save(root/'blocked.json',dict(reason='GPU health query failed; no further dispatch',error=str(e)));report(root,baseline);return
            last_health=time.monotonic()
            save(root/'gpu_health.json',dict(utc=datetime.now(timezone.utc).isoformat(),devices=observed))
        for i,p in list(live.items()):
            if p.poll() is not None:
                out=Path(jobs[i]['output']);status=out/'status.json'
                state=json.loads(status.read_text())['stage'] if status.exists() else 'failed'
                if p.returncode or state not in ('complete','quality_incomplete'):state='failed'
                done[i]=state;del live[i];continue
            try:
                state=next(x for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('State:')).split()[1]
                wchan=Path(f'/proc/{p.pid}/wchan').read_text().strip()
            except (OSError,StopIteration):state='unknown';wchan='unknown'
            if state=='D':wait_since.setdefault(i,time.monotonic())
            else:wait_since.pop(i,None)
            limit=900 if wchan=='wait_on_page_bit_common' else 120
            if i in wait_since and time.monotonic()-wait_since[i]>limit:
                reason=f'Worker sustained kernel wait ({wchan}); stopped dispatch, no GPU reset or unrelated process action'
                save(root/'blocked.json',dict(reason=reason,worker_pid=p.pid,scene=jobs[i]['scene']))
                write_quality(Path(jobs[i]['output']),quality_record({},reason));report(root,baseline);return
        wait_since={i:t for i,t in wait_since.items() if i in live}
        for device in devices:
            gpu=device['gpu'];peers=[i for i in live if jobs[i].get('gpu')==gpu]
            # One process/card, including CPU quality, avoids noisy paired-runtime claims.
            if peers or wait_since:continue
            try:current=query_inventory([device['uuid']])[0]
            except Exception:continue
            if current['memory_used_mib']>256:continue
            i=next((i for i in range(len(jobs)) if i not in done and i not in live),None)
            if i is None:break
            if any(digest(branch/n)!=h for n,h in hashes.items()):raise RuntimeError('Frozen source changed')
            j=jobs[i];j.update(gpu=gpu,device_uuid=device['uuid'])
            cmd=[sys.executable,'scripts/evaluate_reuse_exploration.py','--method','reuse_explore',
                '--scene',j['scene'],'--frames','128','--output-dir',j['output'],
                '--checkpoint',str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'),
                '--expected-input',str(baseline/j['scene']/'dense/input.json')]
            logpath=root/'logs'/f"{j['scene']}.log";logpath.parent.mkdir(exist_ok=True)
            handle=logpath.open('w');logs.append(handle)
            live[i]=subprocess.Popen(cmd,cwd=branch,env={**os.environ,'CUDA_VISIBLE_DEVICES':device['uuid'],
                'PYTHONPATH':str(branch),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        statuses=[dict(j,state=done.get(i,'running' if i in live else 'queued'),pid=live[i].pid if i in live else None)
            for i,j in enumerate(jobs)]
        save(root/'queue_status.json',dict(jobs=statuses,counts=dict(Counter(j['state'] for j in statuses)),
            all_terminal=len(done)==len(jobs),fixed_gpu_allowlist=[4,5]))
        report(root,baseline)
        if len(done)<len(jobs):time.sleep(15)
    for h in logs:h.close()
    save(root/'completion.json',dict(all_seven_complete=all(v=='complete' for v in done.values()),
        states=done,utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,default=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128')
    p.add_argument('--pin',default=PIN,help='Frozen upstream commit for this independent evaluation')
    p.add_argument('--report-only',action='store_true');a=p.parse_args()
    PIN=a.pin
    if a.report_only:report(a.output_dir.resolve(),a.baseline_dir.resolve())
    else:run(a)
