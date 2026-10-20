"""Idle-GPU queue for Dense-teacher ratio search and actual sparse validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_dense_teacher_ratios import policies,save
from scripts.quality_metrics import FIELDS,LABELS,quality_record,quality_cells


def select_recommendation(rows):
    manual=next(r for r in rows if r['name']=='manual')
    eligible=[r for r in rows if r['name']!='manual' and r['quality']['complete']
        and r['seconds']<=manual['seconds'] and r['speedup_vs_vggt']>=1]
    # GT metrics are deliberately not part of selection, even after evaluation.
    return min(eligible,key=lambda r:(r['teacher_train_nmse'],r['name']))['name'] if eligible else None


def report_scene(root,scene):
    folder=root/scene;dense=json.loads((folder/'dense_result.json').read_text())
    selected=json.loads((folder/'selected.json').read_text())
    specs=[dict(name='manual',schedule=str(folder/'manual.json'),
        stages=[.2,.8,.2],train_nmse=selected['manual_baseline']['train_nmse']),*selected['selected']]
    rows=[];missing=[]
    for spec in specs:
        path=folder/'validation'/spec['name']/'region'/scene/'result.json'
        if not path.exists():missing.append(spec['name']);continue
        d=json.loads(path.read_text())
        if d['input']!=dense['input'] or d['hardware']!=dense['hardware']:
            raise ValueError('Dense and sparse input/hardware mismatch')
        cfg=json.loads(Path(spec['schedule']).read_text())
        if d['config']!=cfg:raise ValueError('Validation did not execute the searched schedule')
        if d['protocol']['warmup']!=1 or d['protocol']['repeats']!=3:
            raise ValueError('Sparse timing protocol mismatch')
        rows.append(dict(name=spec['name'],stages=spec['stages'],teacher_train_nmse=spec['train_nmse'],
            quality=quality_record(d['metrics']),seconds=d['median_seconds'],peak_gib=d['median_peak_gib'],
            speedup_vs_vggt=dense['median_seconds']/d['median_seconds'],samples=d['samples'],source=str(path)))
    result=dict(scene=scene,frames=dense['frames'],complete=not missing,missing=missing,
        dense=dict(quality=quality_record(dense['metrics']),seconds=dense['median_seconds'],
            peak_gib=dense['median_peak_gib'],samples=dense['samples'],speedup_vs_vggt=1.),
        rows=rows,recommendation=select_recommendation(rows) if not missing else None,
        sequence_specific_oracle=True,held_out_benchmark=False,ground_truth_used_for_selection=False,
        speed_gate='Measured sparse median <= same-GPU manual median AND <= Dense median; point estimates, not statistical guarantees',
        search_cost_included_in_sparse_seconds=False,
        warning='Dense search is offline oracle overhead. Including it for every new sequence invalidates a sparse-only speedup claim.')
    save(folder/'comparison.json',result)
    lines=[f'# Sequence-specific Dense-teacher ratio search: {scene}', '',
        f"Complete: {not missing}. Frames: {dense['frames']}. Missing: {missing}.", '',
        '**Diagnostic sequence-specific oracle search, NOT a deployable adaptive policy or held-out paper result.** '
        'Ratios are ranked ONLY by sampled Dense Attention AV error; end-to-end GT scores are evaluation only. '
        'Teacher forcing does not reproduce accumulated sparse hidden-state error, hence actual validation is required.', '',
        '| Scheme | Attention shallow/middle/deep | '+' | '.join(LABELS)+' | Seconds | Speedup | Peak GiB |',
        '|---|---|'+'---:|'*9]
    base=result['dense']
    lines.append('| Dense | — | '+' | '.join(quality_cells(base['quality']))+
        f" | {base['seconds']:.4f} | 1.000x | {base['peak_gib']:.4f} |")
    for row in rows:
        lines.append('| '+row['name']+' | '+str(row['stages'])+' | '+' | '.join(quality_cells(row['quality']))+
            f" | {row['seconds']:.4f} | {row['speedup_vs_vggt']:.3f}x | {row['peak_gib']:.4f} |")
    lines+=['',f"Speed-feasible Dense-teacher recommendation: {result['recommendation']}.", '',
        'Total target retention20%, history80%, whole Parent selection, all24 global layers. '
        'Layer0 remains pure Value. Fixed-ratio inference includes all normal initialization/selection/history costs. '
        'Dense baseline and every validation use one warmup and three complete forwards on the same GPU. '
        'Search/Dense-teacher overhead is separate in teacher_result.json and must be added for an online-search claim. '
        'All six quality metrics are reported; CD=(ACC+Comp)/2 in metres, F1 threshold0.05m. '
        'Only the top2 teacher candidates and the manual schedule receive full timing validation; '
        'this is not a certified speed-feasible optimum over all75 schedules. '
        'No quality acceptance threshold is silently relaxed; no Dense fallback is inserted.', '']
    (folder/'comparison.md').write_text('\n'.join(lines))
    return result


def run(args):
    from saf3r.patch.value_region_av import ValueRegionAVConfig
    if not args.gpus or len(args.gpus)!=len(set(args.gpus)):raise ValueError('Distinct GPUs required')
    if args.output_dir.exists():raise ValueError('Use a fresh search directory')
    root=args.output_dir;root.mkdir(parents=True)
    core=['scripts/run_dense_ratio_search.py','scripts/search_dense_teacher_ratios.py','scripts/evaluate_fused_shared_av.py',
        'scripts/quality_metrics.py','saf3r/patch/value_region_av.py','saf3r/patch/fused_shared_av.py',
        'saf3r/patch/value_clustering_gpu.py','saf3r/patch/parent_region_selection.py',
        'saf3r/patch/region_history_attention.py','saf3r/triton/fused_shared_av.py',
        'saf3r/triton/value_region_selection.py','saf3r/triton/shared_av_routing.py',
        'evaluation/evaluate_da3bench.py','evaluation/benchmarks/da3/da3bench_utils.py',
        'evaluation/benchmarks/da3/bench/datasets/sevenscenes.py']
    hashes={}
    for name in core:
        data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
        p=root/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    save(root/'manifest.json',dict(created_unix=time.time(),scenes=args.scenes,frames=args.frames,
        candidates=policies(),candidate_count=75,query_groups=args.query_groups,top=args.top,
        gpus=args.gpus,source_sha256=hashes,ground_truth_used_for_selection=False,
        purpose='Dense-teacher sequence-specific oracle; no held-out benchmark claims',
        selection='Mean relative squared Attention AV error on even sampled groups, layers1..23',
        validation='Odd sampled groups not used for ranking; then full sparse autoregressive forward',
        allocation='Dynamic idle GPU workers; preserve existing128-frame and300-frame jobs'))
    pending=list(args.scenes);lock=Lock();jobs={s:dict(stage='queued') for s in pending};failures=[]
    def update(scene,**fields):
        with lock:
            jobs[scene].update(fields);save(root/'status.json',dict(stage='running',jobs=jobs,failures=failures))
    save(root/'status.json',dict(stage='queued',jobs=jobs,failures=[]))
    def ready(gpu):
        dependency={4:args.after128,6:args.after300}.get(gpu)
        if dependency:
            p=dependency/'status.json'
            if not p.exists() or json.loads(p.read_text()).get('stage') not in ('complete','failed'):return False
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=60).strip())
        return used<=512
    def execute(command,log,env):
        for name,digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Experiment source changed: {name}')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    def worker(gpu):
        while True:
            with lock:
                if not pending:return
            if not ready(gpu):time.sleep(15);continue
            with lock:
                if not pending:return
                scene=pending.pop(0)
            folder=root/scene
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            try:
                update(scene,stage='teacher',gpu=gpu,started_unix=time.time())
                print('SEARCH_START',gpu,scene,flush=True)
                execute([sys.executable,'scripts/search_dense_teacher_ratios.py','--output-dir',str(folder),
                    '--scene',scene,'--frames',str(args.frames),'--query-groups',str(args.query_groups),
                    '--top',str(args.top)],root/'logs'/f'{scene}_teacher.log',env)
                manual=asdict(ValueRegionAVConfig(parent_only=True,history_fraction=.8));save(folder/'manual.json',manual)
                selected=json.loads((folder/'selected.json').read_text())['selected']
                specs=[dict(name='manual',schedule=str(folder/'manual.json')),*selected]
                for spec in specs:
                    update(scene,stage='validation',candidate=spec['name'])
                    print('VALIDATE',gpu,scene,spec['name'],flush=True)
                    execute([sys.executable,'scripts/evaluate_fused_shared_av.py','--output-dir',str(folder/'validation'/spec['name']),
                        '--mode','region','--scene',scene,'--frames',str(args.frames),'--schedule',spec['schedule'],
                        '--parent-only','--warmup','1','--repeats','3','--tasks','pose','depth','recon_unposed'],
                        root/'logs'/f"{scene}_{spec['name']}.log",env)
                report_scene(root,scene)
                save(folder/'status.json',dict(stage='validation_complete',finished_unix=time.time()))
                update(scene,stage='complete',finished_unix=time.time())
                print('SEARCH_COMPLETE',gpu,scene,flush=True)
            except Exception as error:
                with lock:failures.append(dict(scene=scene,error=str(error)))
                update(scene,stage='failed',error=str(error))
                print('SEARCH_FAILED',scene,str(error),flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in args.gpus]
        for future in futures:future.result()
    save(root/'status.json',dict(stage='failed' if failures else 'complete',jobs=jobs,failures=failures,
        finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',default=['heads','office','stairs']);p.add_argument('--frames',type=int,default=128)
    p.add_argument('--gpus',type=int,nargs='+',default=[5,4,6]);p.add_argument('--query-groups',type=int,default=16)
    p.add_argument('--top',type=int,default=2)
    p.add_argument('--after128',type=Path,default=ROOT/'outputs/parent_region_av/three_method_quality_7scenes128')
    p.add_argument('--after300',type=Path,default=ROOT/'outputs/parent_region_av/history80_vs_dense_chess300')
    run(p.parse_args())
