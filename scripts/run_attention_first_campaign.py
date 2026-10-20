"""Sequential phases, parallel idle GPU4–6 workers; resumable, source-frozen."""
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
from scripts.evaluate_attention_first_residual_value import save
from scripts.quality_metrics import FIELDS,LABELS,quality_cells

SCENES=('chess','fire','heads','office','pumpkin','redkitchen','stairs')


def worst_scenes(previous,count=3):
    rows=[]
    for scene in SCENES:
        dense=json.loads((previous/'dense'/scene/'result.json').read_text())
        old=json.loads((previous/'region'/scene/'result.json').read_text())
        a=(dense['metrics']['pose']['auc30']-old['metrics']['pose']['auc30'])*100
        f=(dense['metrics']['recon_unposed']['fscore']-old['metrics']['recon_unposed']['fscore'])*100
        # Diagnostic scope only, never policy tuning. Use prior measured losses.
        rows.append(dict(scene=scene,score=max(a/1.,f/3.,0.),auc30_drop_pp=a,f1_drop_pp=f))
    return sorted(rows,key=lambda row:(-row['score'],row['scene']))[:count]


def report(root):
    rows=[];missing=[]
    for scene in SCENES:
        for method in ('dense','attention_first'):
            file=root/'quality'/method/scene/'result.json'
            if file.exists():rows.append(dict(scene=scene,method=method,**json.loads(file.read_text())))
            else:missing.append(f'{scene}/{method}')
    macro=[]
    if not missing:
        for method in ('dense','attention_first'):
            chosen=[r for r in rows if r['method']==method]
            values={key:sum(r['quality']['values'][key] for r in chosen)/7
                if all(r['quality']['values'][key] is not None for r in chosen) else None for key in FIELDS}
            macro.append(dict(method=method,quality=dict(values=values),
                mean_seconds=sum(r['median_seconds'] for r in chosen)/7,
                mean_peak_gib=sum(r['median_peak_gib'] for r in chosen)/7,
                speedup_vs_vggt=1. if method=='dense' else
                    sum(r['median_seconds'] for r in rows if r['method']=='dense')/sum(r['median_seconds'] for r in chosen)))
    save(root/'quality_comparison.json',dict(complete=not missing,missing=missing,macro=macro,
        rows=[{k:r[k] for k in ('scene','method','quality','median_seconds','median_peak_gib')} for r in rows]))
    lines=['# New Attention-first / Dense 7Scenes evaluation','',
        f'Complete: {not missing}; 128 frames by default, 1 warmup + 3 full forwards.', '',
        '| Scene | Method | '+' | '.join(LABELS)+' | Seconds | Peak GiB | Speedup |',
        '|---|---|'+'---:|'*9]
    for row in rows:
        speed=row['quality']['speedup_vs_vggt']
        lines.append('| '+row['scene']+' | '+row['method']+' | '+' | '.join(quality_cells(row['quality']))+
            f" | {row['median_seconds']:.4f} | {row['median_peak_gib']:.4f} | "+(f'{speed:.3f}x' if speed is not None else 'N/A')+' |')
    for row in macro:
        lines.append('| Seven-scene macro | '+row['method']+' | '+' | '.join(quality_cells(row['quality']))+
            f" | {row['mean_seconds']:.4f} | {row['mean_peak_gib']:.4f} | {row['speedup_vs_vggt']:.3f}x |")
    lines+=['','Quality differences are evaluation only; no ratios/thresholds are tuned on 7Scenes GT.',
        'Sparse support is20%, not20% total work. Timing includes gate/Value/routing/history/reset. '
        'A partial subset is not a seven-scene macro. Two-frame diagnostics and hooked output comparisons are not benchmark timings.']
    (root/'quality_comparison.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    lock=Lock();manifest_path=root/'manifest.json'
    core=['scripts/run_attention_first_campaign.py','scripts/evaluate_attention_first_residual_value.py',
        'scripts/diagnose_attention_first.py','scripts/quality_metrics.py','saf3r/patch/attention_first_residual_value.py',
        'saf3r/triton/attention_first_selection.py','saf3r/patch/av_stage_ablation.py',
        'saf3r/patch/fused_shared_av.py','saf3r/patch/value_region_av.py','saf3r/patch/parent_region_selection.py',
        'saf3r/patch/value_clustering_gpu.py','saf3r/patch/region_history_attention.py','saf3r/patch/value_evolving_attention.py',
        'saf3r/triton/fused_shared_av.py','saf3r/triton/shared_av_routing.py','saf3r/triton/value_region_selection.py',
        'evaluation/evaluate_da3bench.py','evaluation/benchmarks/da3/da3bench_utils.py',
        'evaluation/benchmarks/da3/bench/datasets/sevenscenes.py','configs/evaluation/attention_first_residual_value.json']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in core}
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['source_sha256']!=hashes:raise ValueError('Source changed; use a new campaign directory')
        if manifest['frames']!=args.frames:raise ValueError('Frame count changed')
    else:
        worst=worst_scenes(args.previous)
        manifest=dict(created_unix=time.time(),frames=args.frames,gpus=args.gpus,source_sha256=hashes,
            selected=worst,selection='Prior old-vs-Dense max(AUC30 loss /1pp, F1 loss /3pp,0), no parameter fitting',
            old='Manual alpha=[0]+[.2]*9+[.8]*7+[.2]*7;20% retention;history80%;whole Parent;original executor',
            new_config=json.loads((ROOT/'configs/evaluation/attention_first_residual_value.json').read_text()),
            phases=['7Scenes Dense/new quality','two-frame masks +128-frame full output deviations','old/new shallow/middle/deep ablations'],
            ablation='Sparse state maintained in all layers; original Dense outputs in inactive layers; all auxiliary costs included')
        save(manifest_path,manifest)
        for name in core:
            p=root/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    jobs={};failures=[];phase='dependencies'
    save(root/'controller.json',dict(pid=os.getpid(),started_unix=time.time(),gpus=args.gpus))
    save(root/'status.json',dict(stage='running',phase=phase,jobs=jobs,failures=failures))
    def update(name,**data):
        with lock:
            jobs.setdefault(name,{}).update(data)
            save(root/'status.json',dict(stage='running',phase=phase,jobs=jobs,failures=failures))
    def execute(command,log,env):
        for name,digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Source changed: {name}')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def ready(gpu):
        if args.after:
            file=args.after/'status.json'
            if not file.exists() or json.loads(file.read_text()).get('stage')!='complete':return False
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=60).strip())
        return used<=512
    def parallel(specs,callback):
        pending=list(specs)
        def worker(gpu):
            while True:
                with lock:
                    if not pending:return
                if not ready(gpu):time.sleep(15);continue
                with lock:
                    if not pending:return
                    spec=pending.pop(0)
                name='/'.join(spec) if isinstance(spec,tuple) else spec
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),
                    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                try:
                    update(name,stage='running',gpu=gpu,started_unix=time.time())
                    callback(spec,env)
                    update(name,stage='complete',finished_unix=time.time())
                except Exception as exc:
                    with lock:failures.append(dict(job=name,error=str(exc)))
                    update(name,stage='failed',error=str(exc))
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(worker,gpu) for gpu in args.gpus]
            for future in futures:future.result()
    phase='quality'
    def quality(scene,env):
        for method in ('dense','attention_first'):
            result=root/'quality'/method/scene/'result.json'
            if result.exists():continue
            command=['scripts/evaluate_attention_first_residual_value.py','--output-dir',str(root/'quality'),
                '--mode',method,'--scene',scene,'--frames',str(args.frames)]
            if method=='attention_first':command+=['--baseline-result',str(root/'quality/dense'/scene/'result.json')]
            execute(command,root/'logs'/f'quality_{scene}_{method}.log',env)
        with lock:report(root)
    parallel(SCENES,quality)
    report(root)
    # Explicit phase barriers: never begin expensive diagnostics before all
    # seven scenes have been attempted. Individual failures remain visible.
    selected=[row['scene'] for row in manifest['selected']]
    phase='diagnostics'
    def diagnostic(scene,env):
        for action in ('visualize','deviation'):
            if (root/'diagnostics'/action/scene/'result.json').exists():continue
            execute(['scripts/diagnose_attention_first.py','--output-dir',str(root/'diagnostics'),
                '--scene',scene,'--action',action,'--frames',str(args.frames)],
                root/'logs'/f'{action}_{scene}.log',env)
    parallel(selected,diagnostic)
    phase='ablations'
    def ablation(spec,env):
        scene,method,stage=spec
        folder=root/'ablations'/stage
        if (folder/method/scene/'result.json').exists():return
        # A same-worker Dense timing for this scene/stage makes speed explicit;
        # no diagnostic-hook timing is used as a benchmark baseline.
        dense_folder=root/'ablation_dense'/scene/stage/method
        baseline=dense_folder/'dense'/scene/'result.json'
        if not baseline.exists():
            execute(['scripts/evaluate_attention_first_residual_value.py','--output-dir',str(dense_folder),
                '--mode','dense','--scene',scene,'--frames',str(args.frames)],root/'logs'/f'ablation_{scene}_{method}_{stage}_dense.log',env)
        execute(['scripts/evaluate_attention_first_residual_value.py','--output-dir',str(folder),
            '--mode',method,'--scene',scene,'--frames',str(args.frames),'--active-stage',stage,
            '--baseline-result',str(baseline)],root/'logs'/f'ablation_{scene}_{method}_{stage}.log',env)
    parallel([(scene,method,stage) for scene in selected for method in ('old','attention_first')
        for stage in ('shallow','middle','deep')],ablation)
    ablation_rows=[]
    for scene in selected:
        for method in ('old','attention_first'):
            for stage in ('shallow','middle','deep'):
                file=root/'ablations'/stage/method/scene/'result.json'
                if file.exists():ablation_rows.append(dict(scene=scene,method=method,stage=stage,**json.loads(file.read_text())))
    save(root/'ablation_comparison.json',dict(rows=ablation_rows,expected=18,complete=len(ablation_rows)==18))
    lines=['| Scene | Scheme | Sparse stage | '+' | '.join(LABELS)+' | Seconds incl auxiliary | Speedup |',
           '|---|---|---|'+'---:|'*8]
    for row in ablation_rows:
        lines.append('| '+row['scene']+' | '+row['method']+' | '+row['stage']+' | '+' | '.join(quality_cells(row['quality']))+
            f" | {row['median_seconds']:.4f} | {row['quality']['speedup_vs_vggt']:.3f}x |")
    (root/'ablation_comparison.md').write_text('\n'.join(lines)+'\n')
    save(root/'status.json',dict(stage='failed' if failures else 'complete',phase='finished',jobs=jobs,
        failures=failures,finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--frames',type=int,default=128);p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6])
    p.add_argument('--previous',type=Path,default=ROOT/'outputs/parent_region_av/three_method_quality_7scenes128')
    p.add_argument('--after',type=Path,default=ROOT/'outputs/parent_region_av/dense_teacher_ratios_128')
    args=p.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or set(args.gpus)-{4,5,6}:raise ValueError('Use distinct GPU4–6')
    run(args)
