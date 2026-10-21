"""One-variable residual-population History80 ablation on frozen dataset inputs.

Reuse the prior exact Dense outputs and SAF3R scores; run only the changed
History80. Never overwrite a prior method or manufacture paired speed timings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_full_global_output_comparison import (save,snapshot,run_method,scene_report,
    history80_config,SOURCE_FILES)
from scripts.run_full_global_output_comparison import report,dataset_summary
from scripts.quality_metrics import quality_record,quality_cells,LABELS
from scripts.evaluate_attention_first_residual_value import write_quality


def worker(args):
    scene=args.output_dir/args.dataset/args.scene
    if scene.exists():raise ValueError('Fresh scene required')
    scene.mkdir(parents=True);hashes=snapshot(scene)
    baseline=args.baseline/args.dataset/args.scene
    for method in ('dense','saf3r'):
        (scene/method).symlink_to((baseline/method).resolve(),target_is_directory=True)
    save(scene/'baseline_reuse.json',dict(source=str(baseline.resolve()),
        reason='Unchanged exact frame/checkpoint/protocol baseline, reused not retimed',
        changed_method='History80 config override: '+str(args.history80_config_path) if args.history80_config_path else
            'History80 residual population logits; history updated from compensated probabilities'))
    try:
        run_method(args,'history80',scene,hashes)
        summary=scene_report(scene)
        if not summary['outputs_complete'] or (not args.smoke and not summary['quality_complete']):
            raise RuntimeError('Incomplete paired output/quality')
        save(scene/'status.json',dict(stage='smoke_complete' if args.smoke else 'complete'))
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
        save(scene/'history80/status.json',dict(stage='failed',error=reason,traceback=traceback.format_exc()))
        write_quality(scene/'history80',quality_record({},reason))
        save(scene/'status.json',dict(stage='failed',error=reason));scene_report(scene)
        raise


def ablation_report(root,baseline):
    """Four-method report with old and compensated History80 side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    current=json.loads((root/'dataset_summary.json').read_text())
    original=json.loads((baseline/'dataset_summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    tuned=bool(manifest.get('parameter_ablation'))
    baseline_label='History80 population baseline' if tuned else 'History80 original'
    current_label='History80 middle tuned' if tuned else 'History80 population'
    lines=['# History80 middle-parameter ablation' if tuned else '# History80 residual-population ablation','',
        'Identical checkpoint,frame lists,preprocessing and20% target.',
        'Only middle parameters changed; see frozen selected config and calibration selection.' if tuned else
            'Only change: Value logit += log(cluster population - real Attention members of this cluster).',
        'Specials and real Attention Keys have unit mass. Weighted same-pass evidence updates history.',
        'Dense/SAF3R/old History80 are reused exact-input baselines,not new timing runs.',
        'All24 global layers are sparse; frame/local unchanged. No Oracle, Dense fallback or test-truth tuning.',
        'Layer error is projected output before residual; both independent rollout and own-input Dense references.',
        'AUC/F1 percent; distances metres; CD=(ACC+Comp)/2 unsquared. Speeds N/A: instrumented.',
        '7Scenes threshold0.05m,voxel0.0078125m; ETH3D threshold0.25m,voxel0.0390625m.']
    for ds,d in current['datasets'].items():
        old=original['datasets'][ds]
        if not d['outputs_complete'] or not d['quality_complete']:
            lines+=['',f'## {ds}: PARTIAL; paired output {len(d["output_scenes"])}/{len(d["expected_scenes"])}',
                'Do not compare partial current macros against complete historical macros.']
            continue
        lines+=['',f'## {ds}: complete {len(d["expected_scenes"])}/{len(d["expected_scenes"])}',
            '', '| Method | '+' | '.join(LABELS)+' | Speedup |','|---|'+'---:|'*7]
        choices=[('Dense',old['quality_macro']['dense']),('SAF3R lazy Top4',old['quality_macro']['saf3r']),
            (baseline_label,old['quality_macro']['history80']),(current_label,d['quality_macro']['history80'])]
        for name,q in choices:lines.append('| '+name+' | '+' | '.join(quality_cells(dict(values=q)))+
            (' | 1.00x reference |' if name=='Dense' else ' | N/A: no paired warmed timing |'))
        for comp in ('rollout_vs_dense','same_input_vs_dense'):
            fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
            series=[]
            for source,method,label in ((old,'saf3r','SAF3R'),(old,'history80',baseline_label),(d,'history80',current_label)):
                rows=sorted([r for r in source['layer_macro'] if r['method']==method and r['position']=='module_output'
                    and r['scope']=='all_queries' and r['comparison']==comp],key=lambda r:r['layer'])
                series.append(rows)
                axes[0].plot([r['layer'] for r in rows],[100*r['relative_l2_frobenius'] for r in rows],marker='.',label=label)
                axes[1].plot([r['layer'] for r in rows],[r['mean_per_query_cosine'] for r in rows],marker='.',label=label)
            for ax in axes:ax.set_xlabel('Global layer');ax.set_xticks(range(0,24,2));ax.grid(alpha=.3);ax.legend()
            axes[0].set_ylabel('Equal-scene mean relative L2 (%)');axes[1].set_ylabel('Mean Query cosine')
            fig.suptitle(f'{ds}: {comp} | {len(d["expected_scenes"])} scenes')
            fig.savefig(root/f'{ds}_population_{comp}.png',dpi=170);plt.close(fig)
            lines+=['',f'### {comp}','', '| Layer | Original error % | Population error % | Original cosine | Population cosine |',
                '|---|---:|---:|---:|---:|']
            for a,b in zip(series[1],series[2]):
                lines.append(f'| {a["layer"]} | {100*a["relative_l2_frobenius"]:.4f} | {100*b["relative_l2_frobenius"]:.4f} | '
                    f'{a["mean_per_query_cosine"]:.6f} | {b["mean_per_query_cosine"]:.6f} |')
    (root/'population_comparison.md').write_text('\n'.join(lines)+'\n')


def campaign(args):
    if args.output_dir.exists():raise ValueError('Fresh campaign root required')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus):raise ValueError('Distinct idle GPUs required')
    usage={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.used',
        '--format=csv,noheader,nounits'],text=True).strip()) for g in args.gpus}
    if any(v>512 for v in usage.values()) and not args.wait_busy_gpus:
        raise ValueError('GPU busy; do not interrupt unrelated work')
    manifest=json.loads((args.baseline/'manifest.json').read_text())
    base_summary=json.loads((args.baseline/'dataset_summary.json').read_text())
    if not base_summary['quality_complete'] or not base_summary['outputs_complete']:raise ValueError('Complete baseline required')
    names=(*SOURCE_FILES,'scripts/run_history80_population_comparison.py','scripts/run_full_global_output_comparison.py')
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    args.output_dir.mkdir(parents=True)
    config=replace(history80_config(),population_compensation=True)
    if args.history80_config_path:
        from saf3r.patch.value_region_av import ValueRegionAVConfig
        config=ValueRegionAVConfig(**json.loads(args.history80_config_path.read_text()))
        if not config.population_compensation:raise ValueError('This ablation keeps population compensation enabled')
        names=(*names,str(args.history80_config_path))
        hashes[str(args.history80_config_path)]=hashlib.sha256(args.history80_config_path.read_bytes()).hexdigest()
    manifest.update(population_compensation=True,history80=asdict(config),parameter_ablation=bool(args.history80_config_path),
        config_override=str(args.history80_config_path) if args.history80_config_path else None,
        baseline_reused=str(args.baseline.resolve()),created_unix=time.time(),gpus=args.gpus,source_sha256=hashes)
    save(args.output_dir/'manifest.json',manifest)
    jobs=[(d,s) for d,scenes in manifest['scenes'].items() for s in scenes]
    states={f'{d}/{s}':dict(stage='queued') for d,s in jobs};lock=Lock()
    def gpu_loop(gpu):
        while True:
            if args.wait_busy_gpus:
                with lock:
                    if not any(v['stage']=='queued' for v in states.values()):return
                used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'],text=True).strip())
                if used>512:
                    time.sleep(5);continue
            with lock:
                key=next((k for k,v in states.items() if v['stage']=='queued'),None)
                if key is None:return
                states[key]=dict(stage='running',gpu=gpu,started_unix=time.time())
                save(args.output_dir/'queue_status.json',states)
            ds,scene=key.split('/')
            try:
                for n,digest in hashes.items():
                    if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+n)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                log=args.output_dir/'logs'/f'{ds}_{scene}.log';log.parent.mkdir(exist_ok=True)
                with log.open('w') as stream:
                    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--dataset',ds,
                        '--scene',scene,'--frames','128','--baseline',str(args.baseline),'--output-dir',str(args.output_dir)]
                    if args.history80_config_path:command+=['--history80-config-path',str(args.history80_config_path)]
                    result=subprocess.run(command,
                        env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'Exit{result.returncode}; see {log}')
                state=dict(stage='complete',gpu=gpu)
            except Exception as exc:state=dict(stage='failed',gpu=gpu,error=str(exc))
            with lock:
                states[key].update(state,finished_unix=time.time());save(args.output_dir/'queue_status.json',states)
                report(args.output_dir);ablation_report(args.output_dir,args.baseline)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in as_completed([pool.submit(gpu_loop,g) for g in args.gpus]):future.result()
    save(args.output_dir/'status.json',dict(complete=all(v['stage']=='complete' for v in states.values()),jobs=states))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=list(range(7)))
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset',choices=['7scenes','eth3d'])
    parser.add_argument('--scene');parser.add_argument('--frames',type=int,default=128)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--history80-config-path',type=Path)
    parser.add_argument('--wait-busy-gpus',action='store_true',help='Claim work only when GPU becomes idle; never interrupt other tasks')
    args=parser.parse_args();args.population_compensation=True
    if args.worker:worker(args)
    else:campaign(args)
