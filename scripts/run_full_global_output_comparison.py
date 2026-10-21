"""Freeze and queue all7Scenes/all11 ETH3D full-global output comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_full_global_output_comparison import METHODS,SOURCE_FILES,save,history80_config
from scripts.quality_metrics import quality_cells,LABELS


def mean_optional(values):
    return sum(values)/len(values) if values and all(v is not None and math.isfinite(v) for v in values) else None


def dataset_summary(expected,scenes):
    """Equal-scene means on identical paired scenes, never Token pooling."""
    paired={s:x for s,x in scenes.items() if x['outputs_complete']}
    quality_paired={s:x for s,x in scenes.items() if x['quality_complete']}
    result=dict(expected_scenes=expected,output_scenes=list(paired),quality_scenes=list(quality_paired),
        outputs_complete=len(paired)==len(expected),quality_complete=len(quality_paired)==len(expected),
        missing_outputs=[s for s in expected if s not in paired],
        missing_quality=[s for s in expected if s not in quality_paired],
        macro_definition='Arithmetic mean of per-scene metrics, equal scene weight; partial means explicitly subset')
    layer_rows=[];head_rows=[]
    for method in METHODS:
        for layer in range(24):
            for position in ('module_output','preprojection'):
                for comparison in ('rollout_vs_dense','same_input_vs_dense'):
                    if method=='dense' and comparison=='same_input_vs_dense':continue
                    for scope in ('all_queries','patch_queries','special_queries'):
                        ms=[x['methods'][method]['layers'][layer][position][comparison][scope] for x in paired.values()]
                        if not ms:continue
                        layer_rows.append(dict(method=method,layer=layer,position=position,comparison=comparison,
                            scope=scope,scenes=len(ms),relative_l2_frobenius=mean_optional([v['relative_l2_frobenius'] for v in ms]),
                            mean_per_query_cosine=mean_optional([v['mean_per_query_cosine'] for v in ms]),
                            flattened_cosine=mean_optional([v['flattened_cosine'] for v in ms])))
            for head in range(16):
                for comparison in ('rollout_vs_dense','same_input_vs_dense'):
                    if method=='dense' and comparison=='same_input_vs_dense':continue
                    ms=[x['methods'][method]['layers'][layer]['heads'][head][comparison]['all_queries'] for x in paired.values()]
                    if ms:head_rows.append(dict(method=method,layer=layer,head=head,comparison=comparison,
                        scenes=len(ms),relative_l2_frobenius=mean_optional([v['relative_l2_frobenius'] for v in ms]),
                        mean_per_query_cosine=mean_optional([v['mean_per_query_cosine'] for v in ms])))
    result['layer_macro']=layer_rows;result['head_macro']=head_rows;result['quality_macro']={}
    for method in METHODS:
        qs=[x['quality'][method]['quality']['values'] for x in quality_paired.values()]
        if qs:result['quality_macro'][method]={k:mean_optional([q[k] for q in qs]) for k in qs[0]}
    return result


def report(root):
    manifest=json.loads((root/'manifest.json').read_text());datasets={};allrows=[];per_scene=[]
    for dataset,expected in manifest['scenes'].items():
        scenes={}
        for scene in expected:
            p=root/dataset/scene/'comparison.json'
            if p.exists():
                x=json.loads(p.read_text());scenes[scene]=x
                for method,q in x['quality'].items():
                    per_scene.append(dict(dataset=dataset,scene=scene,method=method,frames=q['frames'],**q['quality']['values']))
        d=dataset_summary(expected,scenes);datasets[dataset]=d
        allrows += [dict(dataset=dataset,**r) for r in d['layer_macro']]
    complete=all(x['outputs_complete'] for x in datasets.values())
    qcomplete=all(x['quality_complete'] for x in datasets.values())
    save(root/'dataset_summary.json',dict(outputs_complete=complete,quality_complete=qcomplete,datasets=datasets))
    for name,rows in [('layer_macro.csv',allrows),('quality_per_scene.csv',per_scene)]:
        if rows:
            with (root/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Full SAF3R vs History80, all24 global layers','',
        f'All output comparisons complete: {complete}; all six-metric quality complete: {qcomplete}.',
        'Repository default SAF3R lazy first-correspondence-head Top4, NOT lazy Top128 or a verified paper-DINO reproduction.',
        'History80 frozen config: Attention ratios='+str(manifest['history80']['attention_ratios'])+
            '; incoming history='+str(manifest['history80'].get('incoming_history_fractions') or manifest['history80']['history_fraction'])+
            ';20% target,whole Parents; population correction='+str(manifest.get('population_compensation',False))+'.',
        'Every global0–23 layer uses its method; frame/local attention unchanged; no Dense seed/fallback or prefix.',
        'Primary layer output is after output projection but before residual. Per-head raw AV is separately available.',
        'Cosine averaged per Query; relative L2=||sparse-Dense||_F/||Dense||_F.',
        'Rollout reference includes prior-layer drift. Same-input reference is each method\'s own current input, not one common teacher input.',
        'Equal-scene means; partial paired subsets explicitly labelled. Datasets are not combined.',
        'Instrumented single forwards include capture/comparison/extra Dense costs, not valid speed measurements.']
    for dataset,d in datasets.items():
        lines+=['',f'## {dataset}: output {len(d["output_scenes"])}/{len(d["expected_scenes"])}; quality {len(d["quality_scenes"])}/{len(d["expected_scenes"])}','',
            'Full dataset macro.' if d['outputs_complete'] else 'PARTIAL paired-scene subset, NOT a complete dataset macro.',
            'Missing outputs: '+str(d['missing_outputs']), 'Missing quality: '+str(d['missing_quality']),
            '', '| Method | '+' | '.join(LABELS)+' | Speedup |','|---|'+'---:|'*7]
        for method,q in d['quality_macro'].items():
            lines.append('| '+method+' | '+' | '.join(quality_cells(dict(values=q)))+' | '+('1.00x reference' if method=='dense' else 'N/A: instrumented diagnosis')+' |')
        for comp in ('rollout_vs_dense','same_input_vs_dense'):
            lines+=['',f'### Module outputs: {comp}','', '| Layer | SAF3R error % | SAF3R cosine | History80 error % | History80 cosine |',
                '|---|---:|---:|---:|---:|']
            for layer in range(24):
                cells=[]
                for method in ('saf3r','history80'):
                    v=next((r for r in d['layer_macro'] if r['method']==method and r['layer']==layer and
                        r['position']=='module_output' and r['comparison']==comp and r['scope']=='all_queries'),None)
                    cells += [f"{100*v['relative_l2_frobenius']:.4f}",f"{v['mean_per_query_cosine']:.6f}"] if v else ['N/A','N/A']
                lines.append('| '+str(layer)+' | '+' | '.join(cells)+' |')
        if d['output_scenes']:plot(root,dataset,d)
    lines+=['','7Scenes reconstruction: threshold0.05m,downsample0.0078125m. ETH3D: threshold0.25m,downsample0.0390625m.',
        'All reconstruction distances metres,CD=(ACC+Comp)/2 unsquared; AUC/F1 percent in tables, fraction in JSON.',
        'Default7Scenes sequences, not every original video; ETH3D11 configured scenes, actual14–76 valid images each.',
        'Per-scene exact frame lists/checkpoint/input geometry and frozen source snapshots are preserved.']
    (root/'dataset_report.md').write_text('\n'.join(lines)+'\n')


def plot(root,dataset,d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for comp in ('rollout_vs_dense','same_input_vs_dense'):
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
        for method in ('saf3r','history80'):
            rows=sorted([r for r in d['layer_macro'] if r['method']==method and r['position']=='module_output'
                         and r['scope']=='all_queries' and r['comparison']==comp],key=lambda r:r['layer'])
            axes[0].plot([r['layer'] for r in rows],[100*r['relative_l2_frobenius'] for r in rows],marker='.',label=method)
            axes[1].plot([r['layer'] for r in rows],[r['mean_per_query_cosine'] for r in rows],marker='.',label=method)
        for ax in axes:ax.set_xlabel('Global layer (zero based)');ax.set_xticks(range(0,24,2));ax.grid(alpha=.3);ax.legend()
        axes[0].set_ylabel('Equal-scene mean relative L2 error (%)');axes[1].set_ylabel('Equal-scene mean Query cosine')
        fig.suptitle(f'{dataset} | {comp} | paired scenes {len(d["output_scenes"])}/{len(d["expected_scenes"])} | projected attention output')
        fig.savefig(root/f'{dataset}_{comp}.png',dpi=170);plt.close(fig)


def run(args):
    import yaml
    if args.report_only:report(args.output_dir);return
    if args.output_dir.exists():raise ValueError('Choose fresh campaign root')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus):raise ValueError('Distinct GPUs required')
    usage={g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.used',
        '--format=csv,noheader,nounits'],text=True).strip()) for g in args.gpus}
    if any(v>512 for v in usage.values()):raise ValueError('Requested GPU is not idle; do not interrupt unrelated work')
    cfg=yaml.safe_load((ROOT/'configs/evaluation/default_datasets.yaml').read_text())
    scenes={d:cfg['datasets'][d]['scenes'] for d in ('7scenes','eth3d')}
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    inventory={}
    for ds,names in scenes.items():
        dataset=MV_REGISTRY.get(ds)();inventory[ds]={}
        for name in names:
            indices,data=sample_frames(dataset.get_data(name),name,128,sample_mode='uniform',seed=0)
            if not data.image_files or any(not Path(p).exists() for p in data.image_files):raise ValueError('Missing input')
            if not Path(data.aux.gt_mesh_path).exists():raise ValueError('Missing ground-truth mesh')
            inventory[ds][name]=dict(frames=len(data.image_files),frame_indices=indices,image_paths=data.image_files,
                quality_threshold_m=dataset.eval_threshold,quality_downsample_m=dataset.down_sample)
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in (*SOURCE_FILES,'scripts/run_full_global_output_comparison.py')}
    args.output_dir.mkdir(parents=True)
    save(args.output_dir/'manifest.json',dict(created_unix=time.time(),scenes=scenes,inventory=inventory,gpus=args.gpus,
        max_frames=128,seed=0,methods=METHODS,quality_tasks=['pose','recon_unposed'],history80=history80_config().__dict__,
        saf3r='Repository complete lazy Top4; not historical Top128 variant',source_sha256=hashes,
        requested_all_global_layers=True,diagnostic_not_speed_benchmark=True))
    jobs=[(d,s) for d,names in scenes.items() for s in names]
    states={f'{d}/{s}':dict(stage='queued') for d,s in jobs};lock=Lock()
    def update(key,**kw):
        with lock:
            states[key].update(kw);save(args.output_dir/'queue_status.json',dict(jobs=states,updated_unix=time.time()))
    def worker(gpu):
        while True:
            with lock:
                available=next((key for key,v in states.items() if v['stage']=='queued'),None)
                if available is None:return
                states[available]=dict(stage='running',gpu=gpu,started_unix=time.time())
                save(args.output_dir/'queue_status.json',dict(jobs=states,updated_unix=time.time()))
            ds,scene=available.split('/')
            try:
                for name,digest in hashes.items():
                    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+name)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                log=args.output_dir/'logs'/f'{ds}_{scene}.log';log.parent.mkdir(exist_ok=True)
                cmd=[sys.executable,str(ROOT/'scripts/evaluate_full_global_output_comparison.py'),
                    '--dataset',ds,'--scene',scene,'--frames','128','--output-dir',str(args.output_dir)]
                with log.open('w') as stream:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if p.returncode:raise RuntimeError(f'Exit{p.returncode}; see {log}')
                summary=json.loads((args.output_dir/ds/scene/'comparison.json').read_text())
                update(available,stage='complete' if summary['quality_complete'] else 'outputs_complete_quality_incomplete',finished_unix=time.time())
            except Exception as exc:update(available,stage='failed',error=str(exc),finished_unix=time.time())
            with lock:report(args.output_dir)
    save(args.output_dir/'queue_status.json',dict(jobs=states,updated_unix=time.time()))
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,g) for g in args.gpus]
        for future in as_completed(futures):future.result()
    report(args.output_dir)
    save(args.output_dir/'campaign_status.json',dict(stage='complete' if all(v['stage']=='complete' for v in states.values()) else 'finished_with_incomplete_runs',jobs=states))


if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'evaluation'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(7)))
    p.add_argument('--report-only',action='store_true')
    run(p.parse_args())
