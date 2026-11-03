"""Summarize the history-fix campaign and preserve historical controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

METRICS = ['auc03', 'auc30', 'acc', 'comp', 'cd', 'f1']
HIGHER = {'auc03', 'auc30', 'f1'}
VARIANTS = ['single_probe_repaired', 'full_repaired']
METHODS = ['dense', 'saf3r', 'grid_local', 'previous_optimized_v8', 'old_single_probe', 'old_full', *VARIANTS]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def csvwrite(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def table(rows, keys):
    def cell(key, value):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            return f'{100*value:.4f}%' if key in HIGHER else f'{value:.6f}'
        return str(value)
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---']*len(keys)) + ' |'] +
                     ['| ' + ' | '.join(cell(k, row.get(k)) for k in keys) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--historical-outputs', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((out/'manifest.json').read_text())
    reference_index = out/'references/index.json'
    if args.historical_outputs:
        historical = args.historical_outputs.resolve()
        references = []
        prior = list(csv.DictReader((historical/'grid_local_final_20261102/per_scene.csv').open()))
        for row in prior:
            if row['method'] in {'dense', 'saf3r', 'grid_local'}:
                references.append(dict(dataset=row['dataset'], scene=row['scene'], method=row['method'], original=row['source']))
        for dataset, scenes in manifest['scenes'].items():
            for scene in scenes:
                for method, source in [
                    ('previous_optimized_v8', historical/f'efficiency_quality_v8_20261103/{dataset}/{scene}/efficiency/result.json'),
                    ('old_single_probe', historical/f'light_reuse_quality_v2_20261103/single_probe/{dataset}/{scene}/result.json'),
                    ('old_full', historical/f'light_reuse_quality_v2_20261103/full/{dataset}/{scene}/result.json')]:
                    references.append(dict(dataset=dataset, scene=scene, method=method, original=str(source)))
        for entry in references:
            path = out/f"references/{entry['method']}/{entry['dataset']}/{entry['scene']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry['original'], path)
            entry['packaged'] = str(path.relative_to(out))
            entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(reference_index, references)
    references = json.loads(reference_index.read_text())
    rows, audit, monitors, layer_rows, missing = [], [], [], [], []
    reference_data = {}
    for entry in references:
        path = out/entry['packaged']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        raw = json.loads(path.read_text())
        assert raw['quality']['complete'], str(path)
        reference_data[entry['method'], entry['dataset'], entry['scene']] = raw
        rows.append(dict(dataset=entry['dataset'], scene=entry['scene'], method=entry['method'],
                         frames=raw['frames'], **raw['quality']['values'], quality_complete=True, reused=True,
                         timing_batch=entry['original'], seconds=raw['timing']['median_seconds'],
                         peak_allocated_gib=max(x['peak_allocated_gib'] for x in raw['timing']['samples']),
                         dense_speedup=1.0 if entry['method']=='dense' else None,
                         dense_reason='Dense identity' if entry['method']=='dense' else 'Historical batch; not contemporaneous Dense timing'))
    for variant in VARIANTS:
        for dataset, scenes in manifest['scenes'].items():
            for scene in scenes:
                folder = out/f'quality/{variant}/{dataset}/{scene}'
                path = folder/'result.json'
                if not path.exists():
                    missing.append(str(path.relative_to(out)))
                    continue
                raw = json.loads(path.read_text())
                dense = reference_data['dense', dataset, scene]
                layers = json.loads((folder/'layers.json').read_text())
                monitor_path=out/f'monitors/{variant}_{dataset}_{scene}.json'
                if not monitor_path.exists():
                    missing.append(str(monitor_path.relative_to(out)))
                    continue
                monitor = json.loads(monitor_path.read_text())
                config = dict(manifest['configs'][variant])
                config.setdefault('budget_repair', False)
                checks = dict(quality_complete=raw['quality']['complete'], input_paths=raw['input']['image_paths']==dense['input']['image_paths'],
                              frame_indices=raw['input']['frame_indices']==dense['input']['frame_indices'],
                              input_shape=raw['input']['shape']==dense['input']['shape'],
                              checkpoint=Path(raw['input']['checkpoint']).resolve()==Path(dense['input']['checkpoint']).resolve(),
                              protocol=raw['quality_protocol']==dense['quality_protocol'],
                              config=raw['config']==config, layers=len(layers)==24,
                              budget=all(x['counts_mean']<=x['patch_budget'] for x in layers))
                audit.append(dict(variant=variant, dataset=dataset, scene=scene, checks=checks, passed=all(checks.values())))
                good_monitor = not monitor['auxiliary_cuda_context'] and not monitor['foreign_gpu_pids'] and not monitor['same_gpu_overlap_detected'] and monitor['returncode']==0 and not monitor['aborted'] and monitor['max_simultaneous_gpu_applications']==1
                monitors.append(dict(label=f'{variant}_{dataset}_{scene}', passed=good_monitor, **monitor))
                for layer in layers:
                    layer_rows.append(dict(method=variant, dataset=dataset, scene=scene, **{k:v for k,v in layer.items() if not isinstance(v, dict)}))
                rows.append(dict(dataset=dataset, scene=scene, method=variant, frames=raw['frames'], **raw['quality']['values'],
                                 quality_complete=raw['quality']['complete'], reused=False, timing_batch='gpt6_history_fix_eval_20261103',
                                 seconds=raw['timing']['median_seconds'],
                                 peak_allocated_gib=max(x['peak_allocated_gib'] for x in raw['timing']['samples']),
                                 dense_speedup=None, dense_reason='One-sample quality forward; no paired Dense timing'))
    macros=[]
    for dataset, scenes in manifest['scenes'].items():
        for method in METHODS:
            group=[r for r in rows if r['dataset']==dataset and r['method']==method]
            complete=len(group)==len(scenes) and all(x['quality_complete'] for x in group)
            macros.append(dict(dataset=dataset, method=method, scenes=len(group), complete=complete,
                               **{k:statistics.mean(x[k] for x in group) if complete else None for k in METRICS},
                               mean_single_sample_seconds=statistics.mean(x['seconds'] for x in group) if complete else None,
                               peak_allocated_gib=max((x['peak_allocated_gib'] for x in group), default=None),
                               dense_speedup=1.0 if method=='dense' else None))
    changes=[]
    for row in macros:
        if row['method'] not in VARIANTS or not row['complete']:
            continue
        for reference in ['dense','grid_local','previous_optimized_v8', 'old_single_probe' if row['method']=='single_probe_repaired' else 'old_full']:
            baseline=next(x for x in macros if x['dataset']==row['dataset'] and x['method']==reference)
            delta={k:100*(row[k]/baseline[k]-1) for k in METRICS}
            changes.append(dict(dataset=row['dataset'], method=row['method'], reference=reference,
                                **{k+'_relative_percent':v for k,v in delta.items()},
                                **{k+'_percentage_points':100*(row[k]-baseline[k]) for k in HIGHER},
                                all_six_degrade_at_most_1pct=all(v>=-1 if k in HIGHER else v<=1 for k,v in delta.items())))
    timing, profiles = [], []
    for variant in VARIANTS:
        path=out/f'timing/{variant}/result.json'
        if not path.exists():
            missing.append(str(path.relative_to(out)))
            continue
        raw=json.loads(path.read_text())
        monitor=json.loads((out/f'monitors/timing_{variant}.json').read_text())
        passed=not monitor['auxiliary_cuda_context'] and not monitor['same_gpu_overlap_detected'] and not monitor['foreign_gpu_pids'] and not monitor['aborted'] and monitor['returncode']==0 and monitor['max_simultaneous_gpu_applications']==1
        monitors.append(dict(label='timing_'+variant, passed=passed, **monitor))
        timing.append(dict(method=variant, median_seconds=raw['median_seconds'], samples=[x['seconds'] for x in raw['samples']],
                           cold_seconds=raw['cold']['seconds'],
                           peak_allocated_gib=max(x['peak_allocated_gib'] for x in raw['samples']), dense_speedup=None,
                           dense_reason=manifest['dense_reason'], gpu=monitor['gpu'], exclusive_monitor_passed=passed))
        profile=raw['profile']; times=dict(profile['exclusive_seconds'], unattributed=profile['unattributed_seconds'])
        assert abs(sum(times.values())-profile['wall_seconds'])<1e-8
        for operation, seconds in times.items():
            profiles.append(dict(method=variant, operation=operation, seconds=seconds,
                                 percent=seconds/profile['wall_seconds']*100, profile_wall_seconds=profile['wall_seconds']))
    detailed=None
    for label in ['numerical_validation','detailed_budget_profile']:
        monitor_path=out/f'monitors/{label}.json'
        if not monitor_path.exists():
            missing.append(str(monitor_path.relative_to(out)))
            continue
        monitor=json.loads(monitor_path.read_text())
        passed=not monitor['auxiliary_cuda_context'] and not monitor['same_gpu_overlap_detected'] and not monitor['foreign_gpu_pids'] and not monitor['aborted'] and monitor['returncode']==0 and monitor['max_simultaneous_gpu_applications']==1
        monitors.append(dict(label=label,passed=passed,**monitor))
    if (out/'detailed_budget_profile/result.json').exists():
        detailed=json.loads((out/'detailed_budget_profile/result.json').read_text())
        assert abs(sum(detailed['exclusive_seconds'].values())-detailed['wall_seconds'])<1e-8
    else:
        missing.append('detailed_budget_profile/result.json')
    hashes=json.loads((out/'source_sha256.json').read_text())
    unchanged=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    retention=[]
    for variant in VARIANTS:
        for dataset, scenes in manifest['scenes'].items():
            group=[x for x in layer_rows if x['method']==variant and x['dataset']==dataset]
            if len(group)==24*len(scenes):
                retention.append(dict(method=variant,dataset=dataset,
                    global_attention_retained_fraction=statistics.mean(x['actual_unique_density'] for x in group),
                    global_attention_sparse_fraction=1-statistics.mean(x['actual_unique_density'] for x in group),
                    mean_patch_budget_fill=statistics.mean(x['counts_mean']/x['patch_budget'] for x in group),
                    scope='Equal scene and layer averages of global attention edge counts including special keys; not whole-model FLOPs'))
    complete=not missing and len(audit)==36 and all(x['passed'] for x in audit) and len(monitors)==40 and all(x['passed'] for x in monitors) and unchanged
    quality_gate={variant:all(x['all_six_degrade_at_most_1pct'] for x in changes if x['method']==variant and x['reference']=='grid_local')
                  and len([x for x in changes if x['method']==variant and x['reference']=='grid_local'])==2 for variant in VARIANTS}
    summary=dict(complete=complete, source_commit=manifest['source_commit'], macro=macros, timing=timing, profiles=profiles,
                 relative_changes=changes, protocol_audit=audit, monitor_audit=monitors, source_unchanged=unchanged, missing=missing,
                 quality_gate_vs_grid_local=quality_gate, detailed_budget_profile=detailed, retention=retention, controls_executed=0, dense_speedup=None, dense_reason=manifest['dense_reason'])
    save(out/'summary.json', summary)
    save(out/'retention_summary.json', retention)
    for name, data in [('per_scene.csv',rows),('macro.csv',macros),('relative_changes.csv',changes),('timing.csv',timing),('profile.csv',profiles),('per_layer.csv',layer_rows)]:
        csvwrite(out/name,data)
    text=['# Historical Rep / budget repair: GPU evaluation','',
          f"Requested remote branch: fix/gpt6-single-probe-history-20261103; tested source commit `{manifest['source_commit']}`. Complete: **{complete}**. No model formula/configuration was changed during evaluation. The only evaluator change archives the newly added repair module in each source manifest.",'',
          'Evaluation completeness is separate from method acceptance. All-six1% relative-degradation check versus grid_local across both datasets: '+str(quality_gate)+'. A recovery from the old collapsed result is not proof of quality equivalence.','',
          'Only the two changed repaired methods ran. Dense, SAF3R, grid_local, previous optimized v8 and old single/full quality are reused historical controls (original batch paths and raw results preserved in references/). No paper scores are mixed in. SAF3R is lazy first-correspondence-head Top4, ETH3D profile, not paper DINO feature Top-K.','',
          '7Scenes: all seven scenes,128 uniform frames. ETH3D: all11 scenes,14–76 uniform frames. Same checkpoint, exact image list/frame indices, preprocessing and pose/unposed reconstruction protocol audited per scene. AUC/F1 are percentages below; ACC/Comp/CD are unsquared metres, CD=(ACC+Comp)/2. JSON/CSV AUC/F1 values are fractions. Reconstruction threshold/voxel:7Scenes0.05m/0.0078125m; ETH3D0.25m/0.0390625m. Equal-scene macro averages.','',
          table(macros,['dataset','method','scenes','complete',*METRICS]),'',
          '## Actual global attention retention','',
          'The20% cap, Drop/Rep/History/Exact states and original clustering configuration are retained. This is separate from the earlier layered no-Drop/Rep-always method. Retention below counts global attention edges including special keys, averaged equally over scenes and layers; it does not describe whole-model FLOPs.','',
          table(retention,['dataset','method','global_attention_retained_fraction','global_attention_sparse_fraction','mean_patch_budget_fill']),'',
          '## Full-forward runtime and memory','',
          'Timing: fixed 7Scenes heads128, RTX5090 D GPU7, one warmup and median of three complete forwards. Includes reset, initialization within forward, clustering, routing, budget repair sorting, history, reordering and output heads; excludes model/image loading and GT evaluation. All GPU4–7 jobs have CPU-only locking/monitoring; one CUDA process per physical GPU, held until process exit. Quality-run one-sample timings are retained separately and are not benchmark medians. Dense speedup is N/A: unchanged Dense was not rerun. Dense identity is1.00x. Historical times from other batches are not paired speedups.','',
          table(timing,['method','median_seconds','cold_seconds','peak_allocated_gib','gpu','dense_speedup','exclusive_monitor_passed']),'',
          '## Disjoint stage timing','',
          'Independent event profile after the three benchmark samples; insertion overhead makes its wall time different from the warmed median. Percentages use the profile wall time. Budget includes both the16-step solver and the new two-pass greedy repair/sort. Unattributed includes the other model layers, CPU launch gaps and uninstrumented work; it is not all mask selection. History producer is charged once.','',
          table(profiles,['method','operation','seconds','percent','profile_wall_seconds']),'',
          'A separate diagnostic splits solver/repair with temporary function event wrappers, restored after profiling; model math and files are unchanged. Nested children replace their budget parent in the disjoint sum. This independent sample must not be added to the earlier profile table. Detailed raw result: detailed_budget_profile/result.json.','',
          table([dict(operation=k,seconds=v) for k,v in (detailed['exclusive_seconds'] if detailed else {}).items() if k.startswith('detail_budget')],['operation','seconds']),'',
          '## Recovery comparisons','',
          'relative_changes.csv separates changes against old collapsed variants, previous optimized v8, grid_local and matched local Dense. The recorded1% relative-degradation check is descriptive; passing it does not prove mathematical equivalence. No parameter was tuned against test truth.','']
    for change in changes:
        text.append(f"- {change['dataset']} / {change['method']} vs {change['reference']}: all-six degradation within1% = {change['all_six_degrade_at_most_1pct']}; AUC@3 {change['auc03_percentage_points']:+.4f} pp, AUC@30 {change['auc30_percentage_points']:+.4f} pp, CD {change['cd_relative_percent']:+.3f}%, F1 {change['f1_percentage_points']:+.4f} pp.")
    text+=['','Per-scene raw metrics and six-metric Markdown tables: quality/. Layer counts, state usage and actual global attention density: per_layer.csv. Source snapshot and SHA256 are frozen before validation; unchanged: '+str(unchanged)+'. Numerical-only and CPU-only tests report all six quality fields unavailable with concrete reasons. Missing artifacts: '+str(missing)+'.','',
           'Reproduce the summary using `python scripts/report_history_fix_evaluation.py --output-dir outputs/gpt6_history_fix_eval_20261103`. A complete reproduction of GPU runs needs external data/checkpoints and a fresh output directory.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(complete=complete, missing=missing, macro=[x for x in macros if x['method'] in VARIANTS], timing=timing),indent=2))


if __name__=='__main__':
    main()
