"""Matched-scene interim statistics; never a complete ScanNet-50 macro."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/scannet50_paper_dino64_20261027'
FIELDS=['auc03','auc30','acc','comp','cd','f1']

def main():
    source=json.loads((OUT/'summary.json').read_text())
    by={m:{r['scene']:r for r in source['per_scene'] if r['requested_frames']==300 and r['method']==m and r['complete']} for m in ['dense','saf3r']}
    shared=sorted(set(by['dense'])&set(by['saf3r']))
    timed=[s for s in shared if by['saf3r'][s]['speedup'] is not None]
    stats={m:{k:mean(by[m][s][k] for s in shared) for k in FIELDS} for m in by}
    seconds={m:mean(by[m][s]['seconds'] for s in timed) for m in by}
    peak={m:max(by[m][s]['peak_gib'] for s in shared) for m in by}
    result=dict(scope='Partial 300-frame-cap matched subset only; NOT full ScanNet-50 macro',
        scene_count=len(shared),scenes=shared,six_metric_equal_scene_means=stats,
        uncontended_timing_scene_count=len(timed),timed_scenes=timed,mean_of_scene_warmed_medians_seconds=seconds,
        speedup_vs_dense=seconds['dense']/seconds['saf3r'],peak_allocated_gib=peak,
        quality_exclusions=sorted(set(source_scene for source_scene in json.loads((OUT/'progress.json').read_text())['scenes'])-set(shared)),
        timing_exclusions=sorted(set(shared)-set(timed)))
    (OUT/'matched_partial_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# ScanNet-50 DINO64: partial matched evaluation','',
        f'300-frame cap, {len(shared)} common completed scenes. This is NOT a full 50-scene macro.',
        'Non-lazy DINO per-frame Top64 with max2048 cap; ETH3D profile. Same checkpoint/input per pair. '
        'Uniform valid-pose sampling. No lowered image resolution, frame cap, or K.',
        'Unposed predicted-camera depth unprojection; FastVGGT gauge and bounding-box scale/center alignment. '
        'Confidence>=1; 100k point cap; 5cm voxel and F1 threshold; distances clipped at0.5m. '
        'CD=(ACC+Comp)/2, unsquared metres. Not the official sum-CD.', '',
        '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in by:
        cells=[f"{stats[m][k]*(100 if k in ['auc03','auc30','f1'] else 1):.5f}" for k in FIELDS]
        lines.append('| '+' | '.join([m,*cells])+' |')
    lines+=['',f'Timing uses only {len(timed)} uncontended matched pairs: Dense={seconds["dense"]:.4f}s, '
        f'SAF3R={seconds["saf3r"]:.4f}s, speedup={result["speedup_vs_dense"]:.4f}x. '
        'One warmup and three complete forwards per run. Includes DINO similarity/caching. '
        'Other timings must not be included in this speedup.',
        '', 'Completion: 300 Dense48/50, SAF3R44/50; 500 Dense49/50, SAF3R0/50; 700 both0/50. '
        'Preflight OOM prevented remaining 500-SAF3R/700 tasks from starting. '
        'Six fields for failed/not-started runs are N/A with reasons in failure.json/quality.json/blocked_jobs.json.',
        '', '## Exact matched scenes','```json',json.dumps(shared,indent=2),'```']
    (OUT/'matched_partial_summary.md').write_text('\n'.join(lines)+'\n')
    paired=[by[m][s] for s in shared for m in by]
    with (OUT/'matched_partial_per_scene.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
