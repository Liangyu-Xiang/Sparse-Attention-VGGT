"""Non-negotiable quality/speed/memory gates; partial output is never a pass."""
import argparse
import json
from pathlib import Path
from statistics import mean

SCENES=['chess','fire','heads','office','pumpkin','redkitchen','stairs']


def quality_gates(pairs):
    if not pairs:return dict(available=False,passed=False)
    tasks=['pose','depth','recon_unposed']
    if any(any(t not in x['metrics'] for t in tasks) for pair in pairs for x in pair):
        return dict(available=False,passed=False,reason='All three official tasks required')
    macro=[{t:{k:mean(p[j]['metrics'][t][k] for p in pairs) for k in pairs[0][j]['metrics'][t]}
        for t in tasks} for j in [0,1]]
    d,c=macro
    macro_tests=dict(auc30=d['pose']['auc30']-c['pose']['auc30']<=.002+1e-12,
        fscore=d['recon_unposed']['fscore']-c['recon_unposed']['fscore']<=.005+1e-12,
        absrel=c['depth']['Abs Rel']<=d['depth']['Abs Rel']*1.01+1e-12)
    per_scene={}
    for a,b in pairs:
        per_scene[a['scene']]=dict(auc30=a['metrics']['pose']['auc30']-b['metrics']['pose']['auc30']<=.01+1e-12,
            fscore=a['metrics']['recon_unposed']['fscore']-b['metrics']['recon_unposed']['fscore']<=.03+1e-12,
            absrel=b['metrics']['depth']['Abs Rel']<=a['metrics']['depth']['Abs Rel']*1.05+1e-12)
    return dict(available=True,passed=all(macro_tests.values()) and all(all(v.values()) for v in per_scene.values()),
        dense_macro=d,fused_macro=c,macro_tests=macro_tests,per_scene_tests=per_scene)


def report(root,mode='fused',expected=SCENES):
    pairs=[];missing=[]
    for scene in expected:
        paths=[root/m/scene/'result.json' for m in ['dense',mode]]
        if not all(p.exists() for p in paths):missing.append(scene);continue
        a,b=[json.loads(p.read_text()) for p in paths]
        if a['input']!=b['input']:raise ValueError('Paired inputs/checkpoints differ')
        if a['source_sha256']!=b['source_sha256']:raise ValueError('Paired source snapshots differ')
        if a['hardware']!=b['hardware']:raise ValueError('Paired GPU/software environments differ')
        pairs.append((a,b))
    if not pairs:raise ValueError('No complete paired results')
    speedup=mean(a['median_seconds'] for a,b in pairs)/mean(b['median_seconds'] for a,b in pairs)
    memory_ok=all(b['median_peak_gib']<=a['median_peak_gib'] for a,b in pairs)
    warmed=all(x['protocol']['warmup']>=1 and x['protocol']['repeats']>=3 and not x['protocol']['capture'] and not x['config']['capture']
        for pair in pairs for x in pair)
    quality=quality_gates(pairs)
    scope_complete=not missing and sorted(expected)==sorted(SCENES) and all(x['frames']==128 and x['dataset']=='7scenes' for pair in pairs for x in pair)
    speed_ok=speedup>=2 and warmed
    stages={}
    for a,b in pairs:
        stage_sums={}
        for row in b['profile']:stage_sums[row['stage']]=stage_sums.get(row['stage'],0)+row['seconds']
        stages[b['scene']]=stage_sums
    summary=dict(scope_complete=scope_complete,missing=missing,quality=quality,speedup=speedup,
        frames=sorted({x['frames'] for pair in pairs for x in pair}),
        datasets=sorted({x['dataset'] for pair in pairs for x in pair}),
        speed_passed=speed_ok,memory_passed=memory_ok,warmed_protocol_valid=warmed,
        accepted=scope_complete and quality['passed'] and speed_ok and memory_ok,
        stages=stages,per_scene=[dict(scene=a['scene'],dense_seconds=a['median_seconds'],fused_seconds=b['median_seconds'],
            dense_peak_gib=a['median_peak_gib'],fused_peak_gib=b['median_peak_gib'],
            dense_metrics=a['metrics'],fused_metrics=b['metrics']) for a,b in pairs])
    out=root/'report';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Fused full-coverage Value / evolving Attention','',
        'Shared32-query support, manual0/20/80/20%Attention schedule, all24global layers; '
        'actual per-layer retention is recorded in configs. No Dense seed/fallback. '
        'Primary warmed timings include all state/reset/selection costs; independent profiles are not primary samples.','',
        f"Full acceptance: **{summary['accepted']}**. Complete seven-scene128-frame scope: {scope_complete}. "
        f"Quality gate: {quality['passed']}; speed gate: {speed_ok}; memory gate: {memory_ok}.",
        '', '| Scene | Dense(s) | Fused(s) | Dense GiB | Fused GiB |','|---|---:|---:|---:|---:|']
    for a,b in pairs:lines.append(f"| {a['scene']} | {a['median_seconds']:.4f} | {b['median_seconds']:.4f} | {a['median_peak_gib']:.3f} | {b['median_peak_gib']:.3f} |")
    lines+=['',f'Paired mean-time ratio Dense/fused: {speedup:.4f}x. Missing: {missing}.',
        'An unavailable quality gate is NOT a pass. Small-input pose checks are not dataset accuracy. '
        'Terminal selection performs extra probe QK; its winning final QK is currently repeated and counted. '
        'Value local-source groups and spatial probe pooling are a new explicit approximation, not bitwise replay of the old branch.','',
        '## Independent profile (seconds; inclusive selection includes terminal)','',
        '```json',json.dumps(stages,indent=2),'```']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--mode',choices=['fused','value','attention'],default='fused');p.add_argument('--scenes',nargs='+',default=SCENES)
    a=p.parse_args();report(a.directory,a.mode,a.scenes)
