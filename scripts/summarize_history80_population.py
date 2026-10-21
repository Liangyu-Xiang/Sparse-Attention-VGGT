"""Finalize complete population ablation tables and explicit prior quality gates."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_full_global_output_comparison import save
from scripts.quality_metrics import LABELS,quality_cells


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(root,baseline):
    current=json.loads((root/'dataset_summary.json').read_text())
    old=json.loads((baseline/'dataset_summary.json').read_text())
    if not current['outputs_complete'] or not current['quality_complete']:
        raise ValueError('Complete datasets required for final summary')
    manifest=json.loads((root/'manifest.json').read_text())
    tuned=bool(manifest.get('parameter_ablation'))
    baseline_history_name='history80_population_baseline' if tuned else 'history80'
    current_name='history80_middle_tuned' if tuned else 'history80_population'
    rows=[];head_rows=[];gates={}
    lines=['# Per-scene middle-parameter comparison' if tuned else '# Per-scene population comparison','',
        'AUC/F1 percent,ACC/Comp/CD metres,CD=(ACC+Comp)/2. Same exact scene input/checkpoint/protocol.',
        'Dense,SAF3R and original History80 reused from the completed baseline; no new speed comparison.',
        'SAF3R repository lazy Top4; all methods cover all24 global layers; frame/local unchanged.',
        '7Scenes threshold0.05m,voxel0.0078125m; ETH3D threshold0.25m,voxel0.0390625m.',
        '', '| Dataset | Scene | Method | '+' | '.join(LABELS)+' | Speedup |',
        '|---|---|---|'+'---:|'*7]
    for ds,names in manifest['scenes'].items():
        macro=current['datasets'][ds]['quality_macro']['history80'];dense=old['datasets'][ds]['quality_macro']['dense']
        a=100*(macro['auc30']-dense['auc30']);f=100*(macro['f1']-dense['f1'])
        gate=dict(auc30_delta_pp=a,f1_delta_pp=f,macro_auc30_pass=a>=-.2,macro_f1_pass=f>=-.5,
            depth_gate='N/A: depth not evaluated',speed_gate='N/A: no paired warmed uninstrumented timings',
            memory_gate='N/A: captured extra Dense outputs; not deployment peak',scene_failures=[])
        for s in names:
            pop=json.loads((root/ds/s/'history80/result.json').read_text())
            data=[]
            for m in ('dense','saf3r','history80'):
                previous=json.loads((baseline/ds/s/m/'result.json').read_text())
                if previous['input']!=pop['input'] or previous['quality_protocol']!=pop['quality_protocol']:
                    raise ValueError(f'Input/protocol mismatch: {ds}/{s}/{m}')
                data.append((baseline_history_name if m=='history80' else m,previous))
            data.append((current_name,pop))
            for m,result in data:
                if not result['quality']['complete']:raise ValueError('Incomplete quality')
                q=result['quality']['values'];speed=1. if m=='dense' else None
                rows.append(dict(dataset=ds,scene=s,method=m,frames=result['frames'],**q,speedup_vs_vggt=speed))
                lines.append('| '+ds+' | '+s+' | '+m+' | '+' | '.join(quality_cells(dict(values=q)))+
                    (' | 1.00x reference |' if m=='dense' else ' | N/A: no paired warmed timing |'))
            q=pop['quality']['values'];d=data[0][1]['quality']['values']
            aa=100*(q['auc30']-d['auc30']);ff=100*(q['f1']-d['f1'])
            if aa< -1 or ff< -3:gate['scene_failures'].append(dict(scene=s,auc30_delta_pp=aa,f1_delta_pp=ff))
        gate['per_scene_pose_recon_pass']=not gate['scene_failures'];gates[ds]=gate
        head_rows.extend(dict(dataset=ds,**r) for r in current['datasets'][ds]['head_macro'])
    write_csv(root/'population_quality_per_scene.csv',rows);write_csv(root/'head_macro.csv',head_rows)
    (root/'population_per_scene.md').write_text('\n'.join(lines)+'\n')
    save(root/'quality_gates.json',dict(definition='Prior gates retained, not tuned against results',datasets=gates))
    if tuned:
        rendered=root/'population_comparison.md'
        if rendered.exists():
            rendered.write_text(rendered.read_text().replace(
                'Original error % | Population error % | Original cosine | Population cosine',
                'Before error % | After error % | Before cosine | After cosine'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();summarize(args.output_dir,args.baseline)
