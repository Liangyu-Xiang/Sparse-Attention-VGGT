"""Audit complete matched six-metric quality and original/grouped SAF3R speed."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def quality_table(rows):
    text=['| Scope | Method | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[f'{r[k]*100:.3f}%' if k in ('auc03','auc30','f1') else f'{r[k]:.6f}' for k in FIELDS]
        text.append('| '+' | '.join([r['scope'],r['method']]+values)+' |')
    return '\n'.join(text)

def expected_pairs(groups,frames,patches):
    per=patches+5;n=frames*per;special={f*per+i for f in range(frames) for i in range(5)};result=0;heads=0
    for cfg in groups:
        for head in cfg['head_ids']:
            heads+=1;mode=cfg['mode']
            if mode in ('all_to_first','global_to_frame'):pairs=n*per
            elif mode=='stride':
                stride=cfg.get('stride',4);start=head%stride if cfg.get('shift',False) else 0
                ids=set(range(start,n,stride))
                if cfg.get('include_as',True):ids|=special|set(range(per))
                pairs=n*len(ids)
            elif mode=='q_probe_topk':
                assert not cfg.get('include_as',False)
                count=max(1,n//cfg['stride']) if 'stride' in cfg else min(cfg.get('topk',1024),n)
                pairs=n*count
            elif mode=='dino_topk':pairs=len(special)*len(range(0,n,cfg.get('dino_sp2sp_stride',1)))+frames*patches*(4*frames)
            else:raise ValueError(mode)
            result+=pairs
    assert heads==16
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/eth3d_speed_fix_20261105');a=p.parse_args()
    out=a.output;dest=out/'reports';dest.mkdir(exist_ok=True);plan=read(out/'frozen_plan_grouped.json');spec=plan['methods']['grouped_saf3r'];config=read(Path(spec['config']))
    before_root=ROOT/'outputs/integrated_quality_speed_20261105/final_runs';runs=out/'final_runs'
    profile_path=ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json';profile=read(profile_path)
    audits=[];scenes=[];macro=[];density=[];run_index=[]
    assert hashlib.sha256(Path(spec['config']).read_bytes()).hexdigest()==spec['config_sha256']
    for dataset,names in plan['datasets'].items():
        for scene in names:
            oldpath=before_root/dataset/scene/'main/result.json';newpath=runs/dataset/scene/'grouped_saf3r/result.json';old=read(oldpath);new=read(newpath)
            assert old['quality']['complete'] and new['quality']['complete']
            checks={k:old['input'][k]==new['input'][k] for k in ['image_paths','frame_indices','shape','checkpoint_sha256']}
            checks['quality_protocol']=old['quality_protocol']==new['quality_protocol']
            checks['evaluation_library']=all(old['source_sha256'].get(k)==v for k,v in new['source_sha256'].items() if k.startswith('evaluation/'))
            checks['preprocessing']=old['source_sha256']['saf3r/utils/data_utils.py']==new['source_sha256']['saf3r/utils/data_utils.py']
            checks['frozen_source']=all(new['source_sha256'].get(k)==v for k,v in plan['model_source_sha256'].items())
            checks['frozen_config']=new['config']==config
            checks['native_top4']=new['saf3r']['topk']==4 and new['saf3r']['lazy_dino_topk']
            checks['execution_policy']=new['execution_policy']==dict(release_cuda_cache_each_block=False,same_pattern_head_batching=True)
            monitor=read(runs/f'{dataset}_{scene}_grouped_saf3r.monitor.json');checks['gpu_timing_overlap_free']=monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
            checks['cpu_affinity']=new['hardware']['cpu_affinity']==plan['timing_protocol']['cpu_affinity_map'][str(monitor['gpu'])]
            assert all(checks.values()),(dataset,scene,checks)
            audits.append(dict(dataset=dataset,scene=scene,checks=checks))
            for method,d,path in [('before_main',old,oldpath),('grouped_saf3r',new,newpath)]:
                v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-9
                scenes.append(dict(dataset=dataset,scene=scene,scope=f'{dataset}/{scene}',method=method,frames=d['frames'],**v,source=str(path),speedup_vs_dense=d['quality'].get('speedup_vs_vggt'),speedup_unavailable=d['quality'].get('speedup_unavailable')))
            layers=read(newpath.parent/'grouped_layers.json');assert len(layers)==24
            frames=new['frames'];height,width=new['input']['shape'][-2:];patches=(height//14)*(width//14);n=frames*(patches+5)
            exact=[expected_pairs(profile[str(i)],frames,patches) for i in range(24)]
            assert [l['main_qk_pairs'] for l in layers]==exact,(dataset,scene,'density/profile mismatch')
            global_main=sum(exact);dense_global=24*16*n*n;local=24*16*frames*(patches+5)**2
            drow=dict(dataset=dataset,scene=scene,frames=frames,global_sparsity=1-global_main/dense_global,including_unchanged_frame_attention_sparsity=1-(global_main+local)/(dense_global+local),same_main_qk_pairs_as_original_saf3r=True,extra_query_probe_qk_pairs=sum(l['query_probe_qk_pairs'] for l in layers),extra_lazy_correspondence_qk_pairs=sum(l['lazy_correspondence_qk_pairs'] for l in layers),scope='Unique main-attention logical QK; routing overhead separately counted and fully timed; not whole-model FLOP sparsity')
            density.append(drow)
            run_index.append(dict(dataset=dataset,scene=scene,method='grouped_saf3r',quality_complete=True,quality=new['quality'],timing=new['timing'],hardware=new['hardware'],source=str(newpath)))
        for method in ['before_main','grouped_saf3r']:
            rows=[r for r in scenes if r['dataset']==dataset and r['method']==method]
            assert len(rows)==len(names)
            macro.append(dict(dataset=dataset,scope=f'{dataset} ({len(names)}/{len(names)}, equal-scene)',method=method,**{k:statistics.mean(r[k] for r in rows) for k in FIELDS}))
    speed=read(dest/'paired_speed.json');assert speed['scenes']==7
    for scene in plan['datasets']['7scenes']:
        for method in ['dense','saf3r']:
            path=runs/'7scenes'/scene/method/'timing_result.json';d=read(path)
            assert not d['quality']['complete'] and all(v is None for v in d['quality']['values'].values())
            assert d['execution_policy']==dict(release_cuda_cache_each_block=True,same_pattern_head_batching=False)
            run_index.append(dict(dataset='7scenes',scene=scene,method=method,quality_complete=False,quality=d['quality'],timing=d['timing'],hardware=d['hardware'],source=str(path)))
    summary=dict(complete=True,quality_runs=18,timing_only_control_runs=14,macro=macro,per_scene=scenes,sparsity=density,speed=speed,audits=audits,run_index=run_index,units=dict(auc03='fraction',auc30='fraction',f1='fraction',acc='m',comp='m',cd='m'),cd_definition='(ACC+Comp)/2, unsquared nearest-neighbour distances',published_reference_rows=[])
    save(dest/'summary.json',summary);save(dest/'audit.json',audits)
    for label,rows in [('per_scene',scenes),('sparsity',density)]:
        with (dest/f'{label}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# ETH3D regression repair: grouped SAF3R','',
          'Frozen before dataset quality. 7Scenes: all seven scenes,128frames each. ETH3D: all11scenes, up to128frames (all available selected valid frames). Same checkpoint SHA-256, exact input paths/indices, preprocessing, pose protocol, alignment/fusion, downsampling and thresholds as before_main. Dataset rows are equal-scene macro averages. No7Scenes test-truth parameter selection.','',quality_table(macro),'',
          'AUC/F1 are percentages in this table and fractions in raw JSON. Distances are metres. CD=(ACC+Comp)/2, unsquared distances. 7Scenes reconstruction threshold/downsample:0.05m /0.0078125m. ETH3D:0.25m /0.0390625m. Every new quality run includes pose and recon_unposed.','',
          '## Adopted implementation and diagnosis','',
          'The final implementation uses the original frozen SAF3R per-head routes, batches identical static patterns, and retains the allocator cache instead of purging it after every block. Dynamic query-probe heads and lazy correspondence heads use their original scalar implementation. Image-dependent correspondence Top-K is reset/recomputed inside every complete-forward timer. Static geometry-index banks cache positions only, never image content.','',
          'The failed shared-Value method used a fixed10% /22% /10% layer schedule and one support bank across heads. That schedule was calibrated at128frames and did not reproduce the original SAF3R masks on shorter ETH3D sequences. On pipes, its measured global retained density was15.963%, while the original SAF3R profile retains20.201%; late stride-4 heads also retain the anchor/special-token union. This is a real budget/layout difference, not an evaluation-score adjustment. The final masks have exactly the original SAF3R logical QK counts on every scene, independently checked from the profile.','',
          'A second diagnostic issue was local calibration on already corrupted sparse Q/K/V. Tiny local Camera Query error hid accumulated independent-Dense feature error. Protecting anchor Keys and preserving separate-head Value features improved the approximation, but those research variants either remained too slow or retained significant geometric loss. They were not promoted. The final route change is explicit: the shared-Value selector is not part of the adopted inference path.','',
          'Current Dense quality was reproduced on pipes and playground in the regression-diagnosis batch; baseline drift did not explain the severe loss. The grouped implementation was also checked against original SAF3R on a disjoint RGB128-frame calibration: pose, full depth and depth-confidence outputs were elementwise equal, including after allocator-cache reuse. Those numerical/timing diagnostics have six geometric fields N/A; they are not dataset quality evaluations.','',
          'SAF3R configuration: repository eth3d-train-fltr-calib_cmpmse.json, lazy first-correspondence-head Top-4 per frame (512 selected patch indices at128frames). This is not the paper DINO-feature Top-K, not Top-128, and not a claimed paper reproduction. No published/H100 reference is used as a local timing baseline.','',
          '## Paired speed','',
          'See [speed.md](speed.md) and paired_speed.json. Original VGGT Dense, original SAF3R and grouped SAF3R share exact inputs, physical GPU, CPU affinity, checkpoint and1warmup+3complete-forward timing per scene. All model stages, routing/Top-K resets, grouping/gathers/scatters and cache misses are included. Loading checkpoint/images is outside full-forward timing for all methods. Original controls retain their original per-block cache purge; only the explicit grouped candidate changes that allocator policy.','',
          'The14control runs were explicitly timing-only. All six quality values remain N/A with the concrete reason; no control quality scores are fabricated. ETH3D cold quality-forward times are not a speed benchmark. Earlier batches and noisy development timings are not used as paired speed baselines. Peak allocated and reserved memory are recorded separately in raw timings; allocator-cache reuse can increase reserved memory.','',
          '## Per-scene quality','',quality_table(scenes),'',
          '## Sparsity','',
          '| Dataset / scene | Frames | Global main-attention sparsity | Including unchanged frame attention |','|---|---:|---:|---:|']
    for d in density:text.append(f"| {d['dataset']}/{d['scene']} | {d['frames']} | {d['global_sparsity']*100:.4f}% | {d['including_unchanged_frame_attention_sparsity']*100:.4f}% |")
    text+=['','These are logical attention-QK sparsities, not whole-model FLOP reductions. Query-probe and lazy correspondence selection costs are fully timed and separately listed in sparsity.csv/summary.json.','',
           'All18input/protocol/source/CPU/GPU/density audits passed. Raw per-run six-metric tables, metrics, source snapshots, input lists and GPU monitors remain under final_runs/. The earlier full-head and compact-Value campaigns remain separate research artifacts.']
    (dest/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(macro=macro,speed=speed['macro'],speed_gates=speed['gates'],audits=len(audits)),indent=2))
if __name__=='__main__':main()
