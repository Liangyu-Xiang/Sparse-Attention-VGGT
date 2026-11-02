"""Package raw reviewable evidence without checkpoints or duplicate source trees."""
import argparse,csv,gzip,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--refresh',action='store_true');args=p.parse_args();base=args.results_root;dest=ROOT/'outputs/grid_local_history_20261102';dest.mkdir(exist_ok=args.refresh,parents=True)
    def copy(src,relative):
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix=='.csv':target.write_bytes(src.read_bytes().replace(b'\r\n',b'\n'))
        else:shutil.copy2(src,target)
    final=base/'grid_local_final_20261102';summary=json.loads((final/'summary.json').read_text());assert summary['complete']
    for f in final.iterdir():
        if f.is_file():copy(f,Path('summary')/f.name)
    groups=['grid_local_validation_final_20261102','grid_local_real_window_v2_20261102','grid_local_tiles_20261102','grid_local_kernel_audit_20261102','grid_local_rollout_stability_20261102','grid_local_paired_heads128_20261102']
    for group in groups:
        folder=base/group
        for f in folder.iterdir():
            if f.suffix=='.json':
                if f.name.endswith('_profile.json'):
                    target=dest/'diagnostics'/group/(f.name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
                    with f.open('rb') as source,gzip.open(target,'wb') as output:shutil.copyfileobj(source,output)
                else:copy(f,Path('diagnostics')/group/f.name)
        copy(base/(group+'_monitor.json'),Path('diagnostics')/(group+'_monitor.json'))
    sources=[]
    for r in csv.DictReader((final/'per_scene.csv').open()):
        path=Path(r['source']);prefix=Path('quality')/r['dataset']/r['scene']/r['method']
        copy(path,prefix/'result.json')
        if r['method']=='grid_local':
            for name in ['quality.json','quality.md','layers.json','input.json','config.json','source_sha256.json','status.json','forward.json']:
                f=path.parent/name
                if f.exists():copy(f,prefix/name)
            copy(path.parent.parent/'lease_monitor.json',prefix/'lease_monitor.json')
        sources.append(dict(dataset=r['dataset'],scene=r['scene'],method=r['method'],original_source=str(path),packaged_source=str(prefix/'result.json')))
    for group in ['grid_local_quality_20261102','grid_local_quality_retry_20261102']:
        for name in ['manifest.json','progress.json','completion.json','scheduled.json']:
            if (base/group/name).exists():copy(base/group/name,Path('quality_manifests')/group/name)
    copy(base/'grid_local_real_window_20261102/result.json',Path('rejected_layout_attempt/result.json'))
    (dest/'rejected_layout_attempt/rejected.json').write_text(json.dumps(dict(accepted=False,reason='Non-contiguous observation layout was interpreted as contiguous. Fixed in v2; do not use first-attempt timing or numerical result as validated evidence.'),indent=2)+'\n')
    (dest/'quality_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    report=ROOT/'docs/experiments/grid_local_history.md';report.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(final/'report.md',report)
    hashes={str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file() and f.name!='artifact_sha256.json'}
    (dest/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(files=len(hashes),bytes=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()),destination=str(dest)),indent=2))


if __name__=='__main__':main()
