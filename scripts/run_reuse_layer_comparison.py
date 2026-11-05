"""GPU4/5/6 dynamic diagnostic queue; each worker owns one GPU at a time."""
import concurrent.futures
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/reuse_layer_comparison_20261105'
REFERENCE = ROOT.parent/'SAF3R-original-reuse-reference-20261105'


def prepare_reference():
    snapshot=ROOT/'outputs/reuse_single_change_7scenes128_20261029/heads/original'
    manifest=json.loads((snapshot/'source_sha256.json').read_text())
    if not REFERENCE.exists():
        subprocess.run(['git','worktree','add','--detach',str(REFERENCE),'HEAD'],cwd=ROOT,check=True)
        for name,digest in manifest.items():
            source=snapshot/'source_snapshot'/name
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
            shutil.copyfile(source,REFERENCE/name)
        for name in ('checkpoints','workspace'):
            path=REFERENCE/name
            if not path.exists():path.symlink_to(ROOT/name,target_is_directory=True)
        (REFERENCE/'reference_overlay_manifest.json').write_text(json.dumps(dict(source=str(snapshot),sha256=manifest),indent=2)+'\n')
    for name,digest in manifest.items():
        assert hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()==digest, f'Historical source changed: {name}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--causal-controls',action='store_true')
    parser.add_argument('--causal-independent',action='store_true')
    parser.add_argument('--support-controls',action='store_true')
    parser.add_argument('--cluster-controls',action='store_true')
    parser.add_argument('--cluster-validation',action='store_true')
    args=parser.parse_args()
    prepare_reference()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes=['pipes','playground','courtyard']
    # Use the recorded minimal method config, not reporting/runner metadata.
    original=json.loads((ROOT/'outputs/reuse_single_change_7scenes128_20261029/heads/original/result.json').read_text())['config']
    original_config=OUT/'original_config.json'
    original_config.write_text(json.dumps(original,indent=2)+'\n')
    phases=[('original',REFERENCE,original_config,None,False),
            ('main_independent',ROOT,ROOT/'outputs/integrated_quality_speed_20261105/frozen_configs/main.json','original',False),
            ('main_teacher_forced',ROOT,ROOT/'outputs/integrated_quality_speed_20261105/frozen_configs/main.json','original',True),
            ('fast_repair_independent',ROOT,ROOT/'outputs/eth3d_repair_20261105/frozen_configs/fast_repair.json','original',False),
            ('fast_repair_teacher_forced',ROOT,ROOT/'outputs/eth3d_repair_20261105/frozen_configs/fast_repair.json','original',True)]
    if args.causal_controls:
        scenes=['pipes']
        phases=[('original_repeat',REFERENCE,original_config,'original',False)]+[
            (name+'_teacher_forced',ROOT,OUT/'frozen_causal_configs'/f'{name}.json','original',True)
            for name in ('uniform20','history_unit_weight','matched_frontend_v2')]
    if args.causal_independent:
        scenes=['pipes']
        phases=[(name+'_independent',ROOT,OUT/'frozen_causal_configs'/f'{name}.json','original',False)
                for name in ('uniform20','history_unit_weight','matched_frontend_v2')]
    if args.cluster_validation:
        scenes=['courtyard','playground']
        phases=[('original_cluster_count',ROOT,OUT/'frozen_causal_configs/original_cluster_count.json','original',False)]
    support_names=('source_anchored_only','query32_only','source_anchored_query32')
    if args.cluster_controls:
        support_names=('original_cluster_count','uniform20_original_cluster_count','uniform20_original_cluster_count_lloyd2')
    grouped_controls=args.support_controls or args.cluster_controls
    if grouped_controls:
        scenes=['pipes::'+name for name in support_names]
    def physical_scene(scene):return scene.split('::')[0]
    def scene_phases(scene):
        if not grouped_controls:return phases
        name=scene.split('::')[1]
        return [(name,ROOT,OUT/'frozen_causal_configs'/f'{name}.json','original',False)]
    def worker(gpu,scene,phase):
        scene=physical_scene(scene)
        name,source,config,reference,forced=phase
        folder=OUT/scene/name
        cmd=[sys.executable,str(ROOT/'scripts/compare_reuse_layers.py'),'--source-root',str(source),
             '--input',str(ROOT/f'outputs/eth3d_repair_20261105/{scene}_rgb.json'),
             '--config',str(config),'--output-dir',str(folder)]
        if reference:cmd+=['--reference',str(OUT/scene/reference)]
        if forced:cmd+=['--teacher-forced']
        lease=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),
               '--monitor',str(OUT/scene/f'{name}_gpu.json'),'--log',str(OUT/scene/f'{name}.log'),'--',*cmd]
        print('START',gpu,scene,name,flush=True)
        result=subprocess.run(lease,cwd=source)
        print('EXIT',gpu,scene,name,result.returncode,flush=True)
        if result.returncode:
            (OUT/scene/'failure.json').write_text(json.dumps(dict(phase=name,returncode=result.returncode),indent=2))
        return result.returncode
    def complete(scene,phase):
        scene=physical_scene(scene)
        path=OUT/scene/phase[0]/'status.json'
        return path.exists() and json.loads(path.read_text()).get('stage')=='diagnostic_complete'
    def free_gpus():
        gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        uuid_to_id={line.split(',')[1].strip():int(line.split(',')[0]) for line in gpu_info.splitlines()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        busy={uuid_to_id[line.split(',')[0].strip()] for line in apps.splitlines() if line.strip()}
        return set((4,5,6))-busy
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        running={};failed=set()
        while True:
            for future,(gpu,scene) in list(running.items()):
                if future.done():
                    if future.result():failed.add(scene)
                    del running[future]
            pending=[(scene,next((phase for phase in scene_phases(scene) if not complete(scene,phase)),None)) for scene in scenes if scene not in failed]
            pending=[(scene,phase) for scene,phase in pending if phase is not None]
            if not pending and not running:break
            used={gpu for gpu,scene in running.values()}
            active={scene for gpu,scene in running.values()}
            available=sorted(free_gpus()-used)
            for scene,phase in pending:
                if not available:break
                if scene in active:continue
                gpu=available.pop(0)
                running[pool.submit(worker,gpu,scene,phase)]=(gpu,scene)
            time.sleep(2)


if __name__=='__main__':main()
