"""Run a pinned revision only after the preceding seven-scene evaluation succeeds."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
from scripts.gpu_recovery_guard import query_inventory,FAULT_UUID

SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
DEVICES={4:'GPU-a67db5fc-c14e-4884-1909-a7c7050ef267',
         5:'GPU-917c95c3-9023-f077-f577-dae0f7f9d717'}

def predecessor_complete(previous):
    marker=previous/'completion.json'
    if not marker.exists():return False
    if not json.loads(marker.read_text()).get('all_seven_complete'):
        raise RuntimeError('Predecessor terminated without seven complete quality results')
    for scene in SCENES:
        p=previous/'quality128'/scene/'result.json'
        if not p.exists() or not json.loads(p.read_text())['quality']['complete']:
            raise RuntimeError('Predecessor completion marker disagrees with raw quality: '+scene)
    return True

def fingerprint(branch):
    names=('saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py',
           'scripts/evaluate_reuse_exploration.py','configs/evaluation/reuse_residual_exploration.json')
    return {n:hashlib.sha256((branch/n).read_bytes()).hexdigest() for n in names}

def run(a):
    root=a.output_dir.resolve();branch=a.branch_dir.resolve();previous=a.previous_output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'dependency.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    commit=subprocess.check_output(['git','-C',str(branch),'rev-parse','HEAD'],text=True).strip()
    if commit!=a.pin:raise RuntimeError('Pinned checkout mismatch')
    frozen=fingerprint(branch)
    save(root/'requested.json',dict(pin=a.pin,previous_output=str(previous),branch_dir=str(branch),
         frames=128,scenes=SCENES,devices=DEVICES,source_sha256=frozen,
         predecessor_must_finish=True,no_baseline_reruns=True))
    write_quality(root,quality_record({},'Queued revision; waiting for preceding seven-scene quality evaluation'))
    save(root/'dependency_status.json',dict(stage='waiting_for_predecessor',previous=str(previous)))
    while not predecessor_complete(previous):
        if (previous/'blocked.json').exists():
            raise RuntimeError('Predecessor is blocked; manual inspection required before next revision')
        time.sleep(30)
    save(root/'dependency_status.json',dict(stage='waiting_for_idle_gpu',previous_complete=True))
    while True:
        idle=[d for d in query_inventory() if d['gpu'] in DEVICES and
              d['uuid']==DEVICES[d['gpu']] and d['uuid'].lower()!=FAULT_UUID.lower() and
              d['memory_used_mib']<=256]
        if idle:break
        time.sleep(30)
    if fingerprint(branch)!=frozen:raise RuntimeError('Frozen implementation changed while queued')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':idle[0]['uuid'],'PYTHONPATH':str(branch),
         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    save(root/'dependency_status.json',dict(stage='unit_tests',gpu=idle[0],previous_complete=True))
    with (root/'validation.log').open('a') as log:
        subprocess.run([sys.executable,'-m','pytest','tests/test_reuse_residual_exploration.py','-q'],
                       cwd=branch,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        save(root/'dependency_status.json',dict(stage='preflight',gpu=idle[0],previous_complete=True))
        subprocess.run([sys.executable,'scripts/evaluate_reuse_exploration.py',
            '--method','reuse_explore','--scene','heads','--frames','2','--output-dir',str(root/'preflight'),
            '--checkpoint',str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'),
            '--warmup','1','--repeats','1','--diagnostic'],cwd=branch,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    if fingerprint(branch)!=frozen:raise RuntimeError('Frozen implementation changed during validation')
    save(root/'dependency_status.json',dict(stage='dataset_evaluation',previous_complete=True))
    with (root/'coordinator.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_unified_branch_7scenes.py'),
            '--branch-dir',str(branch),'--output-dir',str(root),'--pin',a.pin],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    completed=root/'completion.json'
    if not completed.exists() or not json.loads(completed.read_text()).get('all_seven_complete'):
        raise RuntimeError('Dataset evaluation did not complete all seven quality runs; inspect coordinator')
    save(root/'dependency_status.json',dict(stage='complete',previous_complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--previous-output',required=True,type=Path)
    p.add_argument('--pin',required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        a.output_dir.mkdir(parents=True,exist_ok=True)
        reason=f'{type(exc).__name__}: {exc}'
        save(a.output_dir/'dependency_status.json',dict(stage='blocked',reason=reason))
        write_quality(a.output_dir,quality_record({},reason))
        raise
