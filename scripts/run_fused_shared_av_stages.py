"""Gated execution: manual ablations, independent ETH3D calibration, frozen7Scenes.

Expensive stages refuse to start unless warmed128-frame end-to-end speed and
memory pass. The seven-scene truth is never used to choose alpha/retention.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from saf3r.patch.fused_shared_av import FusedSharedAVConfig
from scripts.report_fused_shared_av import report,SCENES


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')


def require_gate(path):
    if path is None:raise ValueError('A warmed128-frame gate report is mandatory before expensive sweeps')
    data=json.loads(path.read_text())
    if data.get('frames')!=[128] or not data.get('speed_passed') or not data.get('memory_passed') or not data.get('warmed_protocol_valid'):
        raise ValueError('128-frame2x speed/no-extra-memory gate FAILED; do not launch full calibration/evaluation')


def evaluate_pair(root,dataset,scenes,schedule,gpus,modes=('dense','fused')):
    def one_gpu(gpu,subset):
        for scene in subset:
            for mode in modes:
                result=root/mode/scene/'result.json'
                if result.exists():raise ValueError('Existing completed result; choose new stage directory')
                cmd=[sys.executable,str(ROOT/'scripts/evaluate_fused_shared_av.py'),
                    '--output-dir',str(root),'--dataset',dataset,'--scene',scene,'--frames','128',
                    '--mode',mode,'--schedule',str(schedule),'--warmup','1','--repeats','3',
                    '--tasks','pose','depth','recon_unposed']
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                root.mkdir(parents=True,exist_ok=True)
                with (root/f'{scene}_{mode}.log').open('w') as log:
                    process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:raise RuntimeError(f'{scene}/{mode} failed; inspect log')
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(one_gpu,gpu,scenes[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for future in futures:future.result()
    return {mode:report(root,mode,scenes) for mode in modes if mode!='dense'}


def calibration(args):
    from omegaconf import OmegaConf
    profile=OmegaConf.load(ROOT/'configs/profile/profile_saf3r_vggt.yaml')
    scenes=list(profile.datasets.eth3d.scenes)
    base=json.loads(args.schedule.read_text())
    # Model/cluster/probe semantics fixed. Only predeclared manual budget choices
    # are tried here; this is not an online self-adaptive controller.
    selected=None;trial=0;records=[]
    def attempt(settings,label):
        nonlocal trial,selected
        trial+=1;directory=args.output_dir/f'{trial:02d}_{label}'
        config=FusedSharedAVConfig(**settings)
        schedule=directory/'schedule.json';save(schedule,asdict(config))
        summary=evaluate_pair(directory,'eth3d',scenes,schedule,args.gpus)['fused']
        accepted=summary['quality']['passed'] and summary['speed_passed'] and summary['memory_passed']
        latency=sum(r['fused_seconds'] for r in summary['per_scene'])/len(scenes)
        records.append(dict(label=label,accepted=accepted,seconds=latency,root=str(directory)))
        if accepted and (selected is None or latency<selected['seconds']):
            selected=dict(settings=asdict(config),seconds=latency,trial=str(directory))
        save(args.output_dir/'calibration_progress.json',dict(trials=records,selected=selected))
        return accepted
    for rho in [.2,.15,.1]:attempt(dict(base,retentions=[rho]*24),f'uniform{round(rho*100)}')
    if selected is None:
        save(args.output_dir/'calibration_status.json',dict(status='no_candidate_passed',frozen=False));return
    for name,start,stop in [('shallow',0,10),('middle',10,17),('deep',17,24)]:
        for rho in [.15,.1]:
            candidate=json.loads(json.dumps(selected['settings']))
            if any(candidate['retentions'][i]<rho for i in range(start,stop)):continue
            for i in range(start,stop):candidate['retentions'][i]=rho
            attempt(candidate,f'{name}{round(rho*100)}')
    target=args.output_dir/'frozen_schedule.json';save(target,selected['settings'])
    save(args.output_dir/'calibration_status.json',dict(status='frozen',selected=selected,
        schedule_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        calibration_profile='configs/profile/profile_saf3r_vggt.yaml',scenes=scenes,
        used_7scenes_truth=False))


def run(args):
    require_gate(args.gate_report)
    if len(set(args.gpus))!=len(args.gpus) or not args.gpus:raise ValueError('Distinct GPUs required')
    telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    usage={int(line.split(',')[0]):int(line.split(',')[1]) for line in telemetry.strip().splitlines()}
    if any(gpu not in usage or usage[gpu]>512 for gpu in args.gpus):
        raise RuntimeError('Requested GPUs are not idle; do not interfere with other users')
    if args.output_dir.exists():raise ValueError('Use new stage output directory')
    FusedSharedAVConfig(**json.loads(args.schedule.read_text()))
    args.output_dir.mkdir(parents=True)
    save(args.output_dir/'stage_manifest.json',dict(stage=args.stage,started_unix=time.time(),
        schedule_source=str(args.schedule),schedule_sha256=hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
        gate_source=str(args.gate_report),gpus=args.gpus,status='running'))
    if args.stage=='calibrate':calibration(args)
    elif args.stage=='ablate':
        # All three modes share inputs/budgets; Value-only and Attention-only
        # are controls, never automatic replacements for a failed fusion result.
        evaluate_pair(args.output_dir/'ablations','7scenes',SCENES,args.schedule,args.gpus,
            modes=('dense','fused','value','attention'))
    else:
        frozen=args.schedule.parent/'calibration_status.json'
        if not frozen.exists():raise ValueError('Final evaluation requires independently frozen calibration schedule')
        metadata=json.loads(frozen.read_text())
        if metadata.get('used_7scenes_truth') is not False or metadata.get('schedule_sha256')!=hashlib.sha256(args.schedule.read_bytes()).hexdigest():
            raise ValueError('Invalid calibration freeze provenance')
        evaluate_pair(args.output_dir/'evaluation','7scenes',SCENES,args.schedule,args.gpus)
    save(args.output_dir/'stage_status.json',dict(status='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['ablate','calibrate','final'],required=True)
    p.add_argument('--gate-report',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',required=True)
    run(p.parse_args())
