"""Preflight-gated ScanNet-50 DINO64/Dense evaluation on idle local GPUs."""
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import re
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.evaluate_scannet_paper64 import DATA,GT,PROTOCOL
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import FIELDS,LABELS,quality_record,quality_cells

OUT=ROOT/'outputs/scannet50_paper_dino64_20261027'
GPUS=['GPU-f0913c13-448b-e88d-92fe-8d47264d279f','GPU-8e332fc5-55f4-0d85-1a2d-40bd87342c1f',
      'GPU-a67db5fc-c14e-4884-1909-a7c7050ef267','GPU-917c95c3-9023-f077-f577-dae0f7f9d717']
LOCK=threading.Lock();STATE={}

def active(gpu):
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(p[1].strip()) for line in s.splitlines() if len(p:=line.split(','))==2 and p[0].strip()==gpu}

def progress(key,**data):
    with LOCK:
        STATE.setdefault('jobs',{})[key]=data
        save(OUT/'progress.json',STATE)

def worker_run(gpu,frames,scene,method,phase):
    folder=OUT/phase/str(frames)/scene/method
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())['quality']['complete']
    if folder.exists():raise ValueError(f'Refuse to overwrite {folder}')
    key=f'{phase}/{frames}/{scene}/{method}'
    while active(gpu):
        progress(key,state='waiting_for_idle_gpu',gpu_uuid=gpu)
        time.sleep(15)
    log=OUT/'logs'/f'{phase}_{frames}_{scene}_{method}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    cmd=[sys.executable,'scripts/evaluate_scannet_paper64.py','--scene',scene,'--frames',str(frames),'--method',method,'--output-dir',str(folder)]
    foreign=set()
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        progress(key,state='running',pid=p.pid,gpu_uuid=gpu,log=str(log))
        while p.poll() is None:
            if not (folder/'forward.json').exists():foreign.update(active(gpu)-{p.pid})
            time.sleep(3)
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'gpu_contention.json',dict(uuid=gpu,foreign_pids=sorted(foreign),monitored=True))
    ok=p.returncode==0 and (folder/'result.json').exists() and json.loads((folder/'result.json').read_text())['quality']['complete']
    progress(key,state='complete' if ok else 'failed',returncode=p.returncode,gpu_uuid=gpu,log=str(log))
    return ok

def pair_speed(folder):
    ps=[folder/m/'result.json' for m in ['dense','saf3r']]
    if not all(p.exists() for p in ps):return
    a,b=[json.loads(p.read_text()) for p in ps]
    checks=all(a[k]==b[k] for k in ['input','hardware','protocol'])
    for m in ['dense','saf3r']:
        p=folder/m/'gpu_contention.json'
        checks=checks and p.exists() and not json.loads(p.read_text())['foreign_pids']
    if checks:
        b['quality'].update(speedup_vs_vggt=a['timing']['median_seconds']/b['timing']['median_seconds'],speedup_unavailable=None)
        save(ps[1],b);write_quality(ps[1].parent,b['quality'])

def report():
    rows=[];scenes=STATE.get('scenes',[])
    for frames in [300,500,700]:
        for scene in scenes:
            for method in ['dense','saf3r']:
                p=OUT/'quality'/str(frames)/scene/method/'result.json'
                # Preflight scene is reused only as raw quality, never mislabelled as a 50-scene macro.
                if not p.exists() and scene=='scene0000_00':p=OUT/'preflight'/str(frames)/scene/method/'result.json'
                if p.exists():
                    d=json.loads(p.read_text());q=d['quality']
                    rows.append(dict(requested_frames=frames,scene=scene,method=method,frames=d['frames'],**q['values'],
                        complete=q['complete'],seconds=d['timing']['median_seconds'],speedup=q['speedup_vs_vggt'],
                        peak_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples']),unavailable=q['unavailable']))
    macro=[]
    for frames in [300,500,700]:
        for method in ['dense','saf3r']:
            group=[x for x in rows if x['requested_frames']==frames and x['method']==method]
            if len(group)==50 and all(x['complete'] for x in group):
                macro.append(dict(requested_frames=frames,method=method,**{k:sum(x[k] for x in group)/50 for k in FIELDS}))
    save(OUT/'summary.json',dict(per_scene=rows,complete_50_scene_macro=macro,preflight=STATE.get('preflight'),protocol=PROTOCOL))
    if rows:
        with (OUT/'per_scene.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ScanNet-50: non-lazy DINO64 public configuration','',
        'Not an exact reproduction of unpublished scene-wise paper K. Local RTX4090 timings are not H100 paper timings.',
        'All 50 official scenes; 300/500/700 valid-pose uniform frame caps; no duplicated frames.',
        'Six-metric reconstruction extension: 5cm F1, 5cm voxel, 100k point cap, 0.5m clipping; CD=(ACC+Comp)/2. Official sum retained separately.',
        '', '| Frames cap | Scene | Method | Actual frames | '+' | '.join(LABELS)+' | Seconds | Speedup | Peak GiB |',
        '|---:|---|---|---:|'+'---:|'*9]
    for x in rows:
        vals=['N/A' if x[k] is None else f"{x[k]*(100 if k in ['auc03','auc30','f1'] else 1):.5f}" for k in FIELDS]
        speed='N/A' if x['speedup'] is None else f"{x['speedup']:.3f}x"
        lines.append('| '+' | '.join([str(x['requested_frames']),x['scene'],x['method'],str(x['frames']),*vals,f"{x['seconds']:.3f}",speed,f"{x['peak_gib']:.3f}"])+' |')
    lines+=['','## Complete 50-scene macro only','```json',json.dumps(macro,indent=2),'```',
        '', '## Preflight gates','```json',json.dumps(STATE.get('preflight',{}),indent=2),'```']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')

def main():
    if OUT.exists():raise ValueError('Fresh output root required')
    (OUT/'logs').mkdir(parents=True)
    STATE.update(started_unix=time.time(),phase='stage0',jobs={},scenes=sorted(p.name for p in DATA.glob('scene*') if p.is_dir()),protocol=PROTOCOL)
    assert len(STATE['scenes'])==50
    reference=urllib.request.urlopen('https://raw.githubusercontent.com/mystorm16/FastVGGT/main/eval/scannet_50.yaml',timeout=30).read().decode()
    assert set(re.findall(r'scene\d{4}_\d{2}',reference))==set(STATE['scenes']), 'Official scene list mismatch'
    (OUT/'official_scannet_50.yaml').write_text(reference)
    with (OUT/'stage0.log').open('w') as log:
        for cmd in [[sys.executable,'-m','py_compile','scripts/evaluate_scannet_paper64.py','scripts/run_scannet_paper64.py'],
                    [sys.executable,'-m','pytest','-q','tests/test_scannet_paper64.py']]:
            if subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode:raise RuntimeError('Stage0 failed')
    from scripts.evaluate_scannet_paper64 import select
    manifest={}
    # Freeze valid-pose-selected frames for every scene/size before any inference.
    for scene in STATE['scenes']:
        for frames in [300,500,700]:
            ids,paths,poses,first,mesh=select(scene,frames)
            manifest[f'{frames}/{scene}']=dict(actual_frames=len(ids),frame_ids=ids,image_paths=paths,gt_mesh=str(mesh))
    save(OUT/'input_manifest.json',manifest)
    STATE['phase']='preflight';save(OUT/'progress.json',STATE)
    gates={}
    def preflight(gpu,frames):
        results={}
        for method in ['dense','saf3r']:
            results[method]=worker_run(gpu,frames,'scene0000_00',method,'preflight')
        pair_speed(OUT/'preflight'/str(frames)/'scene0000_00')
        return frames,results
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(preflight,gpu,n) for gpu,n in zip(GPUS,[300,500,700])]
        for f in concurrent.futures.as_completed(futures):
            frames,passed=f.result();gates[str(frames)]=passed
            with LOCK:
                STATE['preflight']=gates.copy();save(OUT/'progress.json',STATE);report()
    tasks=queue.Queue()
    blocked=[]
    for frames in [300,500,700]:
        for scene in STATE['scenes']:
            if scene=='scene0000_00':continue
            methods=[m for m in ['dense','saf3r'] if gates[str(frames)][m]]
            for method in ['dense','saf3r']:
                if method not in methods:
                    failure=OUT/'preflight'/str(frames)/'scene0000_00'/method/'failure.json'
                    reason=json.loads(failure.read_text())['reason'] if failure.exists() else 'Preflight quality failed; inspect preflight results'
                    blocked.append(dict(frames=frames,scene=scene,method=method,state='not_started_preflight_failed',quality=quality_record({},reason)))
            if methods:tasks.put((frames,scene,methods))
    save(OUT/'blocked_jobs.json',blocked)
    STATE['phase']='quality' if not tasks.empty() else 'blocked_by_preflight'
    save(OUT/'progress.json',STATE)
    def consume(gpu):
        while True:
            try:frames,scene,methods=tasks.get_nowait()
            except queue.Empty:return
            for method in methods:worker_run(gpu,frames,scene,method,'quality')
            pair_speed(OUT/'quality'/str(frames)/scene)
            with LOCK:report()
            tasks.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(consume,gpu) for gpu in GPUS]
        for f in fs:f.result()
    STATE['finished_unix']=time.time()
    STATE['phase']='finished' if all(all(x.values()) for x in gates.values()) else 'partial_or_blocked'
    save(OUT/'progress.json',STATE);report()

if __name__=='__main__':main()
