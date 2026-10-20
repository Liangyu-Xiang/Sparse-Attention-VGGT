"""Quality-first heads diagnosis, up to two processes/GPU, no timing claims."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import FIELDS,LABELS,quality_cells


def gpu_state(gpu):
    line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.total',
        '--format=csv,noheader,nounits'],text=True,timeout=60).strip()
    used,total=map(int,line.split(','))
    output=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=60)
    pids={int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}
    return used,total,pids


def summarize(root,campaign):
    baseline=campaign/'quality/dense/heads/result.json'
    oldbase=ROOT/'outputs/parent_region_av/dense_teacher_ratios_128/heads/validation/manual/region/heads/result.json'
    specs=[('Dense','all',baseline),('old','all',oldbase),
        ('attention_first','all',campaign/'quality/attention_first/heads/result.json')]
    specs += [(method,stage,root/stage/method/'heads/result.json')
        for method in ('old','attention_first') for stage in ('shallow','middle','deep')]
    rows=[];missing=[];dense=None
    for method,stage,file in specs:
        if not file.exists():missing.append(dict(method=method,stage=stage));continue
        data=json.loads(file.read_text())
        if dense is None:dense=data
        if data['input']!=dense['input']:raise ValueError('Ablation input differs from Dense')
        quality=dict(data['quality'])
        quality.update(speedup_vs_vggt=None,speedup_unavailable='Concurrent quality-first diagnosis; runtime comparisons intentionally disabled')
        rows.append(dict(method=method,stage=stage,quality=quality,source=str(file),
            delta_vs_dense={key:quality['values'][key]-dense['quality']['values'][key]
                if quality['values'][key] is not None and dense['quality']['values'][key] is not None else None for key in FIELDS}))
    save(root/'comparison.json',dict(rows=rows,missing=missing,complete=not missing,
        note='One forward, no warmup/repeated timing; sparse state maintained in inactive Dense-output layers'))
    lines=['# heads128 quality-first stage ablations','',
        '| Scheme | Sparse output layers | '+' | '.join(LABELS)+' |',
        '|---|---|'+'---:|'*6]
    for row in rows:
        layers={'all':'0–23','shallow':'0–9','middle':'10–16','deep':'17–23'}[row['stage']]
        if row['method']=='Dense':layers='none'
        lines.append('| '+row['method']+' | '+layers+' | '+' | '.join(quality_cells(row['quality']))+' |')
    lines+=['','All remaining layers return original Dense output. Sparse Value/history state still evolves there.',
        'Speedup: N/A; concurrent accuracy diagnosis, not a performance benchmark.',
        'Do not infer a causal failing layer from a full-rollout layer error alone; use stage ablations first.',
        '',f'Pending rows: {missing}']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    controller=json.loads((args.campaign/'controller.json').read_text());pid=controller['pid']
    stopped=False
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/run_attention_first_campaign.py' not in command or str(args.campaign) not in command:
            raise ValueError('Broad queue PID identity mismatch')
        os.kill(pid,signal.SIGSTOP);stopped=True
    except FileNotFoundError:pass
    jobs={};failures=[];active={};pending=[(m,s) for m in ('old','attention_first') for s in ('shallow','middle','deep')]
    sources=['scripts/prioritize_heads_ablation.py','scripts/evaluate_attention_first_residual_value.py',
        'saf3r/patch/attention_first_residual_value.py','saf3r/patch/value_region_av.py','saf3r/patch/av_stage_ablation.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    save(root/'manifest.json',dict(frames=128,gpus=args.gpus,controller_pid=pid,broad_dispatch_stopped=stopped,
        warmup=0,repeats=1,timing_comparison=False,source_sha256=hashes,
        modes=['old','attention_first'],stages=dict(shallow=list(range(10)),middle=list(range(10,17)),deep=list(range(17,24))),
        admission='At most2 processes/GPU and20GiB free before admitting old; new waits for exclusive GPU because an earlier standalone full run OOMed'))
    def status():
        save(root/'status.json',dict(stage='running',jobs=jobs,failures=failures,broad_dispatch_stopped=stopped))
        summarize(root,args.campaign)
    def launch(gpu,method,stage):
        for source,digest in hashes.items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed')
        folder=root/stage
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),
                 OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        log=root/'logs'/f'{method}_{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
        stream=log.open('a')
        process=subprocess.Popen([sys.executable,'-u','scripts/evaluate_attention_first_residual_value.py',
            '--output-dir',str(folder),'--mode',method,'--scene','heads','--frames','128',
            '--active-stage',stage,'--warmup','0','--repeats','1'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return process,stream
    try:
        status()
        while pending or active:
            for key,(gpu,process,stream) in list(active.items()):
                code=process.poll()
                if code is None:continue
                stream.close();del active[key]
                method,stage=key.split('/')
                file=root/stage/method/'heads/result.json'
                if code==0 and file.exists():
                    data=json.loads(file.read_text());data['quality'].update(speedup_vs_vggt=None,
                        speedup_unavailable='Concurrent quality-first run; no timing baseline or speed claim')
                    data['benchmark_timing']=False;save(file,data);write_quality(file.parent,data['quality'])
                    jobs[key].update(stage='complete',finished_unix=time.time())
                else:
                    jobs[key].update(stage='failed',returncode=code)
                    failures.append(dict(job=key,returncode=code))
                status()
            if pending:
                for gpu in args.gpus:
                    used,total,pids=gpu_state(gpu)
                    reserved=[proc.pid for g,proc,_ in active.values() if g==gpu]
                    external=pids-set(reserved)
                    occupied=len(external)+len(reserved)
                    if occupied>=2 or total-used<20*1024:continue
                    index=None
                    for i,(method,stage) in enumerate(pending):
                        if method=='attention_first' and occupied:continue
                        index=i;break
                    if index is None:continue
                    method,stage=pending.pop(index);key=f'{method}/{stage}'
                    if (root/stage/method/'heads/result.json').exists():
                        jobs[key]=dict(stage='complete',resumed=True);status();continue
                    process,stream=launch(gpu,method,stage);active[key]=(gpu,process,stream)
                    jobs[key]=dict(stage='running',gpu=gpu,pid=process.pid,started_unix=time.time());status()
                    print('START',gpu,key,process.pid,flush=True)
            time.sleep(10)
    finally:
        if stopped:
            os.kill(pid,signal.SIGCONT);stopped=False
    save(root/'status.json',dict(stage='failed' if failures else 'complete',jobs=jobs,failures=failures,
        broad_dispatch_stopped=False,finished_unix=time.time()))
    summarize(root,args.campaign)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--campaign',type=Path,default=Path('outputs/attention_first_residual_value/campaign_7scenes128'))
    p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6]);run(p.parse_args())
