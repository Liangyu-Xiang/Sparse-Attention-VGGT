"""Frozen full ETH3D quality run, GPU4/5/6 queue plus independent CPU fusion."""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/eth3d_layout_fixed_full_20261105'
SCENES=('courtyard','electro','kicker','pipes','relief','delivery_area','facade','office','playground','relief_2','terrains')
FIELDS=('auc03','auc30','acc','comp','cd','f1')
BEFORE=ROOT/'outputs/integrated_quality_speed_20261105/final_runs/eth3d'


def read(path):return json.loads(path.read_text())
def save(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(out):
    out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir();(out/'runs').mkdir()
    config=out/'frozen_config.json'
    config.write_bytes((ROOT/'outputs/integrated_quality_speed_20261105/frozen_configs/main.json').read_bytes())
    sources=[p for folder in ('saf3r','evaluation') for p in (ROOT/folder).rglob('*.py')]
    sources += [ROOT/'scripts/evaluate_zdrop_history_states.py',ROOT/'scripts/evaluate_saved_reuse_predictions.py',ROOT/'scripts/launch_local_evaluation.py',ROOT/'scripts/quality_metrics.py']
    counts={}
    for scene in SCENES:
        before=read(BEFORE/scene/'main/result.json')
        assert before['quality']['complete'] and before['dataset']=='eth3d'
        save(out/'inputs'/f'{scene}.json',before['input']);counts[scene]=before['frames']
    plan=dict(method='main_layout_fixed',model_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scenes=list(SCENES),frames=counts,max_frames=128,sampling='uniform',seed=0,config=str(config),config_sha256=sha(config),
        frozen_before_dataset_evaluation=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        tasks=['pose','recon_unposed'],gpus=[4,5,6],cpu_quality_workers=3,
        timing_scope='ETH3D quality only: one cold full forward, no warmed speed comparison',
        control_policy='Existing complete before_main results only; no unchanged control model reruns',
        intervention='Only Exact candidate mask layout/stride repair; no oracle, retention change or quality retuning')
    save(out/'frozen_plan.json',plan)
    return plan


def verify(plan):
    assert sha(Path(plan['config']))==plan['config_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name


def free_gpus():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={line.split(',')[1].strip():int(line.split(',')[0]) for line in rows.splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={mapping[line.split(',')[0].strip()] for line in apps.splitlines() if line.strip()}
    return sorted({4,5,6}-busy)


def gpu_forward(out,plan,scene,gpu):
    verify(plan);folder=out/'runs'/scene
    cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),
        '--dataset','eth3d','--scene',scene,'--frames','128','--method','zdrop_history_states',
        '--config',plan['config'],'--expected-input',str(out/'inputs'/f'{scene}.json'),
        '--output-dir',str(folder),'--warmup','0','--repeats','1','--defer-quality']
    print('GPU_START',gpu,scene,flush=True)
    code=subprocess.call([sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),
        '--monitor',str(out/(scene+'_gpu.json')),'--log',str(out/(scene+'_forward.log')),'--',*cmd],cwd=ROOT)
    print('GPU_END',gpu,scene,code,flush=True)
    if code:raise RuntimeError(f'{scene} GPU inference failed; retained log/output, returncode={code}')


def leased_batch_worker(out,plan,gpu):
    """Claim scenes only after a permitted GPU lease is acquired."""
    save(out/f'gpu{gpu}_batch_started.json',dict(pid=os.getpid(),gpu=gpu))
    while True:
        with (out/'batch_queue.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=read(out/'batch_queue.json')
            if not state['pending']:return
            scene=state['pending'].pop(0);state['claims'][scene]=dict(gpu=gpu,stage='forward')
            save(out/'batch_queue.json',state)
        verify(plan)
        cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),
            '--dataset','eth3d','--scene',scene,'--frames','128','--method','zdrop_history_states',
            '--config',plan['config'],'--expected-input',str(out/'inputs'/f'{scene}.json'),
            '--output-dir',str(out/'runs'/scene),'--warmup','0','--repeats','1','--defer-quality']
        print('GPU_BATCH_START',gpu,scene,flush=True)
        with (out/(scene+'_forward.log')).open('w') as log:
            code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with (out/'batch_queue.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=read(out/'batch_queue.json')
            state['claims'][scene].update(stage='complete' if code==0 else 'failed',returncode=code)
            save(out/'batch_queue.json',state)
        print('GPU_BATCH_END',gpu,scene,code,flush=True)
        if code:raise SystemExit(code)


def batch_campaign(out,plan):
    import psutil
    pending=[s for s in sorted(SCENES,key=lambda s:plan['frames'][s],reverse=True) if not (out/'runs'/s/'deferred_forward.json').exists()]
    save(out/'batch_queue.json',dict(pending=pending,claims={}))
    processes={};cancelled=set();cpu_jobs={};submitted=set();failed=[]
    for gpu in (4,5,6):
        cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),
            '--monitor',str(out/f'gpu{gpu}_batch_monitor.json'),'--log',str(out/f'gpu{gpu}_batch.log'),'--',
            sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--gpu-worker',str(gpu)]
        processes[gpu]=subprocess.Popen(cmd,cwd=ROOT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            for scene in SCENES:
                folder=out/'runs'/scene
                if (folder/'deferred_forward.json').exists() and not (folder/'result.json').exists() and scene not in submitted:
                    cpu_jobs[pool.submit(cpu_quality,out,plan,scene)]=scene;submitted.add(scene)
            for future,scene in list(cpu_jobs.items()):
                if future.done():
                    del cpu_jobs[future]
                    try:future.result()
                    except Exception as exc:failed.append(dict(scene=scene,phase='quality',error=str(exc)))
            state=read(out/'batch_queue.json')
            all_forward=all((out/'runs'/s/'deferred_forward.json').exists() for s in SCENES)
            if all_forward:
                for gpu,proc in processes.items():
                    if proc.poll() is not None:continue
                    # A lease still waiting for another task has no child and
                    # no claimed scenes. Cancel only this unused own waiter.
                    if (out/f'gpu{gpu}_batch_started.json').exists():continue
                    if not any(v['gpu']==gpu for v in state['claims'].values()):
                        try:children=psutil.Process(proc.pid).children()
                        except psutil.NoSuchProcess:children=[]
                        if not children:proc.terminate();cancelled.add(gpu)
            exited=all(p.poll() is not None for p in processes.values())
            if exited and not cpu_jobs:break
            save(out/'progress.json',dict(pending=state['pending'],gpu_active=[dict(scene=s,**v) for s,v in state['claims'].items() if v['stage']=='forward'],
                cpu_active=list(cpu_jobs.values()),completed=[s for s in SCENES if (out/'runs'/s/'result.json').exists()],failures=failed,
                waiting_gpu_leases=[gpu for gpu,p in processes.items() if p.poll() is None and not (out/f'gpu{gpu}_batch_started.json').exists()]))
            time.sleep(2)
    for gpu,proc in processes.items():
        code=proc.wait()
        if code and gpu not in cancelled:failed.append(dict(gpu=gpu,phase='batch_forward',returncode=code))
    state=read(out/'batch_queue.json')
    for scene,claim in state['claims'].items():
        monitor=read(out/f"gpu{claim['gpu']}_batch_monitor.json")
        save(out/(scene+'_gpu.json'),dict(monitor,scene=scene,scope='Exclusive GPU batch lease; all child forward processes monitored',batch_gpu=claim['gpu']))
    save(out/'batch_execution.json',dict(claims=state['claims'],cancelled_unused_own_gpu_waiters=sorted(cancelled),failures=failed))
    if failed:raise RuntimeError(str(failed))
    report(out,plan)


def cpu_quality(out,plan,scene):
    verify(plan);folder=out/'runs'/scene
    print('QUALITY_START',scene,flush=True)
    cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_saved_reuse_predictions.py'),
        '--scene',scene,'--run',str(folder),'--images',str(folder/'images.pt')]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    with (out/(scene+'_quality.log')).open('w') as log:
        code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'{scene} CPU quality failed; retained log/output, returncode={code}')
    common=read(folder/'deferred_forward.json');quality=read(folder/'saved_quality/result.json')
    assert common['input']==quality['input']
    for key in ('f1_threshold_m','downsample_m'):assert common['quality_protocol'][key]==quality['quality_protocol'][key]
    record=quality['quality'];record['speedup_unavailable']='Speed requested only on7Scenes; no paired Dense ETH3D warmed timing'
    result=dict(common,quality=record,metrics=quality['metrics'],quality_errors=quality['errors'],
        cpu_evaluation_seconds=quality['cpu_evaluation_seconds'],quality_execution='CPU saved-prediction pose/recon_unposed; same repository evaluation functions')
    save(folder/'result.json',result);save(folder/'quality.json',record)
    from scripts.quality_metrics import quality_markdown
    (folder/'quality.md').write_text(quality_markdown(record))
    save(folder/'status.json',dict(stage='complete' if record['complete'] else 'quality_incomplete'))
    print('QUALITY_END',scene,record['values'],flush=True)
    if not record['complete']:raise RuntimeError(f'{scene}: incomplete six-metric quality: {record["unavailable"]}')


def report(out,plan):
    rows=[];audits=[];runtimes=[]
    grouped={'before_main':[],'layout_fixed':[]}
    for scene in SCENES:
        before=read(BEFORE/scene/'main/result.json');after=read(out/'runs'/scene/'result.json')
        assert before['quality']['complete'] and after['quality']['complete'],scene
        checks={key:before['input'][key]==after['input'][key] for key in ('image_paths','frame_indices','shape','checkpoint_sha256')}
        checks['quality_protocol']=before['quality_protocol']==after['quality_protocol']
        checks['unchanged_evaluation']=all(after['source_sha256'].get(k)==v for k,v in before['source_sha256'].items() if k.startswith('evaluation/'))
        checks['frozen_model_and_evaluation']=all(after['source_sha256'].get(k)==v for k,v in plan['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
        checks['frozen_config']=all(after['config'].get(k)==v for k,v in read(Path(plan['config'])).items())
        monitor=read(out/(scene+'_gpu.json'));checks['gpu_scope']=monitor['gpu'] in (4,5,6)
        checks['hardware_uuid']=after['hardware']['gpu_uuid'].removeprefix('GPU-')==monitor['gpu_uuid'].removeprefix('GPU-')
        checks['gpu_isolated']=monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
        assert all(checks.values()),(scene,checks);audits.append(dict(scene=scene,checks=checks))
        for name,result in [('before_main',before),('layout_fixed',after)]:
            v=result['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
            grouped[name].append(v);rows.append(dict(scene=scene,method=name,frames=result['frames'],**v))
        runtime=after['timing'];layers=read(out/'runs'/scene/'layers.json');assert len(layers)==24
        runtimes.append(dict(scene=scene,gpu=monitor['gpu'],gpu_uuid=monitor['gpu_uuid'],frames=after['frames'],
            cold_full_forward_seconds=runtime['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in runtime['samples']),
            warmup=0,repeats=1,mean_actual_key_density=sum(l['actual_unique_density'] for l in layers)/24,
            speedup_vs_dense=None,speedup_unavailable='ETH3D quality-only run; no paired warmed Dense timing'))
    macro={name:{key:sum(v[key] for v in values)/11 for key in FIELDS} for name,values in grouped.items()}
    summary=dict(dataset='eth3d',complete=True,scenes=11,per_scene=rows,macro=macro,audits=audits,runtime=runtimes,
        units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'),
        speedup_vs_dense=None,speedup_unavailable='Speed requested only for7Scenes; no paired warmed Dense ETH3D timing')
    save(out/'summary.json',summary)
    with (out/'per_scene.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    def cells(v):return ' | '.join(f"{v[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS)
    lines=['# 候选掩码布局修复：完整ETH3D评测','',
        '11/11场景；每场景最多128帧、uniform seed0；固定同权重、精确帧表与预处理、Pose协议、unposed融合/对齐和重建阈值。',
        '范围沿用仓库此前的ETH3D协议：terrace/meadow按既有配置排除；本轮不新增场景或图像过滤。',
        '所有场景均为本批重新推理和评测。只使用GPU4/5/6；CPU点云评测与后续GPU推理独立排队。',
        '仅修复Exact候选掩码布局读取；没有oracle推理，也未调整保留率、聚类或状态损失定义。',
        'AUC/F1为百分数，ACC/Comp/CD为米；CD=(ACC+Comp)/2，非平方距离。宏平均为11场景等权平均。', '',
        '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
    for name,values in macro.items():lines.append('| '+name+' | '+cells(values)+' |')
    lines+=['','## 逐场景六指标','',
        '| Scene | Frames | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['scene']+' | '+str(r['frames'])+' | '+r['method']+' | '+cells(r)+' |')
    protocol=read(out/'runs'/SCENES[0]/'result.json')['quality_protocol']
    config=read(Path(plan['config']));rates=config['layer_retentions']
    lines+=['','## 协议、稀疏率与运行记录','',
        f"重建阈值{protocol['f1_threshold_m']}m，下采样{protocol['downsample_m']}m；voxel={protocol['voxel_length_m']}m，sdf_trunc={protocol['sdf_trunc_m']}m，mesh采样{protocol['mesh_sample_points']}点。",
        f"逐层保留率为{rates}，均值{sum(rates)/24:.4f}；fine cluster fraction={config['cluster_fraction']}。方法不使用SAF3R Top-K/profile。",
        '保留率是Key预算上限；逐层实际密度另存各run的layers.json。未使用7Scenes测试真值调参。',
        'ETH3D只做质量评估（0暖机、1次完整前向）。下列时间为冷前向记录，不作速度比较；相对Dense暖机加速全部N/A，原因是用户仅要求7Scenes速度。', '',
        '| Scene | GPU | Cold forward s | Peak allocated GiB | Actual key density |','|---|---:|---:|---:|---:|']
    for r in runtimes:lines.append(f"| {r['scene']} | {r['gpu']} | {r['cold_full_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['mean_actual_key_density']:.6f} |")
    lines+=['','before_main来自既有完整本地测量，未重跑未变化的对照。完整输入、权重SHA、代码快照、raw metrics、每场景六指标与GPU监控均已保存。与论文公开结果无混用。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    save(out/'progress.json',dict(stage='complete',pending=[],gpu_active=[],cpu_active=[],completed=list(SCENES),failures=[]))
    print('FULL_ETH3D_COMPLETE',json.dumps(macro),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--report-only',action='store_true');p.add_argument('--batch-queue',action='store_true')
    p.add_argument('--gpu-worker',type=int,choices=[4,5,6]);a=p.parse_args();out=a.output.resolve()
    sys.path.insert(0,str(ROOT))
    plan=read(out/'frozen_plan.json') if out.exists() else freeze(out)
    verify(plan)
    if a.gpu_worker is not None:leased_batch_worker(out,plan,a.gpu_worker);return
    if a.batch_queue:batch_campaign(out,plan);return
    if a.report_only:report(out,plan);return
    pending=[s for s in sorted(SCENES,key=lambda s:plan['frames'][s],reverse=True) if not (out/'runs'/s/'deferred_forward.json').exists()]
    cpu_pending=[s for s in SCENES if (out/'runs'/s/'deferred_forward.json').exists() and not (out/'runs'/s/'result.json').exists()]
    gpu_jobs={};cpu_jobs={};failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as gpu_pool,concurrent.futures.ThreadPoolExecutor(max_workers=3) as cpu_pool:
        for scene in cpu_pending:cpu_jobs[cpu_pool.submit(cpu_quality,out,plan,scene)]=scene
        while pending or gpu_jobs or cpu_jobs:
            for future,(gpu,scene) in list(gpu_jobs.items()):
                if not future.done():continue
                del gpu_jobs[future]
                try:future.result();cpu_jobs[cpu_pool.submit(cpu_quality,out,plan,scene)]=scene
                except Exception as exc:failed.append(dict(scene=scene,phase='gpu',error=str(exc)))
            for future,scene in list(cpu_jobs.items()):
                if not future.done():continue
                del cpu_jobs[future]
                try:future.result()
                except Exception as exc:failed.append(dict(scene=scene,phase='quality',error=str(exc)))
            used={gpu for gpu,scene in gpu_jobs.values()}
            for gpu in free_gpus():
                if not pending:break
                if gpu in used:continue
                scene=pending.pop(0);gpu_jobs[gpu_pool.submit(gpu_forward,out,plan,scene,gpu)]=(gpu,scene)
            if pending and not gpu_jobs:
                # Join one lease queue when all devices are occupied. Polling
                # alone can miss short release gaps between other queued jobs.
                # Leave other scenes unclaimed for whichever GPU frees first.
                scene=pending.pop(0)
                gpu_jobs[gpu_pool.submit(gpu_forward,out,plan,scene,5)]=(5,scene)
            save(out/'progress.json',dict(pending=pending,gpu_active=[dict(gpu=g,scene=s) for g,s in gpu_jobs.values()],
                cpu_active=list(cpu_jobs.values()),completed=[s for s in SCENES if (out/'runs'/s/'result.json').exists()],failures=failed))
            time.sleep(2)
    if failed:raise RuntimeError(str(failed))
    report(out,plan)


if __name__=='__main__':main()
