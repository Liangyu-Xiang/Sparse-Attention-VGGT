"""Run handoff Stage1 definitions unchanged on all idle, non-quarantined GPUs."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_recovery_guard import query_inventory,FAULT_UUID
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record

OUT=ROOT/'outputs/unified_next_round'
SCENES=('chess','fire','redkitchen')

def main():
    jobs=[]
    def add(name,script,args,path):
        jobs.append(dict(name=name,script=script,args=args,output=str(OUT/path),state='queued'))
    common=['--num-frames','2','--first-frame','0','--frame-gap','20']
    for s in SCENES:
        add('ceiling_'+s,'diagnose_history_reuse_ceiling.py',['--scene',s]+common+[
            '--target-retention','.20','--candidate-retentions','.20','.30','.40','.50','.80',
            '--layer-start','8','--layer-end','17'],Path('reuse_ceiling')/s)
    for s in SCENES:
        add('pipeline_'+s,'diagnose_history_pipeline.py',['--scene',s]+common+[
            '--target-retention','.20','--layer-start','9','--layer-end','16'],Path('history_pipeline')/s)
    for s in SCENES:
        add('compress_'+s,'diagnose_value_cluster_compressibility.py',['--scene',s]+common+[
            '--layer-start','10','--layer-end','15','--top-clusters','64'],Path('compressibility')/s)
    add('scaling','benchmark_history_reuse_scaling.py',['--frames','128','--reuse-ratios','0','.25','.50','.75','.80',
        '--warmup','5','--repeats','30'],Path('history_scaling'))
    for s in SCENES:
        add('granularity_'+s,'diagnose_query_group_granularity.py',['--scene',s]+common+[
            '--retention','.20','--group-sizes','32','16','8','4','1','--layer-start','9','--layer-end','15'],Path('query_granularity')/s)
    (OUT/'logs').mkdir(exist_ok=True)
    live={};handles=[]
    while any(j['state'] in ('queued','running') for j in jobs):
        for i,p in list(live.items()):
            if p.poll() is not None:
                jobs[i]['returncode']=p.returncode;jobs[i]['state']='complete' if p.returncode==0 else 'failed'
                if p.returncode:
                    out=Path(jobs[i]['output']);out.mkdir(parents=True,exist_ok=True)
                    write_quality(out,quality_record({},'Diagnostic failed; see logs/'+jobs[i]['name']+'.log'))
                del live[i]
        devices=query_inventory()
        occupied={jobs[i]['gpu_uuid'] for i in live}
        for d in devices:
            if d['uuid'] in occupied or d['uuid'].lower()==FAULT_UUID.lower() or d['memory_used_mib']>256:continue
            idx=next((i for i,j in enumerate(jobs) if j['state']=='queued'),None)
            if idx is None:break
            j=jobs[idx]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':d['uuid'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(ROOT)}
            log=(OUT/'logs'/(j['name']+'.log')).open('w');handles.append(log)
            cmd=[sys.executable,'scripts/'+j['script']]+j['args']+['--output-dir',j['output']]
            live[idx]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            j.update(state='running',gpu_uuid=d['uuid'],gpu_index=d['gpu'],pid=live[idx].pid)
        save(OUT/'progress.json',dict(jobs=jobs))
        if live or any(j['state']=='queued' for j in jobs):time.sleep(5)
    for f in handles:f.close()
    save(OUT/'execution_complete.json',dict(all_success=all(j['state']=='complete' for j in jobs),jobs=jobs))

if __name__=='__main__':main()
