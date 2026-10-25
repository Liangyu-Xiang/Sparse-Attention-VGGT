"""Strict handoff stages0/1 gate before seven-scene quality; idle GPUs only."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_recovery_guard import query_inventory,FAULT_UUID
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record

OUT=ROOT/'outputs/all_edge_history_no_coverage'
BASE=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128'
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
CHECKPOINT='/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt'

def dispatch(jobs,phase):
    live={};handles=[];waiting={}
    while any(j['state'] in ('queued','running') for j in jobs):
        for i,p in list(live.items()):
            if p.poll() is not None:
                j=jobs[i];j['returncode']=p.returncode;j['state']='complete' if p.returncode==0 else 'failed'
                if j['state']=='complete' and phase=='quality':
                    path=Path(j['output'])/'result.json'
                    if not path.exists() or not json.loads(path.read_text())['quality']['complete']:j['state']='failed'
                del live[i]
            else:
                try:
                    state=next(t for t in Path(f'/proc/{p.pid}/status').read_text().splitlines() if t.startswith('State:')).split()[1]
                    wchan=Path(f'/proc/{p.pid}/wchan').read_text().strip()
                except OSError:continue
                if state=='D':waiting.setdefault(i,time.monotonic())
                else:waiting.pop(i,None)
                limit=900 if wchan=='wait_on_page_bit_common' else 120
                if i in waiting and time.monotonic()-waiting[i]>limit:
                    raise RuntimeError('Sustained kernel wait; no further GPU dispatch: '+jobs[i]['scene'])
        if any(j['state']=='failed' for j in jobs):
            save(OUT/'progress.json',dict(phase=phase,jobs=jobs))
            # Already running siblings are allowed to finish; no new work is dispatched.
            if live:time.sleep(5);continue
            raise RuntimeError('A '+phase+' job failed; inspect logs; later stages not launched')
        busy={jobs[i]['uuid'] for i in live}
        for d in query_inventory():
            if d['uuid'] in busy or d['uuid'].lower()==FAULT_UUID.lower() or d['memory_used_mib']>256:continue
            idx=next((i for i,j in enumerate(jobs) if j['state']=='queued'),None)
            if idx is None:break
            j=jobs[idx];env={**os.environ,'CUDA_VISIBLE_DEVICES':d['uuid'],'PYTHONPATH':str(ROOT),
                'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
            f=(OUT/'logs'/f"{phase}_{j['scene']}.log").open('w');handles.append(f)
            live[idx]=subprocess.Popen([sys.executable]+j['args'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            j.update(state='running',uuid=d['uuid'],gpu=d['gpu'],pid=live[idx].pid)
        save(OUT/'progress.json',dict(phase=phase,jobs=jobs))
        if live or any(j['state']=='queued' for j in jobs):time.sleep(5)
    for h in handles:h.close()
    save(OUT/f'{phase}_complete.json',dict(jobs=jobs,all_success=True))

def main():
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tests=(OUT/'stage0/final_tests.log').read_text()
    if '13 passed' not in tests:raise RuntimeError('Stage0 final test suite must pass before Stage1')
    files=('saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py','scripts/unified_diagnostic_runtime.py','scripts/evaluate_reuse_exploration.py','scripts/evaluate_all_edge_history.py')
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    save(OUT/'manifest.json',dict(upstream='1a7bb41d955da284eded195a2e0b11faf49c2ec3',
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,checkpoint=CHECKPOINT,retention=.2,frames=128,warmup=1,repeats=3,
        quality_tasks=['pose','recon_unposed'],no_baseline_reruns=True,production_matches_upstream=True))
    jobs=[]
    for s in ('chess','fire','redkitchen'):
        out=OUT/'pipeline'/s
        jobs.append(dict(scene=s,state='queued',output=str(out),args=['scripts/diagnose_history_pipeline.py','--scene',s,
            '--num-frames','2','--first-frame','0','--frame-gap','20','--target-retention','.20',
            '--layer-start','9','--layer-end','16','--checkpoint',CHECKPOINT,'--output-dir',str(out)]))
    dispatch(jobs,'pipeline')
    import pandas as pd
    checks=[]
    for s in ('chess','fire','redkitchen'):
        d=pd.read_csv(OUT/'pipeline'/s/'pipeline.csv')
        p=d[d.support=='previous_full'].set_index('layer')
        h=d[d.support=='history_candidates'].set_index('layer')
        if len(h)!=8 or (h.candidate_density<=0).any():raise RuntimeError('History bank empty; stop before quality')
        for field in ('candidate_density','candidate_keys','recall','attention_mass_coverage'):
            if (p[field]-h[field]).abs().max()>1e-5:raise RuntimeError('History does not match previous executed support: '+s+'/'+field)
        checks.append(dict(scene=s,all_eight_layers_match_previous_support=True))
    save(OUT/'pipeline_sanity.json',dict(passed=True,checks=checks))
    # Stage2 is authorized only after all Stage1 sanity checks pass.
    jobs=[]
    for s in SCENES:
        out=OUT/'quality128'/s
        jobs.append(dict(scene=s,state='queued',output=str(out),args=['scripts/evaluate_all_edge_history.py',
            '--method','reuse_explore','--scene',s,'--frames','128','--warmup','1','--repeats','3',
            '--checkpoint',CHECKPOINT,'--expected-input',str(BASE/s/'dense/input.json'),'--output-dir',str(out)]))
    if hashes!={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}:raise RuntimeError('Frozen source changed')
    dispatch(jobs,'quality')
    subprocess.run([sys.executable,'scripts/report_all_edge_history_ablation.py'],cwd=ROOT,check=True)
    save(OUT/'completion.json',dict(all_seven_complete=True,stages_0_through_5_complete=True))
    # Commit only the explicitly scoped deliverables, never local datasets/weights.
    subprocess.run(['git','add','scripts/run_all_edge_history_ablation.py','scripts/report_all_edge_history_ablation.py',
        'outputs/all_edge_history_no_coverage/README.md','outputs/all_edge_history_no_coverage/stage0',
        'outputs/all_edge_history_no_coverage/logs','outputs/all_edge_history_no_coverage/pipeline',
        'outputs/all_edge_history_no_coverage/quality128','outputs/all_edge_history_no_coverage/manifest.json',
        'outputs/all_edge_history_no_coverage/pipeline_complete.json','outputs/all_edge_history_no_coverage/quality_complete.json',
        'outputs/all_edge_history_no_coverage/pipeline_sanity.json','outputs/all_edge_history_no_coverage/progress.json',
        'outputs/all_edge_history_no_coverage/layerwise.csv','outputs/all_edge_history_no_coverage/layerwise.json',
        'outputs/all_edge_history_no_coverage/quality128/layerwise_per_scene.csv',
        'outputs/all_edge_history_no_coverage/stage_averages.csv','outputs/all_edge_history_no_coverage/comparison.md',
        'outputs/all_edge_history_no_coverage/comparison.json','outputs/all_edge_history_no_coverage/analysis.md',
        'outputs/all_edge_history_no_coverage/completion.json'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-qm','Archive complete all-edge History no-coverage ablation and causal analysis'],cwd=ROOT,check=True)
    subprocess.run(['git','push','sparse-attention-vggt','HEAD:refs/heads/refactor/unified-reuse-exploration'],cwd=ROOT,check=True)
    save(OUT/'publication_status.json',dict(pushed=True,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(OUT/'blocked.json',dict(reason=f'{type(exc).__name__}: {exc}'))
        raise
