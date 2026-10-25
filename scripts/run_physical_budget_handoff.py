"""Strict staged physical H/Rep accounting handoff; no method tuning."""
import fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_all_edge_history_ablation as queue
from scripts.evaluate_attention_first_residual_value import save
OUT=ROOT/'outputs/physical_overlap_accounting';queue.OUT=OUT
CONFIG=ROOT/'configs/evaluation/reuse_residual_exploration.json'
FIELDS=('optimizer_cost_mean','counts_mean','physical_budget_utilization_mean','optimizer_physical_gap_mean',
        'unused_budget_mean','history_rep_candidate_overlap_mean','history_rep_merges_mean','history_exact_overlap_selected_mean')

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert '14 passed' in (OUT/'stage0/final_tests.log').read_text(),'Stage0 must pass first'
    sources=('saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py','scripts/evaluate_reuse_exploration.py',
        'scripts/diagnose_history_pipeline.py','scripts/unified_diagnostic_runtime.py','configs/evaluation/reuse_residual_exploration.json',
        'configs/evaluation/reuse_residual_exploration_no_physical_overlap.json')
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    save(OUT/'manifest.json',dict(upstream='1c8019197547c74c2e357748198f5d9246536150',
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,checkpoint=queue.CHECKPOINT,frames=128,retention=.2,warmup=1,repeats=3,
        quality_tasks=['pose','recon_unposed'],capture_disabled_for_quality=True,no_baseline_reruns=True))
    jobs=[]
    for mode,config in [('control','reuse_residual_exploration_no_physical_overlap.json'),('physical','reuse_residual_exploration.json')]:
        out=OUT/f'diagnostic_{mode}_chess'
        jobs.append(dict(scene=mode,state='queued',output=str(out),args=['scripts/evaluate_all_edge_history.py',
            '--method','reuse_explore','--scene','chess','--frames','2','--diagnostic','--capture',
            '--config',str(ROOT/'configs/evaluation'/config),'--checkpoint',queue.CHECKPOINT,'--output-dir',str(out)]))
    queue.dispatch(jobs,'accounting')
    rows=[];inputs=[]
    for mode in ('control','physical'):
        folder=OUT/f'diagnostic_{mode}_chess'
        inputs.append(json.loads((folder/'input.json').read_text()))
        d=pd.read_json(folder/'layers.json')
        assert len(d)==24 and (d.counts_mean<=d.patch_budget+1e-3).all()
        # Audit the captured physical support, not only the optimizer's records.
        import torch
        cap=torch.load(folder/'supports.pt',map_location='cpu')
        duplicate_groups=0;budget_exceeded=0
        for layer,c in cap.items():
            budget=int(d.loc[d.layer==int(layer),'patch_budget'].iloc[0])
            for w in c['windows']:
                for ids in w['ids']:
                    valid=ids[ids>=0]
                    duplicate_groups+=int(len(valid.unique())!=len(valid))
                    budget_exceeded+=int(len(valid)>budget)
        save(folder/'physical_support_audit.json',dict(duplicate_groups=duplicate_groups,budget_exceeded=budget_exceeded))
        if mode=='physical' and (duplicate_groups or budget_exceeded):raise RuntimeError('Physical supports fail uniqueness/budget audit')
        for stage,a,b in [('L0-L8',0,8),('L9-L16',9,16),('L17-L23',17,23)]:
            rows.append(dict(stage=stage,mode=mode,**d[d.layer.between(a,b)][list(FIELDS)].mean().to_dict()))
    assert inputs[0]==inputs[1],'A/B input mismatch'
    pd.DataFrame(rows).to_csv(OUT/'accounting_sanity.csv',index=False)
    save(OUT/'accounting_sanity.json',dict(rows=rows,matched_inputs=True,support_audit_passed=True))
    save(OUT/'progress.json',dict(phase='stage1_awaiting_acceptance',rows=rows))
    # No invented utilization threshold: review measured values against handoff.
    while not (OUT/'stage1_acceptance.json').exists():time.sleep(5)
    if not json.loads((OUT/'stage1_acceptance.json').read_text()).get('accepted'):
        raise RuntimeError('Stage1 acceptance rejected; quality must not run')
    jobs=[]
    for scene in ('chess','fire','redkitchen'):
        out=OUT/'pipeline'/scene
        jobs.append(dict(scene=scene,state='queued',output=str(out),args=['scripts/diagnose_history_pipeline.py',
            '--scene',scene,'--num-frames','2','--first-frame','0','--frame-gap','20','--target-retention','.20',
            '--layer-start','9','--layer-end','16','--config',str(CONFIG),'--checkpoint',queue.CHECKPOINT,'--output-dir',str(out)]))
    queue.dispatch(jobs,'pipeline')
    for scene in ('chess','fire','redkitchen'):
        d=pd.read_csv(OUT/'pipeline'/scene/'pipeline.csv')
        prev=d[d.support=='previous_full'].set_index('layer');cand=d[d.support=='history_candidates'].set_index('layer')
        assert (prev.candidate_density-cand.candidate_density).abs().max()<1e-5,'History bank mismatch'
    jobs=[]
    for scene in queue.SCENES:
        out=OUT/'quality128'/scene
        jobs.append(dict(scene=scene,state='queued',output=str(out),args=['scripts/evaluate_all_edge_history.py',
            '--method','reuse_explore','--scene',scene,'--frames','128','--warmup','1','--repeats','3',
            '--config',str(CONFIG),'--checkpoint',queue.CHECKPOINT,'--expected-input',str(queue.BASE/scene/'dense/input.json'),
            '--output-dir',str(out)]))
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},'Frozen implementation changed'
    queue.dispatch(jobs,'quality')
    subprocess.run([sys.executable,'scripts/report_physical_budget_handoff.py'],cwd=ROOT,check=True)
    save(OUT/'completion.json',dict(all_seven_complete=True,stages_0_through_6_complete=True))
    # This scoped directory contains only requested experimental deliverables.
    subprocess.run(['git','add','.gitignore','scripts/run_physical_budget_handoff.py',
        'scripts/report_physical_budget_handoff.py','outputs/physical_overlap_accounting'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-qm','Archive complete physical History-Rep accounting ablation and analysis'],cwd=ROOT,check=True)
    subprocess.run(['git','push','sparse-attention-vggt','HEAD:refs/heads/refactor/unified-reuse-exploration'],cwd=ROOT,check=True)
    save(OUT/'publication_status.json',dict(pushed=True,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(OUT/'blocked.json',dict(reason=f'{type(exc).__name__}: {exc}'));raise
