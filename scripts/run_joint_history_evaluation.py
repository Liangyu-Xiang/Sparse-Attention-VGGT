"""Preflight, seven-scene quality and fresh matched Dense timings."""
import fcntl,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_all_edge_history_ablation as queue
from scripts.evaluate_attention_first_residual_value import save
OUT=ROOT/'outputs/joint_history_06c5e0c';queue.OUT=OUT

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert '15 passed' in (OUT/'stage0/tests.log').read_text()
    sources=('saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py',
        'scripts/evaluate_reuse_exploration.py','scripts/unified_diagnostic_runtime.py','configs/evaluation/reuse_residual_exploration.json')
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    save(OUT/'manifest.json',dict(upstream='06c5e0c760b34b34c3f1a715a99417dd0406fd76',
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,production_unmodified=True,scenes=queue.SCENES,frames=128,retention=.2,
        warmup=1,repeats=3,paired_dense_timing=True,same_uuid_per_pair=True,no_test_truth_tuning=True,
        quality_tasks=['pose','recon_unposed'],fresh_candidate_slots_plan_implemented=False))
    preflight=OUT/'preflight'
    queue.dispatch([dict(scene='preflight',state='queued',output=str(preflight),args=[
        'scripts/evaluate_all_edge_history.py','--method','reuse_explore','--scene','heads','--frames','2',
        '--warmup','1','--repeats','1','--diagnostic','--checkpoint',queue.CHECKPOINT,'--output-dir',str(preflight)])],'preflight')
    assert json.loads((preflight/'status.json').read_text())['stage']=='diagnostic_complete'
    jobs=[dict(scene=s,state='queued',output=str(OUT/'quality128'/s),args=['scripts/run_joint_pair_worker.py',
        '--scene',s,'--output-dir',str(OUT/'quality128'/s)]) for s in queue.SCENES]
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    queue.dispatch(jobs,'quality')
    subprocess.run([sys.executable,'scripts/report_joint_history_evaluation.py'],cwd=ROOT,check=True)
    save(OUT/'completion.json',dict(all_seven_quality_complete=True,paired_timings_complete=True,report_complete=True))
    # Do not push externally: this request authorizes evaluation, not remote writes.

if __name__=='__main__':
    try:main()
    except Exception as e:save(OUT/'blocked.json',dict(reason=f'{type(e).__name__}: {e}'));raise
