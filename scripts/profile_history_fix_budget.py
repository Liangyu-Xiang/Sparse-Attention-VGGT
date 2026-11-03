"""Separate solver and repair timings without changing model math or files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_light_reuse import build
from saf3r.utils.data_utils import load_images
from saf3r.patch.fused_shared_av import stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=ROOT/'configs/evaluation/reuse_light_single_probe_repaired.json'
    model,state=build(config)
    images=load_images(model,'vggt',json.loads((ROOT/'outputs/fastest_single_probe_snapshot_20261103/timing/input.json').read_text())['image_paths']).cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        raw=model(images)
    torch.cuda.synchronize()
    del raw
    import saf3r.triton.reuse_budget_graph as solver
    import saf3r.patch.reuse_budget_repair as repair
    original_solver=solver.budget
    original_repair=repair.repair_underfilled

    def measured_solver(*a,**kw):
        with stage(state,'detail_budget_solver'):
            return original_solver(*a,**kw)

    def measured_repair(*a,**kw):
        with stage(state,'detail_budget_repair'):
            return original_repair(*a,**kw)

    solver.budget=measured_solver
    repair.repair_underfilled=measured_repair
    state.profiling=True
    state.profile_events=[]
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            raw=model(images)
        torch.cuda.synchronize()
        wall=time.perf_counter()-begin
    finally:
        solver.budget=original_solver
        repair.repair_underfilled=original_repair
    assert len(state.records)==24 and torch.isfinite(raw['depth']).all() and torch.isfinite(raw['pose_enc']).all()
    groups={}
    for layer,name,start,end in state.profile_events:
        groups[name]=groups.get(name,0.)+start.elapsed_time(end)/1000
    parent=groups.pop('light_budget')
    children=sum(groups[k] for k in ['detail_budget_solver','detail_budget_repair'])
    assert children<=parent+1e-5
    groups['detail_budget_other']=parent-children
    groups['unattributed']=wall-sum(groups.values())
    result=dict(complete=True,wall_seconds=wall,exclusive_seconds=groups,
        inclusive_budget_seconds=parent,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scope='Independent one-warmup/one-event-profile diagnostic, not benchmark median; nested budget children replace parent in exclusive sum',
        config=json.loads(config.read_text()),instrumentation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},
        quality_reason='Timing-only event instrumentation; pose/recon quality not evaluated in this diagnostic',
        dense_speedup=None,dense_reason='No unchanged Dense rerun authorized')
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
