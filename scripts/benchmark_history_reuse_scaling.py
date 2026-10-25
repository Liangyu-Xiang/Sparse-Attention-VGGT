#!/usr/bin/env python3
"""Microbenchmark whether more History reduces CURRENT mask-construction work.

This intentionally benchmarks the current selector structure, not a proposed
optimized implementation. Fine Value exploration size is held fixed while the
History pool is varied. If runtime does not decrease with reuse ratio, the
current implementation does not yet realize the intended efficiency property.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.unified_diagnostic_runtime import install
install()
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from saf3r.patch.reuse_residual_exploration import exploration_operating_points,joint_operating_points,budget_states


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--frames",type=int,default=128);p.add_argument("--height",type=int,default=28);p.add_argument("--width",type=int,default=37)
    p.add_argument("--retention",type=float,default=.20);p.add_argument("--window-groups",type=int,default=8);p.add_argument("--dual-iterations",type=int,default=16)
    p.add_argument("--reuse-ratios",type=float,nargs="+",default=[0,.25,.50,.75,.80]);p.add_argument("--warmup",type=int,default=5);p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args()
    if any(not 0<=r<1 for r in a.reuse_ratios):p.error("Invalid ratio")
    return a


def timed(fn,repeats,warmup):
    for _ in range(warmup):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats();s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record();fn();e.record();e.synchronize();samples.append(s.elapsed_time(e))
    return float(np.median(samples)),float(np.percentile(samples,90)),torch.cuda.max_memory_allocated()/2**20


def main():
    a=parse();out=a.output_dir;out.mkdir(parents=True,exist_ok=False);device=torch.device(a.device)
    n=a.frames*(a.height*a.width+5);specials=a.frames*5;budget=math.floor(a.retention*n)-specials;fine=budget;groups=math.ceil(n/32);g=min(a.window_groups,groups)
    torch.manual_seed(0);pi=torch.rand((g,fine),device=device);pi/=pi.sum(1,keepdim=True);rep_dist=torch.rand(fine,device=device)*.2
    population=torch.randint(1,9,(fine,),device=device).float();allowed=torch.zeros((g,fine),device=device,dtype=torch.bool);allowed.scatter_(1,pi.topk(max(1,fine//4),dim=1).indices,True)
    rows=[]
    for ratio in a.reuse_ratios:
        h=round(ratio*budget);hgain=torch.rand((g,h),device=device)/(max(h,1));hvalid=torch.ones((g,h),device=device,dtype=torch.bool)
        def explore():return exploration_operating_points(pi,rep_dist,population,allowed)
        ee,ec=explore()
        def solve():
            errors,costs=joint_operating_points(hgain,hvalid,ee,ec);return budget_states(errors,costs,budget,a.dual_iterations)
        ex_ms,ex_p90,_=timed(explore,a.repeats,a.warmup);sol_ms,sol_p90,peak=timed(solve,a.repeats,a.warmup)
        projected=(groups/g)*(ex_ms+sol_ms)
        rows.append(dict(reuse_ratio=ratio,history_items=h,fine_items=fine,window_groups=g,total_query_groups=groups,
            exploration_ms_window=ex_ms,solver_ms_window=sol_ms,total_ms_window=ex_ms+sol_ms,projected_all_windows_ms=projected,peak_allocated_mib=peak,
            desired_remaining_exploration_fraction=1-ratio))
    with (out/"scaling.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"shape":{"tokens":n,"budget":budget,"fine_items":fine,"query_groups":groups,"window_groups":g},"rows":rows,
        "interpretation":"Current code constructs all Fine exploration states for every ratio, then appends History to the dual solver. Therefore increasing History is not expected to reduce exploration_ms; an optimized reuse design should make exploration work shrink with the unresolved budget/information."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
