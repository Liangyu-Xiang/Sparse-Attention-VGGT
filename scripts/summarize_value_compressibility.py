#!/usr/bin/env python3
"""Rank cheap Rep/Exact signals against diagnostic Value-contribution error."""
import argparse,csv,json
from pathlib import Path
import numpy as np

METRICS=("value_rel_l2","value_cosine_dist","key_rel_l2","key_cosine_dist","qk_logit_std","rep_affinity_error")

def rank(x):
    order=np.argsort(x,kind="stable");r=np.empty(len(x),dtype=float);r[order]=np.arange(len(x));return r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("inputs",type=Path,nargs="+");p.add_argument("--output",type=Path,required=True);p.add_argument("--positive-fraction",type=float,default=.2);a=p.parse_args()
    rows=[]
    for path in a.inputs:
        with path.open() as f:rows.extend(csv.DictReader(f))
    if not rows:raise RuntimeError("No rows")
    y=np.asarray([float(r["mass_preserving_value_error"]) for r in rows]);importance=np.asarray([float(r["importance"]) for r in rows]);weighted=y*importance
    k=max(1,round(a.positive_fraction*len(rows)));truth=np.argsort(-weighted,kind="stable")[:k];truth_set=set(truth.tolist())
    result={}
    for m in METRICS:
        x=np.asarray([float(r[m]) for r in rows]);pred=np.argsort(-x,kind="stable")[:k];hits=len(truth_set.intersection(pred.tolist()))
        result[m]={"rank_correlation_with_weighted_error":float(np.corrcoef(rank(x),rank(weighted))[0,1]),"precision_at_fraction":hits/k,"recall_at_fraction":hits/k}
    best=max(result,key=lambda m:result[m]["precision_at_fraction"])
    summary={"rows":len(rows),"positive_fraction":a.positive_fraction,"target":"importance * mass_preserving_value_error","metrics":result,"best_by_precision":best,
             "note":"This ranks diagnostic signals only; choose an end-to-end ablation criterion from training/diagnostic clips, never from 7Scenes test quality."}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
