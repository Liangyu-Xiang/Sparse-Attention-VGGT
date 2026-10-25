#!/usr/bin/env python3
"""Dense-oracle upper bound on adjacent-layer support reuse.

For each selected global layer, compute per-head/per-query patch-only attention on
an unmodified Dense VGGT trajectory. The target is the current layer's Top-rho
support. Candidate sets come only from the previous layer's Top-r support.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.analyze_sparse_support_transfer import (
    NUM_GLOBAL_LAYERS, NUM_HEADS, build_model, frame_path, patch_indices, qkv_from_module,
)
from saf3r.utils.data_utils import load_and_process_images_vggt


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",required=True); p.add_argument("--sequence",default="seq-01")
    p.add_argument("--first-frame",type=int,default=0); p.add_argument("--frame-gap",type=int,default=20)
    p.add_argument("--num-frames",type=int,default=2); p.add_argument("--target-retention",type=float,default=.20)
    p.add_argument("--candidate-retentions",type=float,nargs="+",default=[.20,.30,.40,.50,.80])
    p.add_argument("--layer-start",type=int,default=8); p.add_argument("--layer-end",type=int,default=17)
    p.add_argument("--data-root",type=Path,default=Path("workspace/benchmark_dataset/7scenes/7Scenes"))
    p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda")
    a=p.parse_args(); a.candidate_retentions=sorted(set(a.candidate_retentions))
    if not 0<a.target_retention<1 or any(not 0<x<1 for x in a.candidate_retentions): p.error("Invalid retention")
    if not 0<=a.layer_start<a.layer_end<NUM_GLOBAL_LAYERS: p.error("Invalid layer range")
    return a


def membership_recall(candidates,target,n):
    mask=torch.zeros((len(candidates),n),device=candidates.device,dtype=torch.bool)
    mask.scatter_(1,candidates.long(),True)
    return mask.gather(1,target.long()).float().mean(1)


def jaccard(a,b,n):
    mask=torch.zeros((len(a),n),device=a.device,dtype=torch.bool); mask.scatter_(1,a.long(),True)
    inter=mask.gather(1,b.long()).float().sum(1)
    return inter/(a.shape[1]+b.shape[1]-inter).clamp_min(1)


class Collector:
    def __init__(self,args): self.args=args; self.layer=0; self.previous=None; self.rows=[]; self.patch_count=None
    def __call__(self,module,input_args,input_kwargs,_output):
        layer=self.layer; self.layer+=1
        if layer<self.args.layer_start-1 or layer>self.args.layer_end: return
        q,k,_=qkv_from_module(module,input_args,input_kwargs)
        ids=patch_indices(q.shape[1],self.args.num_frames,q.device); q,k=q[:,ids],k[:,ids]
        n=q.shape[1]; self.patch_count=n
        max_ratio=max(max(self.args.candidate_retentions),self.args.target_retention)
        max_keep=max(1,round(max_ratio*n)); target_keep=max(1,round(self.args.target_retention*n))
        ranks=[]; saliencies=[]
        for h in range(NUM_HEADS):
            logits=q[h].float()@k[h].float().T*(q.shape[-1]**-.5); probs=logits.softmax(-1)
            rank=logits.topk(max_keep,dim=-1,sorted=True).indices; ranks.append(rank); saliencies.append(probs.mean(0))
            if self.previous is None or layer<=self.args.layer_start-1: continue
            target=rank[:,:target_keep]; target_mass=probs.gather(1,target).sum(1)
            prev_rank=self.previous["rank"][h].to(q.device); prev_sal=self.previous["saliency"][h].to(q.device)
            global_order=prev_sal.topk(max_keep,sorted=True).indices
            for ratio in self.args.candidate_retentions:
                kc=max(1,round(ratio*n)); cand=prev_rank[:,:kc]
                rec=membership_recall(cand,target,n); mass=probs.gather(1,cand).sum(1)
                gc=global_order[:kc][None].expand(n,-1)
                grec=membership_recall(gc,target,n); gmass=probs.gather(1,gc).sum(1)
                same=min(kc,target_keep)
                self.rows.append(dict(source_layer=layer-1,target_layer=layer,head=h,candidate_retention=ratio,
                    target_retention=self.args.target_retention,candidate_keys=kc,target_keys=target_keep,
                    recall=float(rec.mean()),attention_mass_coverage=float(mass.mean()),target_top_mass=float(target_mass.mean()),
                    mass_over_target_top_mass=float((mass/target_mass.clamp_min(1e-12)).mean()),
                    same_prefix_jaccard=float(jaccard(cand[:,:same],target[:,:same],n).mean()),
                    key_saliency_control_recall=float(grec.mean()),key_saliency_control_mass=float(gmass.mean())))
        self.previous={"rank":torch.stack(ranks).cpu(),"saliency":torch.stack(saliencies).cpu()}


def aggregate(rows):
    fields=("recall","attention_mass_coverage","target_top_mass","mass_over_target_top_mass","same_prefix_jaccard",
            "key_saliency_control_recall","key_saliency_control_mass")
    groups={}
    for r in rows:
        key=(r["source_layer"],r["target_layer"],r["candidate_retention"]); g=groups.setdefault(key,{x:[] for x in fields})
        for x in fields:g[x].append(r[x])
    return [dict(source_layer=s,target_layer=t,layer_pair=f"{s}→{t}",candidate_retention=ratio,
                 **{x:float(np.mean(v)) for x,v in g.items()}) for (s,t,ratio),g in sorted(groups.items())]


def save(args,c,frames):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=False); agg=aggregate(c.rows)
    for name,rows in (("per_head.csv",c.rows),("transitions.csv",agg)):
        with (out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/"summary.json").write_text(json.dumps({"scene":args.scene,"frame_indices":frames,"patch_count":c.patch_count,
        "target_retention":args.target_retention,"candidate_retentions":args.candidate_retentions,"transitions":agg,
        "interpretation":{"recall":"Current Top-rho key recall from previous per-query support.",
        "attention_mass_coverage":"Current dense patch-attention mass inside previous support.",
        "key_saliency_control":"Previous query-independent mean key saliency at identical candidate size."}},indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(11,5))
    for ratio in args.candidate_retentions:
        rr=[r for r in agg if r["candidate_retention"]==ratio]
        ax.plot([r["layer_pair"] for r in rr],[100*r["recall"] for r in rr],marker="o",label=f"prev Top {int(100*ratio)}%")
    ax.set_ylim(0,100);ax.set_ylabel("Current Top-20% recall (%)");ax.set_xlabel("Adjacent global layers")
    ax.grid(alpha=.2);ax.legend();ax.tick_params(axis="x",rotation=45);fig.tight_layout();fig.savefig(out/"reuse_ceiling.png",dpi=180);plt.close(fig)


def main():
    a=parse_args(); frames=[a.first_frame+i*a.frame_gap for i in range(a.num_frames)]
    paths=[frame_path(a.data_root,a.scene,a.sequence,f) for f in frames]; missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError("\n".join(missing))
    images=load_and_process_images_vggt([str(p) for p in paths]).to(a.device); model=build_model(a.checkpoint,a.device); c=Collector(a)
    hooks=[b.attn.register_forward_hook(c,with_kwargs=True) for b in model.aggregator.global_blocks]
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda")): model.aggregator(images.unsqueeze(0))
    finally:
        for h in hooks:h.remove()
    if c.layer!=NUM_GLOBAL_LAYERS or not c.rows: raise RuntimeError(f"Bad capture: layers={c.layer}, rows={len(c.rows)}")
    save(a,c,frames)

if __name__=="__main__": main()
