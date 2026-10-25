#!/usr/bin/env python3
"""Locate where the current History pipeline loses reusable Dense Top-K edges.

Runs the CURRENT unified sparse method on a small 7Scenes clip with capture=True.
For L9-L16 it computes a diagnostic full patch QK matrix on the SAME sparse
trajectory before each patched attention layer, then compares four group-shared
supports against the current per-head/per-query Dense Top-rho target:

1. previous layer full sparse support;
2. current layer History candidate bank before budget selection;
3. current layer selected History edges;
4. current layer final sparse support.

This script does not alter the selector and should not be used as an accelerator.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from scripts.analyze_sparse_support_transfer import build_model, frame_path
from saf3r.utils.data_utils import load_and_process_images_vggt
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig, install_reuse_residual_exploration
from saf3r.patch.region_history_attention import prepare_qkv


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",required=True);p.add_argument("--sequence",default="seq-01")
    p.add_argument("--first-frame",type=int,default=0);p.add_argument("--frame-gap",type=int,default=20);p.add_argument("--num-frames",type=int,default=2)
    p.add_argument("--target-retention",type=float,default=.20);p.add_argument("--layer-start",type=int,default=9);p.add_argument("--layer-end",type=int,default=16)
    p.add_argument("--data-root",type=Path,default=Path("workspace/benchmark_dataset/7scenes/7Scenes"))
    p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    p.add_argument("--config",type=Path,default=Path("configs/evaluation/reuse_residual_exploration.json"))
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args()
    if not 0<a.target_retention<1 or not 1<=a.num_frames<=8 or not 1<=a.layer_start<=a.layer_end<24:p.error("Invalid arguments")
    return a


def concatenate_windows(captured,key,groups):
    parts=[w[key] for w in captured["windows"]]
    if not parts:return torch.empty((groups,0),dtype=torch.int32)
    out=torch.cat(parts,0)
    if len(out)!=groups:raise RuntimeError(f"{key}: expected {groups} groups, got {len(out)}")
    return out


def support_metrics(probs,target,candidate_abs,query_abs,patch_abs):
    """probs [H,Q,Np], target [H,Q,K], candidate_abs [G,W]."""
    device=probs.device; npatch=len(patch_abs); total_tokens=int(max(int(patch_abs.max()),int(query_abs.max())))+1
    lookup=torch.full((total_tokens,),-1,device=device,dtype=torch.long);lookup[patch_abs.long()]=torch.arange(npatch,device=device)
    cand=candidate_abs.to(device).long(); valid=(cand>=0)&(cand<total_tokens)
    local=torch.full_like(cand,-1);local[valid]=lookup[cand[valid]];valid&=local>=0
    group=(query_abs//32).long(); q_cand=local[group];q_valid=valid[group]
    counts=torch.zeros((len(query_abs),npatch),device=device,dtype=torch.int32)
    safe=q_cand.clamp_min(0);counts.scatter_add_(1,safe,q_valid.int());mask=counts>0
    recall=mask[None].expand(probs.shape[0],-1,-1).gather(2,target).float().mean(2)
    mass=(probs*mask[None]).sum(2)
    density=q_valid.float().sum(1)/npatch
    return dict(recall=float(recall.mean()),mass=float(mass.mean()),density=float(density.mean()),keys=float(q_valid.float().sum(1).mean()))


def main():
    a=parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    frames=[a.first_frame+i*a.frame_gap for i in range(a.num_frames)];paths=[frame_path(a.data_root,a.scene,a.sequence,f) for f in frames]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError("\n".join(missing))
    images=load_and_process_images_vggt([str(p) for p in paths]).to(a.device);model=build_model(a.checkpoint,a.device)
    cfg=ReuseExploreConfig(**json.loads(a.config.read_text()));cfg=ReuseExploreConfig(**{**cfg.__dict__,"capture":True})
    state=install_reuse_residual_exploration(model,cfg); qk={};handles=[]
    def make_pre(layer):
        def hook(wrapper,args,kwargs):
            if not a.layer_start<=layer<=a.layer_end:return
            x=args[0];pos=kwargs.get("pos",args[1] if len(args)>1 else None);q,k,_=prepare_qkv(wrapper.original,x,pos)
            qk[layer]=(q.detach().to("cpu",dtype=torch.float16),k.detach().to("cpu",dtype=torch.float16))
        return hook
    for layer,b in enumerate(model.aggregator.global_blocks):handles.append(b.attn.register_forward_pre_hook(make_pre(layer),with_kwargs=True))
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda")):model.aggregator(images.unsqueeze(0))
    finally:
        for h in handles:h.remove()
    groups=state.groups;patch_abs=state.patches.detach().cpu().long();query_abs=patch_abs.clone();rows=[]
    for layer in range(a.layer_start,a.layer_end+1):
        q,k=qk[layer];q=q.to(a.device,dtype=torch.float32);k=k.to(a.device,dtype=torch.float32)
        qp=q[:,query_abs.to(a.device)];kp=k[:,patch_abs.to(a.device)];npatch=kp.shape[1];keep=max(1,round(a.target_retention*npatch))
        current=state.captured[layer];prev=state.captured[layer-1]
        prev_full=concatenate_windows(prev,"ids",groups)
        hist_candidates=concatenate_windows(current,"history_candidates",groups)
        hist_selected=concatenate_windows(current,"history_selected",groups)
        final_support=concatenate_windows(current,"ids",groups)
        accum={name:[] for name in ("previous_full","history_candidates","history_selected","final_support")}
        for h in range(q.shape[0]):
            logits=qp[h].T@kp[h]*(q.shape[-1]**-.5);probs=logits.softmax(-1);target=logits.topk(keep,dim=-1,sorted=False).indices
            for name,cand in (("previous_full",prev_full),("history_candidates",hist_candidates),("history_selected",hist_selected),("final_support",final_support)):
                accum[name].append(support_metrics(probs[None],target[None],cand,query_abs.to(a.device),patch_abs.to(a.device)))
        for name,vals in accum.items():
            rows.append(dict(layer=layer,source=f"L{layer-1}",support=name,
                recall=float(np.mean([v["recall"] for v in vals])),attention_mass_coverage=float(np.mean([v["mass"] for v in vals])),
                candidate_density=float(np.mean([v["density"] for v in vals])),candidate_keys=float(np.mean([v["keys"] for v in vals]))))
        del q,k,qp,kp;torch.cuda.empty_cache()
    with (out/"pipeline.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"scene":a.scene,"frame_indices":frames,"target_retention":a.target_retention,"rows":rows,
        "decision_rules":[
            "If previous_full recall is high but history_candidates collapses, candidate-bank construction is the bottleneck.",
            "If history_candidates recall is high but history_selected collapses, the shared budget objective is the bottleneck.",
            "If previous_full itself is low, current sparse support failed to bootstrap useful relations in the preceding layer.",
            "Compare recall jointly with candidate_density; a larger pool is not free."],
        "trajectory":"Current unified sparse trajectory; target uses diagnostic full patch QK at each current layer."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
