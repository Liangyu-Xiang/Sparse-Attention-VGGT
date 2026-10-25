#!/usr/bin/env python3
"""Diagnose when a current Value cluster is safely compressible to one representative.

Runs the current unified method on a small clip, but uses diagnostic current-layer
Q/K/V to evaluate its already-constructed Fine Value clusters. For important
(group, cluster) pairs it records cheap candidate metrics and two diagnostic
references:
- representative affinity error: rep logit vs cluster log-mean-exp affinity;
- mass-preserving Value contribution error: true cluster contribution vs using
  the representative with the SAME true cluster attention mass.

Population compensation is intentionally not evaluated here.
"""
from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.analyze_sparse_support_transfer import build_model,frame_path
from saf3r.utils.data_utils import load_and_process_images_vggt
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from saf3r.patch.region_history_attention import prepare_qkv


def args_parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",required=True);p.add_argument("--sequence",default="seq-01")
    p.add_argument("--first-frame",type=int,default=0);p.add_argument("--frame-gap",type=int,default=20);p.add_argument("--num-frames",type=int,default=2)
    p.add_argument("--layer-start",type=int,default=10);p.add_argument("--layer-end",type=int,default=15);p.add_argument("--top-clusters",type=int,default=64)
    p.add_argument("--data-root",type=Path,default=Path("workspace/benchmark_dataset/7scenes/7Scenes"));p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    p.add_argument("--config",type=Path,default=Path("configs/evaluation/reuse_residual_exploration.json"));p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args()
    if not 1<=a.num_frames<=4 or not 0<=a.layer_start<=a.layer_end<24 or a.top_clusters<1:p.error("Invalid args")
    return a


def cluster_static(x,labels,reps,count):
    size=torch.bincount(labels,minlength=count).float().clamp_min(1)
    rep=x[reps]
    rel_l2=torch.zeros(count,device=x.device).index_add_(0,labels,(x-rep[labels]).square().sum(-1))/torch.zeros(count,device=x.device).index_add_(0,labels,x.square().sum(-1)).clamp_min(1e-12)
    xu=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12);ru=xu[reps]
    cos=torch.zeros(count,device=x.device).index_add_(0,labels,(1-(xu*ru[labels]).sum(-1)).clamp_min(0))/size
    return size,rel_l2,cos


def grouped_logit_stats(logits,labels,reps,size,count):
    """logits [H,PQ,P], return cluster stats [F]."""
    h,pq,p=logits.shape;idx=labels[None,None].expand(h,pq,p)
    sums=torch.zeros((h,pq,count),device=logits.device).scatter_add_(2,idx,logits)
    sq=torch.zeros_like(sums).scatter_add_(2,idx,logits.square())
    mean=sums/size[None,None];var=(sq/size[None,None]-mean.square()).clamp_min(0)
    mx=torch.full_like(sums,-torch.inf).scatter_reduce_(2,idx,logits,reduce="amax",include_self=True)
    exp=torch.exp(logits-mx.gather(2,idx));se=torch.zeros_like(sums).scatter_add_(2,idx,exp)
    logmean=mx+torch.log(se.clamp_min(1e-30)/size[None,None])
    rep_logit=logits[:,:,reps]
    return var.sqrt().mean((0,1)),(rep_logit-logmean).abs().mean((0,1))


def main():
    a=args_parse();out=a.output_dir;out.mkdir(parents=True,exist_ok=False);frames=[a.first_frame+i*a.frame_gap for i in range(a.num_frames)]
    paths=[frame_path(a.data_root,a.scene,a.sequence,f) for f in frames];missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError("\n".join(missing))
    images=load_and_process_images_vggt([str(p) for p in paths]).to(a.device);model=build_model(a.checkpoint,a.device)
    cfg=replace(ReuseExploreConfig(**json.loads(a.config.read_text())),capture=True);state=install_reuse_residual_exploration(model,cfg);cache={};hooks=[]
    def pre(layer):
        def hook(wrapper,args,kwargs):
            if not a.layer_start<=layer<=a.layer_end:return
            x=args[0];pos=kwargs.get("pos",args[1] if len(args)>1 else None);q,k,v=prepare_qkv(wrapper.original,x,pos)
            cache[layer]=(q.detach().cpu().half(),k.detach().cpu().half(),v.detach().cpu().half())
        return hook
    for l,b in enumerate(model.aggregator.global_blocks):hooks.append(b.attn.register_forward_pre_hook(pre(l),with_kwargs=True))
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda")):model.aggregator(images.unsqueeze(0))
    finally:
        for h in hooks:h.remove()
    patch_abs=state.patches.detach().cpu().long();rows=[]
    for layer in range(a.layer_start,a.layer_end+1):
        q,k,v=[x.to(a.device,dtype=torch.float32) for x in cache[layer]];cap=state.captured[layer];labels=cap["fine_labels"].long();reps=cap["fine_reps"].long();count=len(reps);pop=torch.bincount(labels,minlength=count).float()
        kpatch=k[:,patch_abs.to(a.device)];vpatch=v[:,patch_abs.to(a.device)]
        vm=vpatch.mean(0);km=kpatch.mean(0);size,v_l2,v_cos=cluster_static(vm,labels,reps,count);_,k_l2,k_cos=cluster_static(km,labels,reps,count)
        windows=cap["windows"];final_ids=torch.cat([w["ids"] for w in windows],0);final_pop=torch.cat([w["population"] for w in windows],0)
        patch_abs_dev=patch_abs.to(a.device)
        for group in range(state.groups):
            probes=(group*32+torch.tensor([4,12,20,28],device=a.device)).clamp_max(state.n-1)
            logits=q[:,probes]@kpatch.transpose(-1,-2)*(q.shape[-1]**-.5);prob=logits.softmax(-1)
            idx=labels[None,None].expand(logits.shape[0],logits.shape[1],-1);mass=torch.zeros((logits.shape[0],logits.shape[1],count),device=a.device).scatter_add_(2,idx,prob).mean((0,1))
            logit_std,aff_err=grouped_logit_stats(logits,labels,reps,size,count);top=mass.topk(min(a.top_clusters,count)).indices
            ids=final_ids[group].long();pops=final_pop[group].float();valid=ids>=0;idsv=ids[valid];pv=pops[valid]
            for cluster in top.tolist():
                members=torch.where(labels==cluster)[0];rep=int(reps[cluster]);m=prob[:,:,members].sum(-1)
                true=(prob[:,:,members,None]*vpatch[:,None,members]).sum(2)
                repv=vpatch[:,rep][:,None].expand(-1,prob.shape[1],-1);approx=m[:,:,None]*repv
                verr=((true-approx).norm(dim=-1)/(true.norm(dim=-1).clamp_min(1e-8))).mean()
                rep_abs=int(patch_abs_dev[rep]);rep_selected=bool(((idsv==rep_abs)&(pv>1.0001)).any())
                member_abs=patch_abs_dev[members];member_selected=int(torch.isin(idsv,member_abs).sum())
                exact_fraction=member_selected/max(1,int(pop[cluster].item()))
                state_name="rep" if rep_selected else ("exact_like" if exact_fraction>=.8 else "other")
                rows.append(dict(layer=layer,query_group=group,cluster=cluster,cluster_size=int(pop[cluster]),importance=float(mass[cluster]),
                    value_rel_l2=float(v_l2[cluster]),value_cosine_dist=float(v_cos[cluster]),key_rel_l2=float(k_l2[cluster]),key_cosine_dist=float(k_cos[cluster]),
                    qk_logit_std=float(logit_std[cluster]),rep_affinity_error=float(aff_err[cluster]),mass_preserving_value_error=float(verr),
                    selected_state=state_name,selected_member_fraction=float(exact_fraction)))
        del q,k,v,kpatch,vpatch;torch.cuda.empty_cache()
    with (out/"clusters.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    numeric=["value_rel_l2","value_cosine_dist","key_rel_l2","key_cosine_dist","qk_logit_std","rep_affinity_error","mass_preserving_value_error"]
    corr={}
    for x in numeric[:-1]:
        a1=np.asarray([r[x] for r in rows]);b1=np.asarray([r["mass_preserving_value_error"] for r in rows]);corr[x]=float(np.corrcoef(a1,b1)[0,1])
    summary={"scene":a.scene,"frame_indices":frames,"layers":[a.layer_start,a.layer_end],"rows":len(rows),"correlation_with_mass_preserving_value_error":corr,
        "decision_rules":["If Value dispersion is low but QK affinity error is high, Value-only compressibility is insufficient.","A useful cheap criterion should correlate with diagnostic contribution error and separate rep from exact-like cases."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6,5));x=np.asarray([r["value_cosine_dist"] for r in rows]);y=np.asarray([r["rep_affinity_error"] for r in rows]);ax.scatter(x,y,s=5,alpha=.25);ax.set_xlabel("Value cosine distortion");ax.set_ylabel("Representative affinity error");ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/"value_vs_affinity.png",dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
