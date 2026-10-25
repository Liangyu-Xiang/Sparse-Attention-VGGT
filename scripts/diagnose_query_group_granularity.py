#!/usr/bin/env python3
"""Dense diagnostic for the cost of sharing one historical support across queries.

At each source layer, build a patch-key candidate support by averaging Dense
attention probabilities over all heads and over query groups of size
32/16/8/4/1 (global token indexing). At the next layer, measure recall of each
head/query's current Top-rho keys and current attention-mass coverage.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import torch
from scripts.analyze_sparse_support_transfer import NUM_GLOBAL_LAYERS,NUM_HEADS,build_model,frame_path,patch_indices,qkv_from_module
from saf3r.utils.data_utils import load_and_process_images_vggt


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scene",required=True);p.add_argument("--sequence",default="seq-01")
    p.add_argument("--first-frame",type=int,default=0);p.add_argument("--frame-gap",type=int,default=20);p.add_argument("--num-frames",type=int,default=2)
    p.add_argument("--retention",type=float,default=.20);p.add_argument("--group-sizes",type=int,nargs="+",default=[32,16,8,4,1]);p.add_argument("--layer-start",type=int,default=9);p.add_argument("--layer-end",type=int,default=15)
    p.add_argument("--data-root",type=Path,default=Path("workspace/benchmark_dataset/7scenes/7Scenes"));p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"));p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();a.group_sizes=sorted(set(a.group_sizes),reverse=True)
    if not 0<a.retention<1 or any(g<1 for g in a.group_sizes) or not 0<=a.layer_start<a.layer_end<NUM_GLOBAL_LAYERS:p.error("Invalid args")
    return a


def candidate_from_headmean(headmean,query_abs,total_tokens,group_size,keep):
    groups=math.ceil(total_tokens/group_size);gid=(query_abs//group_size).long();scores=torch.zeros((groups,headmean.shape[1]),device=headmean.device);scores.index_add_(0,gid,headmean)
    counts=torch.zeros(groups,device=headmean.device).index_add_(0,gid,torch.ones(len(gid),device=headmean.device));scores/=counts.clamp_min(1)[:,None]
    return scores.topk(keep,dim=1,sorted=False).indices.cpu()


class Collector:
    def __init__(self,a):self.a=a;self.layer=0;self.previous=None;self.rows=[]
    def __call__(self,module,args,kwargs,_out):
        l=self.layer;self.layer+=1
        if l<self.a.layer_start or l>self.a.layer_end:return
        q,k,_=qkv_from_module(module,args,kwargs);patch=patch_indices(q.shape[1],self.a.num_frames,q.device);query_abs=patch;qp=q[:,patch];kp=k[:,patch];n=len(patch);keep=max(1,round(self.a.retention*n));total=q.shape[1]
        probs=[];targets=[]
        for h in range(NUM_HEADS):
            logits=qp[h].float()@kp[h].float().T*(q.shape[-1]**-.5);p=logits.softmax(-1);probs.append(p);targets.append(logits.topk(keep,dim=-1,sorted=False).indices)
        probs=torch.stack(probs);targets=torch.stack(targets)
        if self.previous is not None:
            for gs,candidates in self.previous.items():
                cand=candidates.to(q.device)[(query_abs//gs).long()];mask=torch.zeros((n,n),device=q.device,dtype=torch.int32);mask.scatter_add_(1,cand,torch.ones_like(cand,dtype=torch.int32));mask=mask>0
                recall=mask[None].expand(NUM_HEADS,-1,-1).gather(2,targets).float().mean();mass=(probs*mask[None]).sum(-1).mean()
                self.rows.append(dict(source_layer=l-1,target_layer=l,layer_pair=f"{l-1}→{l}",group_size=gs,candidate_density=keep/n,recall=float(recall),attention_mass_coverage=float(mass)))
        headmean=probs.mean(0);self.previous={gs:candidate_from_headmean(headmean,query_abs,total,gs,keep) for gs in self.a.group_sizes}


def main():
    a=parse();out=a.output_dir;out.mkdir(parents=True,exist_ok=False);frames=[a.first_frame+i*a.frame_gap for i in range(a.num_frames)];paths=[frame_path(a.data_root,a.scene,a.sequence,f) for f in frames]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError("\n".join(missing))
    images=load_and_process_images_vggt([str(p) for p in paths]).to(a.device);model=build_model(a.checkpoint,a.device);c=Collector(a);hooks=[b.attn.register_forward_hook(c,with_kwargs=True) for b in model.aggregator.global_blocks]
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda")):model.aggregator(images.unsqueeze(0))
    finally:
        for h in hooks:h.remove()
    if not c.rows:raise RuntimeError("No rows collected")
    with (out/"granularity.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(c.rows[0]));w.writeheader();w.writerows(c.rows)
    summary={"scene":a.scene,"frame_indices":frames,"retention":a.retention,"group_sizes":a.group_sizes,"rows":c.rows,
        "interpretation":"All variants use the same candidate density. Higher recall at smaller group size isolates the information loss caused by query support sharing; supports are still shared across heads."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
