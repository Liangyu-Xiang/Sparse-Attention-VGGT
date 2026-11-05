"""Reference self-repeat vs equivalent engineering: features, clusters and masks.

Diagnostic captures intentionally add work; these are NOT performance timings.
"""
import argparse,json,sys,types,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_residual_exploration as impl
from saf3r.patch.source_local_unanchored import SourceLocalUnanchoredValue
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
ref=impl.ReuseExploreConfig(**json.loads((ROOT/'configs/evaluation/cluster20_speed_20261105/reference.json').read_text()));state=impl.install_reuse_residual_exploration(model,ref)
current={};old_update=SourceLocalUnanchoredValue.update;old_forward=impl.ReuseResidualExploration.forward

def update(self,x,layer):
    row=current.setdefault(layer,{});row['features']=x.clone();result=old_update(self,x,layer);row['labels']=self.labels.clone();row['reps']=self.reps.clone();return result

def forward(self,x,pos=None):
    row=current.setdefault(self.layer,{});rows=torch.linspace(0,x.shape[1]-1,64,device=x.device).long();row['input_sample']=x[:,rows].clone();out=old_forward(self,x,pos=pos);row['mask']=self.state.history_ids.clone();row['output_sample']=out[:,rows].clone();return out
SourceLocalUnanchoredValue.update=update;impl.ReuseResidualExploration.forward=forward
baseline=None;reports=[];base_pose=None;base_depth=None
for name in ['reference_first','reference_repeat','precision_fast','precision_fast_repeat']:
    if name=='precision_fast':
        fast=impl.ReuseExploreConfig(**json.loads((ROOT/'configs/evaluation/cluster20_speed_20261105/precision_preserving.json').read_text()));state.config=fast;model.aggregator.release_cuda_cache_each_block=False
        from saf3r.triton.rope_fast import forward as rope_forward
        seen=set()
        for block in list(model.aggregator.frame_blocks)+list(model.aggregator.global_blocks):
            block.fuse_inference_residual=True;rope=block.attn.original.rope if hasattr(block.attn,'original') else block.attn.rope
            if rope is not None and id(rope) not in seen:
                rope.forward=types.MethodType(rope_forward,rope);rope._fixed_max_position=max(images.shape[-2]//14,images.shape[-1]//14)+1;seen.add(id(rope))
    current={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();pose=raw['pose_enc'].float();depth=raw['depth'][...,:8,:8,:].float()
    if baseline is None:baseline=current;base_pose=pose.clone();base_depth=depth.clone()
    row=dict(name=name,layers=[])
    for layer,values in current.items():
        ref_values=baseline[layer];q=dict(layer=layer)
        for key,v in values.items():
            original=ref_values[key];q[key+'_equal']=torch.equal(v,original)
            if key in ('labels','reps','mask'):q[key+'_different']=int((v!=original).sum()) if v.shape==original.shape else 'shape_changed'
            else:q[key+'_relative_l2']=float((v.float()-original.float()).norm()/original.float().norm().clamp_min(1e-30))
        row['layers'].append(q)
    row.update(pose_relative_l2=float((pose-base_pose).norm()/base_pose.norm().clamp_min(1e-30)),depth_sample_relative_l2=float((depth-base_depth).norm()/base_depth.norm().clamp_min(1e-30)))
    reports.append(row);print(name,'first feature drift',next((r['layer'] for r in row['layers'] if not r['features_equal']),None),'first label drift',next((r['layer'] for r in row['layers'] if not r['labels_equal']),None),'pose',row['pose_relative_l2'],flush=True)
    del raw,current,pose,depth;gc.collect()
    result=dict(reports=reports,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Full trajectory numerical diagnostic with captures; no geometric evaluation',speedup_vs_dense=None,speedup_reason='Instrumented numerical diagnostic, not timed forward')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
