"""Sampled ideal Dense probabilities, sparse outputs and masks; no dataset GT.

This computes exact Attention for 4 probes in 4 groups/layer, NOT a Dense model
control run. Diagnostic overhead is excluded from all performance timings.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_light_reuse import build
from saf3r.utils.data_utils import load_images

def probability_error(estimate,truth):
 return dict(relative_l1=float((estimate-truth).abs().sum()/truth.sum().clamp_min(1.e-12)),mass_bias=float((estimate.sum()-truth.sum())/truth.sum().clamp_min(1.e-12)),estimated_mass=float(estimate.sum()),true_mass=float(truth.sum()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 model,s=build(a.config);images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();rows={};predictions={};qkv=[None];position=dict(layer=-1,start=0)
 from saf3r.patch.region_history_attention import prepare_qkv as original_qkv
 import saf3r.patch.reuse_residual_exploration as legacy
 import saf3r.patch.reuse_lightweight as light
 import saf3r.patch.reuse_global_history as history
 def prepared(*args,**kwargs):
  q,k,v=original_qkv(*args,**kwargs);qkv[0]=(q.to(v.dtype),k.to(v.dtype),v);return q,k,v
 legacy.prepare_qkv=prepared;light.prepare_qkv=prepared
 original_components=history.components
 def components(state,start,ids,obs):
  bg,hp,aux=original_components(state,start,ids,obs)
  for group in sorted(set([0,state.groups//3,2*state.groups//3,state.groups-1])):
   if start<=group<start+len(ids):predictions[(state.previous_layer+1,group)]=(ids[group-start].clone(),hp[group-start].clone())
  return bg,hp,aux
 history.components=components
 def wrap_attention(module):
  original=module.weighted_attention
  def run(*args,**kw):
   out=original(*args,**kw);layer=s.previous_layer+1
   if position['layer']!=layer:position.update(layer=layer,start=0)
   start=position['start'];g=len(args[3]);position['start']+=g
   q,k,v=qkv[0];actual=args[3];targets=sorted(set([0,s.groups//3,2*s.groups//3,s.groups-1]))
   for group in targets:
    if not start<=group<start+g:continue
    local=group-start;positions=s.probe_positions[group];pr=positions-group*32
    unique=torch.ones(len(pr),device=q.device,dtype=torch.bool);unique[1:]=pr[1:]!=pr[:-1];weights=unique.float()/unique.sum()
    with torch.autocast('cuda',enabled=False):
     logits=q[:,positions].float()@k.float().transpose(-1,-2)*args[6]
     dense=logits.softmax(-1);mean=(dense.mean(0)*weights[:,None]).sum(0)
     truth_output=dense@v.float();sparse=out[0][:,local*32+pr].float()
    flat_a=truth_output[:,unique].flatten();flat_b=sparse[:,unique].flatten()
    valid=actual[local]>=0;keys=actual[local,valid].long();truth=mean[keys];patch=mean[s.patches]
    entry=dict(layer=layer,group=group,selected_edges=int(valid.sum()),unique_selected_edges=len(keys.unique()),special_edges=len(s.specials),dense_selected_mass=float(truth.sum()),dense_top_budget_mass=float(patch.topk(s.budget).values.sum()),dense_patch_mass=float(patch.sum()),output_relative_l2=float((flat_a-flat_b).norm()/flat_a.norm().clamp_min(1.e-12)),output_cosine=float(torch.nn.functional.cosine_similarity(flat_a[None],flat_b[None])))
    old=s.history_ids[group] if s.history_ids is not None else None
    if old is not None:
     membership=torch.zeros(len(s.patches),device=q.device,dtype=torch.bool);membership[old[old>=0].long()]=True
     inverse=torch.full((s.n,),-1,device=q.device,dtype=torch.long);inverse[s.patches]=torch.arange(len(s.patches),device=q.device)
     relative=inverse[keys]
     intersection=int(membership[relative].sum());entry['previous_mask_jaccard']=intersection/(int(valid.sum())+int((old>=0).sum())-intersection)
    predicted=predictions.pop((layer,group),None)
    if predicted is not None:
     old_ids,hp=predicted;good=old_ids>=0;oldactual=s.patches[old_ids[good].long()]
     entry['cross_layer_prediction']=probability_error(hp[good],mean[oldactual])
    entry['_truth']=truth;entry['_valid']=valid;rows[layer,group]=entry
   return out
  module.weighted_attention=run
 import saf3r.triton.reuse_light_attention as full_attention
 import saf3r.triton.reuse_serial_tuned as serial_attention
 wrap_attention(full_attention);wrap_attention(serial_attention)
 original_produce=history.produce
 def produce(state,start,ids,obs,*args,**kwargs):
  result=original_produce(state,start,ids,obs,*args,**kwargs);layer=state.previous_layer+1
  for key,row in rows.items():
   if key[0]==layer and start<=key[1]<start+len(ids):
    row['producer_estimation']=probability_error(result[0][key[1]-start,row['_valid']],row['_truth'])
  return result
 history.produce=produce
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 torch.cuda.synchronize();values=[]
 for row in rows.values():
  row.pop('_truth');row.pop('_valid');values.append(row)
 result=dict(complete=True,rows=values,input_source=str(a.input),reference_precision='FP32 matmul with TF32 disabled on current BF16 Q/K/V',config=json.loads(a.config.read_text()),sampling='4 deterministic query groups × up to 4 distinct probes × 24 layers; single supplied sequence; not dataset-wide',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Attention probability/output/mask diagnostic only; pose and reconstruction GT not accessed',dense_speedup=None,dense_reason='No Dense model control rerun; only sampled Attention reference')
 (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print('complete',len(values),flush=True)
if __name__=='__main__':main()
