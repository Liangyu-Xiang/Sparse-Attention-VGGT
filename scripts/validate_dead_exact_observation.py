"""Real sparse trajectories: removing unread Exact-observation computations."""
import argparse,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch,yaml
 from addict import Dict
 from dataclasses import replace
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record,quality_markdown
 import saf3r.triton.reuse_direct_sampled_probability as direct
 plan=json.loads(a.plan.read_text());cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint']
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',plan['validation']['chess_seq03']['image_paths'][:8]).cuda()
 config=ReuseExploreConfig(**next(r['config'] for r in plan['candidates'] if r['name']=='p3_sketch10_mid21'))
 outputs=[];supports=[];original=direct.direct_sparse_attention_sampled
 for skip in [False,False,True,True]:
  state=install_reuse_residual_exploration(model,replace(config,skip_dead_exact_observation=skip));ids=[]
  def capture(q,k,v,actual,special,scale,bias,**kwargs):
   out=original(q,k,v,actual,special,scale,bias,**kwargs);ids.append(actual.cpu());return out
  direct.direct_sparse_attention_sampled=capture
  try:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
   torch.cuda.synchronize();outputs.append({k:raw[k].float().cpu() for k in ['pose_enc','depth','depth_conf']});supports.append(torch.cat([v.flatten() for v in ids]));print('VALIDATED_FORWARD',skip,flush=True)
  finally:
   direct.direct_sparse_attention_sampled=original;state.reset_handle.remove()
   for b in model.aggregator.global_blocks:b.attn=b.attn.original
   del model.reuse_explore_state
   del state,raw;gc.collect();torch.cuda.empty_cache()
 rows=[]
 for i,j,label in [(0,1,'reference_self_repeat'),(0,2,'skip_vs_reference'),(0,3,'skip_repeat_vs_reference')]:
  errors={k:float((outputs[i][k]-outputs[j][k]).square().sum()/outputs[i][k].square().sum().clamp_min(1e-20))**.5 for k in outputs[0]}
  rows.append(dict(comparison=label,output_relative_l2=errors,support_mismatch_fraction=float((supports[i]!=supports[j]).float().mean())))
 passed=all(max(r['output_relative_l2'].values())<=.005 for r in rows)
 q=quality_record({},'Real8-frame no-GT numerical equivalence diagnostic; pose/recon_unposed quality not evaluated')
 a.output.mkdir(parents=True,exist_ok=False);save(a.output/'result.json',dict(passed=passed,rows=rows,quality=q,frames=8,checkpoint_sha256=plan['checkpoint_sha256'],speedup_vs_dense=None));(a.output/'quality.md').write_text(quality_markdown(q));print(json.dumps(dict(passed=passed,rows=rows),indent=2));assert passed
if __name__=='__main__':main()
