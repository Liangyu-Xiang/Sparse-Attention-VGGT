"""Non-overlapping CUDA-event profile of changed current-probe inference only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record,quality_markdown
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');model=build_model(cfg,'cuda').eval();entry=json.loads(a.input.read_text());images=load_images(model,'vggt',entry['image_paths']).cuda();state=install_reuse_residual_exploration(model,ReuseExploreConfig(**json.loads(a.config.read_text())))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();del raw;state.profiling=True;outer=[];active={};handles=[]
 for layer,block in enumerate(model.aggregator.global_blocks):
  def begin(module,inputs,layer=layer):e=torch.cuda.Event(enable_timing=True);e.record();active[layer]=e
  def end(module,inputs,outputs,layer=layer):e=torch.cuda.Event(enable_timing=True);e.record();outer.append((layer,active.pop(layer),e))
  handles += [block.attn.register_forward_pre_hook(begin),block.attn.register_forward_hook(end)]
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();start.record()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 end.record();torch.cuda.synchronize();wall=time.perf_counter()-t
 stages={};per_layer=[]
 for layer,name,first,last in state.profile_events:
  seconds=first.elapsed_time(last)/1000;stages[name]=stages.get(name,0.)+seconds;per_layer.append(dict(layer=layer,operation=name,seconds=seconds))
 global_time=sum(a.elapsed_time(b) for _,a,b in outer)/1000;full=start.elapsed_time(end)/1000
 remainder=global_time-sum(stages.values());other=full-global_time;assert remainder>=-1e-5 and other>=-1e-5
 stages.update(unattributed_global_prepare_finalize=max(0,remainder),frame_MLP_heads_and_other_model=max(0,other));assert abs(sum(stages.values())-full)<1e-4
 core=stages['sparse_attention_only'];streamed=state.config.streamed_probability_history
 if streamed:stages['sparse_attention_with_inline_history']=stages.pop('sparse_attention_only')
 q=quality_record({},'Separate128-frame CUDA-event implementation profile; no pose/recon_unposed quality evaluated. Normal uninstrumented timings reported separately.')
 result=dict(quality=q,config=json.loads(a.config.read_text()),input=entry,full_gpu_event_seconds=full,wall_seconds=wall,stages=[dict(operation=k,seconds=v,percent=v/full*100,**({'times_attention_with_inline_history':v/core} if streamed else {'times_sparse_core':v/core})) for k,v in sorted(stages.items(),key=lambda x:-x[1])],per_layer=per_layer,attention_core_scope='AV plus inline history export,not bare AV' if streamed else 'Bare sparse AV',global_attention_wrappers_seconds=global_time,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,speedup_vs_dense=None,definition='Mutually exclusive stage events;outer global Attention times only derive remainder,never added again. Other-model remainder=full event time-global wrappers. Includes CPU launch/sync gaps inside event intervals.')
 a.output.mkdir(parents=True,exist_ok=False);save(a.output/'result.json',result);(a.output/'quality.md').write_text(quality_markdown(q));print(json.dumps(result['stages'],indent=2))
if __name__=='__main__':main()
