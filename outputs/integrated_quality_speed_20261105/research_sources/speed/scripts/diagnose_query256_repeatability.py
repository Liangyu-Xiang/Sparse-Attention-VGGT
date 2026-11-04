"""Hash full attention outputs/supports on repeated inputs; no quality tuning."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--scene',default='heads');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
ROOT=a.source_root.resolve();sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import numpy as np,torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from benchmarks.da3.bench.registries import MV_REGISTRY
from benchmarks.da3.da3bench_utils import sample_frames
from scripts.quality_metrics import quality_record
out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
save('quality.json',quality_record({},'Full-input numerical repeatability diagnostic, not pose/recon_unposed quality evaluation; no paired Dense'))
torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');cfg.pop('patch_module');cfg.pop('sparse_config_path')
model=build_model(cfg,'cuda');dataset=MV_REGISTRY.get('7scenes')();positions,scene=sample_frames(dataset.get_data(a.scene),a.scene,128,sample_mode='uniform',seed=0)
images=load_images(model,'vggt',scene.image_files).cuda();state=install_reuse_residual_exploration(model,ReuseExploreConfig(**json.loads(a.config.read_text())))
save('input.json',dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape)))
def digest(t):
 if t is None:return None
 return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
rows=[];current=[]
for i,b in enumerate(model.aggregator.global_blocks):
 def hook(module,args,result,i=i):
  current.append(dict(layer=i,output=digest(result),supports=digest(state.history_ids),history=digest(getattr(state,'ready_edges',None)),labels=digest(state.fine.labels),reps=digest(state.fine.reps)))
 b.attn.register_forward_hook(hook)
for i in range(a.repeats):
 save('status.json',dict(stage='diagnostic_forward',iteration=i));current=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
 row=dict(iteration=i,layers=current,predictions={k:digest(v) for k,v in pred.items() if torch.is_tensor(v)})
 rows.append(row);save('traces.json',rows);del pred
 print('ITERATION',i,'first attention',row['layers'][0],'predictions',row['predictions'],flush=True)
if len(rows)>1:
 compare=[dict(iteration=i,first_unequal_layer=next((n for n,(x,y) in enumerate(zip(rows[0]['layers'],row['layers'])) if x!=y),None),predictions_equal=row['predictions']==rows[0]['predictions']) for i,row in enumerate(rows[1:],1)]
 save('comparison.json',compare);print(compare,flush=True)
save('status.json',dict(stage='diagnostic_complete'))
