"""Full11-scene camera diagnostics; GPU4/5/6 leased dynamic queue, model reuse per worker."""
import argparse,copy,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/camera_token_comparison_20261105'
SCENES=['facade','terrains','delivery_area','electro','courtyard','playground','kicker','office','relief_2','relief','pipes']
def read(p):return json.loads(p.read_text())
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def prepare():
 from scripts.camera_token_diagnostic import sha
 old=read(ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105/frozen_manifest.json')
 source={str(p.relative_to(ROOT)):sha(p) for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')}
 assert all(source.get(k)==v for k,v in old['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
 sources=source|{p:sha(ROOT/p) for p in ['scripts/camera_token_diagnostic.py','scripts/run_camera_token_comparison.py','configs/evaluation/reuse_rep_history_A_all_specials_cluster20.json','configs/evaluation/eval_saf3r_vggt.yaml','configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']}
 save(OUT/'frozen_plan.json',dict(scenes=SCENES,methods=['dense','ours','saf3r'],source_sha256=sources,checkpoint_sha256=read(ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105/quality/A_all_specials_cluster20/inputs/pipes.json')['checkpoint_sha256'],gpus=[4,5,6],config=read(ROOT/'configs/evaluation/reuse_rep_history_A_all_specials_cluster20.json'),sampling='Existing full11 ETH3D exact RGB manifests,377 total frames; all24global+24frame blocks,all16heads,all camera rows,all actual support Keys. No Key/query subsampling. Separate normal-path trajectories, same-QKV and Q/K/V counterfactual diagnostics. No production changes, quality or timing reruns.',saf3r='Original Top-4 lazy first-correspondence-head selection, profile eth3d-train-fltr-calib_cmpmse.json; not paper DINO Top-K; explicitly requested attention comparison'))
 save(OUT/'queue.json',dict(pending=SCENES,claims=[]))
def worker(gpu):
 import torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from scripts.camera_token_diagnostic import run_scene,sha
 plan=read(OUT/'frozen_plan.json')
 for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
 dense=copy.deepcopy(cfg);dense.pop('patch_module');dense.pop('sparse_config_path');assert sha(Path(cfg.ckpt_path))==plan['checkpoint_sha256'];model=build_model(dense,'cuda')
 while True:
  with (OUT/'queue.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);q=read(OUT/'queue.json')
   if not q['pending']:return
   scene=q['pending'].pop(0);q['claims'].append(dict(scene=scene,gpu=gpu,stage='forward'));save(OUT/'queue.json',q)
  print('SCENE_START',gpu,scene,flush=True)
  try:run_scene(model,cfg,scene,OUT/'full',checkpoint_sha=plan['checkpoint_sha256']);code=0
  except Exception as exc:
   from scripts.quality_metrics import quality_record,quality_markdown
   for method in ['dense','ours','saf3r']:
    folder=OUT/'full'/scene/method
    if folder.exists() and not (folder/'result.json').exists():
     reason=f'Camera diagnostic failed: {type(exc).__name__}: {exc}';quality=quality_record({},reason);save(folder/'quality.json',quality);(folder/'quality.md').write_text(quality_markdown(quality));save(folder/'status.json',dict(stage='failed',reason=reason))
   save(OUT/f'{scene}_failure.json',dict(error=repr(exc)));code=1
  with (OUT/'queue.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);q=read(OUT/'queue.json')
   next(c for c in q['claims'] if c['scene']==scene).update(stage='complete' if not code else 'failed',returncode=code);save(OUT/'queue.json',q)
  print('SCENE_END',gpu,scene,code,flush=True)
  if code:raise SystemExit(code)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[4,5,6]);a=p.parse_args()
 if not (OUT/'frozen_plan.json').exists():prepare()
 if a.worker is not None:worker(a.worker);return
 procs={g:subprocess.Popen([sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(g),'--monitor',str(OUT/f'worker{g}_gpu.json'),'--log',str(OUT/f'worker{g}.log'),'--',sys.executable,str(Path(__file__).resolve()),'--worker',str(g)],cwd=ROOT) for g in [4,5,6]}
 while any(p.poll() is None for p in procs.values()):
  q=read(OUT/'queue.json');save(OUT/'progress.json',dict(queue=q,complete=[(s,m) for s in SCENES for m in ['dense','ours','saf3r'] if (OUT/'full'/s/m/'result.json').exists()]));time.sleep(2)
 failed=[g for g,p in procs.items() if p.wait()];save(OUT/'execution.json',dict(failed_workers=failed))
 if failed:raise RuntimeError(str(failed))
 from scripts.report_camera_token_comparison import report
 report()
if __name__=='__main__':main()
