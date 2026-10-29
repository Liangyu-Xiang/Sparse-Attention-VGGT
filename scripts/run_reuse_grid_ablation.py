"""Queue grid-only ablations after current campaign; preserve paired references."""
import os,sys,time,json,shutil,subprocess,hashlib
from pathlib import Path
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_reuse_ablation_eval as campaign
REFERENCE=ROOT/'outputs/reuse_single_change_7scenes128_20261029'
OUTPUT=ROOT/'outputs/reuse_canonical_grid_7scenes128_20261029'
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 campaign.save(OUTPUT/'queue_status.json',dict(stage='waiting_for_previous_campaign',reference=str(REFERENCE)))
 while True:
  path=REFERENCE/'queue_status.json'
  if path.exists() and json.loads(path.read_text()).get('all_terminal'):break
  time.sleep(15)
 # A supplemental CUDA unit test overlapped heads/max_error forward on GPU0.
 # Retain its raw record but replace it with an isolated full re-evaluation.
 campaign.save(OUTPUT/'queue_status.json',dict(stage='rerunning_contended_heads_max_error',reference=str(REFERENCE)))
 affected=REFERENCE/'heads/max_error'
 archive=REFERENCE/'timing_excluded/heads_max_error_gpu_unit_test_overlap'
 if affected.exists() and not archive.exists():
  archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(affected),str(archive))
  campaign.save(archive/'timing_exclusion.json',dict(reason='Supplemental grid CUDA unit tests overlapped this forward on GPU0; excluded from comparisons, isolated full rerun follows'))
 if not affected.exists():
  with (REFERENCE/'logs/heads_max_error_isolated_rerun.log').open('w') as log:
   rc=subprocess.run([sys.executable,'scripts/evaluate_reuse_ablation.py','--method','max_error','--scene','heads','--frames','128','--output-dir',str(affected),'--baseline',str(REFERENCE/'heads/dense/forward.json')],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT).returncode
   print('isolated timing rerun exit',rc,flush=True)
 campaign.report(REFERENCE)
 for scene in campaign.SCENES:
  for method in ('dense','original'):
   target=OUTPUT/scene/method;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():target.symlink_to(REFERENCE/scene/method,target_is_directory=True)
 campaign.METHODS=('dense','original','grid4','grid8')
 original_save=campaign.save
 def save_with_grid_manifest(path,data):
  if path==OUTPUT/'manifest.json':
   names=['configs/evaluation/reuse_ablation_grid4.json','configs/evaluation/reuse_ablation_grid8.json','scripts/run_reuse_grid_ablation.py']
   data['source_sha256'].update({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})
   data.update(reference_campaign=str(REFERENCE),independent_grid_variants={'original':[2,2],'grid4':[4,4],'grid8':[8,8]},baseline_reused='same campaign Dense and original, same per-scene GPU/input/checkpoint/protocol; symlinked raw records, not rerun')
  original_save(path,data)
 campaign.save=save_with_grid_manifest
 campaign.run(Namespace(output_dir=OUTPUT))
 if not json.loads((OUTPUT/'queue_status.json').read_text())['all_terminal']:raise RuntimeError('Queue incomplete')
 print('GRID_CAMPAIGN_TERMINAL',flush=True)
if __name__=='__main__':main()
