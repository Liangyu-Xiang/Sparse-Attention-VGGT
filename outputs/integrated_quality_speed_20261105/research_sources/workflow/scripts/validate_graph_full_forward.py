"""Compare Graph and eager selectors on identical real tensors at every window."""
import argparse,importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_zdrop_history_states import run
from scripts.quality_metrics import quality_record,quality_markdown
parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);a=parser.parse_args()
OUT=ROOT/'outputs/source_local_quality_20261105'/f'graph_equivalence_{a.scene}'
wg=importlib.import_module('saf3r.patch.reuse_window_graph');original=wg.select;rows=[]
def validate(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None):
 actual=original(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=uniform)
 args=(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier)
 expected=wg.eager(*args,uniform=uniform)
 checks={name:(x!=y).sum() for name,x,y in zip(('ids','real','weights'),actual[:3],expected[:3])}
 checks.update({name:(actual[3][name]!=expected[3][name]).sum() for name in actual[3]})
 semantic={}
 if bool(checks['ids']):
  n=prior.shape[1];g=len(prior)
  def mask(ids):
   result=torch.zeros((g,n+1),device=ids.device,dtype=torch.bool)
   result.scatter_(1,torch.where(ids>=0,ids.long(),n),True)
   return result[:,:n]
  am=mask(actual[0]);em=mask(expected[0])
  semantic=dict(added_keys=int((am&~em).sum()),removed_keys=int((em&~am).sum()),changed_groups=int((am!=em).any(1).sum()))
 again=wg.eager(*args,uniform=uniform)
 checks['eager_repeat_id_differences']=(again[0]!=expected[0]).sum()
 rows.append(dict(layer=layer,groups=len(prior),checks=checks,semantic_support_difference=semantic))
 return actual
wg.select=validate
args=SimpleNamespace(output_dir=OUT,method='zdrop_history_states',scene=a.scene,dataset='7scenes',frames=128,
 config=ROOT/'configs/evaluation/reuse_shared_value_loss_q256_adaptive_reference.json',capture=False,warmup=0,repeats=1,diagnostic=True,selector_execution='graph',baseline=None)
run(args)
for r in rows:r['checks']={k:int(v) for k,v in r['checks'].items()}
data=dict(scene=a.scene,scope='All layer-1..23 windows from one 128-frame real forward; Graph vs eager on identical current tensors; timings include duplicate computations and checks and are not normal performance',
 windows=len(rows),all_supports_exact=all(not any(r['checks'][k] for k in ('ids','real','weights')) for r in rows),
 diagnostic_difference_windows=sum(any(v for k,v in r['checks'].items() if k not in ('ids','real','weights','eager_repeat_id_differences')) for r in rows),
 eager_nondeterministic_windows=sum(r['checks']['eager_repeat_id_differences']>0 for r in rows),rows=rows,
 quality=quality_record({},'Explicit all-window implementation diagnostic; pose/recon_unposed not evaluated'))
(OUT/'equivalence.json').write_text(json.dumps(data,indent=2)+'\n');(OUT/'quality.json').write_text(json.dumps(data['quality'],indent=2)+'\n');(OUT/'quality.md').write_text(quality_markdown(data['quality']))
print(json.dumps({k:v for k,v in data.items() if k not in ('rows','quality')}),flush=True)
