import json
from pathlib import Path
from contextlib import contextmanager
import torch
from scripts.deep_operation_timing import GraphRecorder,install_graph_detail,install_clustering
from scripts.quality_metrics import quality_record,quality_markdown
OUT=Path('outputs/deep_saf3r_query256_20261105/validation')
OUT.mkdir(parents=True,exist_ok=True)
q=quality_record({},'Instrumentation correctness/calibration diagnostic, no pose/recon_unposed or paired Dense timing')
(OUT/'quality.json').write_text(json.dumps(q,indent=2)+'\n');(OUT/'quality.md').write_text(quality_markdown(q))
@contextmanager
def nullscope(name):yield
try:
 import saf3r.patch.adaptive_value_clustering as ac
 torch.manual_seed(72);torch.set_num_threads(2)
 x=torch.randn(128,12,device='cuda');labels=torch.arange(128,device='cuda')%16
 ref={m:ac.lloyd(x,labels,16,2,mode=m,neighbors=8) for m in ('global','local')}
 install_clustering(nullscope)
 for m in ('global','local'):
  actual=ac.lloyd(x,labels,16,2,mode=m,neighbors=8)
  for a,b in zip(actual[:3],ref[m][:3]):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
 recorder=GraphRecorder();v=install_graph_detail(recorder)
 (OUT/'result.json').write_text(json.dumps(dict(clustering_scopes_preserve_outputs=True,graph_validation=v,clock_calibration=recorder.calibration),indent=2)+'\n')
 print('PASSED',v,recorder.calibration,flush=True)
except Exception as exc:
 (OUT/'failure.txt').write_text(str(exc));raise
