"""Opt-in frozen current candidate; complete pose + unposed reconstruction."""
import argparse,importlib,json,sys,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
# Avoid loading unrelated optional SAF3R backends for this VGGT-only experiment.
import saf3r
package=types.ModuleType('saf3r.patch');package.__path__=[str(ROOT/'saf3r/patch')]
sys.modules['saf3r.patch']=package;saf3r.patch=package
from scripts.quality_metrics import quality_record,quality_markdown

def main(a):
    from saf3r.utils import model_utils
    from saf3r.patch.beta_head_batching import install as head_batching
    original=model_utils.build_model
    chunk=a.head_chunk if a.head_chunk is not None else (32 if a.dataset=='7scenes' else 8)
    def build(cfg,device):
        model=original(cfg,device)
        if chunk!=8:head_batching(model,chunk,validate=False)
        return model
    model_utils.build_model=build
    module=importlib.import_module('experiments.beta_unified.evaluate_fast_subgroup_'+('7scenes' if a.dataset=='7scenes' else 'eth'))
    module.run(a)
    for file in ['forward.json','result.json']:
        path=a.output/file;data=json.loads(path.read_text())
        data['engineering']=dict(head_frames_chunk=chunk,all_heads_retained=True,
            precision='Unchanged BF16 backbone and FP32 prediction heads',algorithm_engine='fast_subgroup')
        module.save(path,data)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['eth3d','7scenes'],required=True)
    p.add_argument('--scene',required=True)
    for name in ['expected-input','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--alpha',type=float,default=.5);p.add_argument('--tau',type=float,default=.5)
    p.add_argument('--head-chunk',type=int)
    a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        a.output.mkdir(parents=True,exist_ok=True);q=quality_record({},f'{type(exc).__name__}: {exc}')
        (a.output/'quality.json').write_text(json.dumps(q,indent=2))
        (a.output/'quality.txt').write_text(quality_markdown(q))
        (a.output/'failure.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2))
        raise
