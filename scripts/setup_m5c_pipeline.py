"""Fork the verified M5c implementation; optimize Drop and statistic scheduling."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/tmp/zdrop_m5c_equivalent_20261102'));p.add_argument('--destination',type=Path,default=Path('/tmp/zdrop_m5c_pipeline_local'))
    args=p.parse_args()
    if args.destination.exists():raise ValueError('Destination must be new')
    hashes=json.loads((ROOT/'outputs/m5c_equivalent_heads128_20261102/manifest.json').read_text())['source']['after']['sha256']
    assert all(hashlib.sha256((args.source/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'Source differs from tested M5c implementation'
    shutil.copytree(args.source,args.destination,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['saf3r/patch/reuse_m5c_pipeline.py','saf3r/triton/reuse_drop_moment.py']:shutil.copy2(ROOT/name,args.destination/name)
    path=args.destination/'saf3r/patch/reuse_residual_exploration.py';s=path.read_text().replace('from . import reuse_m5c_equivalent as m5c','from . import reuse_m5c_pipeline as m5c')
    needle='geometry=compact.prepare_geometry(zvalues,s.fine.labels,s.fine.reps,population,l2=c.state_l2)';assert needle in s
    s=s.replace(needle,needle+"\n                from saf3r.triton.reuse_drop_moment import prepare as prepare_drop,drop_norm\n                if geometry is not None:geometry['drop_bins']=prepare_drop(population)")
    old="                                totals=torch.zeros((g,s.fine.base_count,vec.shape[1]),device=vec.device)\n                                for j in range(g):totals[j].index_add_(0,s.fine.labels,prior[j,:,None]*vec)\n                                geometry['drop_values']=totals.norm(dim=-1)"
    assert old in s;s=s.replace(old,"                                geometry['drop_values']=drop_norm(prior,vec,geometry,s.fine.base_count,geometry['drop_bins'])")
    s=s.replace('s.m5c_next={};s.m5c_reconstruction_qk=0;','s.m5c_next={};s.m5c_count_errors=[];s.m5c_reconstruction_qk=0;')
    old='        s.history_ids=torch.cat(history);s.history_weights=None';assert old in s
    s=s.replace(old,"        if c.history_mode=='dense_m5c':\n            if bool(torch.stack(s.m5c_count_errors).any()):raise ValueError('History count exceeds region population')\n            s.m5c_reconstruction_qk=int(s.m5c_reconstruction_qk)\n            s.m5c_probe_qk=int(s.m5c_probe_qk)\n"+old)
    s=s.replace("history_cache_device='cpu_grid_statistics_gpu_previous_qk'","history_cache_device='gpu_grid_statistics_gpu_previous_qk'")
    path.write_text(s)
    evaluator=args.destination/'scripts/evaluate_zdrop_history_states.py'
    text=evaluator.read_text().replace('names=set(sources)|{',"names=set(sources)|{'saf3r/patch/reuse_m5c_equivalent.py','saf3r/triton/reuse_m5c_equivalent.py','saf3r/patch/reuse_m5c_pipeline.py','saf3r/triton/reuse_drop_moment.py',")
    evaluator.write_text(text)
    print(args.destination)
if __name__=='__main__':main()
