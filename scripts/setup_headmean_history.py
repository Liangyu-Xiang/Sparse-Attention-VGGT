"""Freeze a head-mean + single-QK candidate without touching the v5 reference."""
import argparse,shutil,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
 if a.destination.exists():raise ValueError('Fresh destination required')
 base=Path('/tmp/zdrop_ready_history_v5_20261102')
 hashes=json.loads((ROOT/'outputs/ready_history_v5_heads128_20261102/manifest.json').read_text())['source']['after']['sha256']
 assert all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
 shutil.copytree(base,a.destination,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 names=['saf3r/patch/reuse_headmean_history.py','saf3r/triton/reuse_headmean_history.py','saf3r/triton/reuse_single_qk_grid.py']
 for n in names:shutil.copy2(ROOT/n,a.destination/n)
 p=a.destination/'saf3r/patch/reuse_residual_exploration.py';s=p.read_text().replace('from . import reuse_ready_history as m5c','from . import reuse_headmean_history as m5c').replace('from saf3r.triton.reuse_explore_av_ready import weighted_attention','from saf3r.triton.reuse_single_qk_grid import weighted_attention')
 s=s.replace('single_qk=c.evidence_mode=="saved_logits",relative_reference=reference,emit_evidence=self.layer<23)', 'key_grid=s.grid[ids.clamp_min(0)],regions=s.regions,emit_evidence=self.layer<23)')
 a0=s.index("                mass=mass/mass.sum(1,keepdim=True).clamp_min(1e-30)",s.index("with stage(s,'history_summary')"));a1=s.index("                history.append",a0)
 s=s[:a0]+"                patch_mass=mass[:,0].sum(1,keepdim=True).clamp_min(1e-30)\n                grid=mass[:,0]/patch_mass\n                real_grid=mass[:,1]/patch_mass\n                real=ids>=0\n"+s[a1:]
 s=s.replace("evidence_extra_qk_per_head=(joined['counts']*rows).sum() if c.evidence_mode=='second_qk' else 0",'evidence_extra_qk_per_head=0')
 s=s.replace('history_estimator_head_aware=True','history_estimator_head_aware=False')
 s=s.replace("history_estimator='Ready D2 conditionals + M5c'","history_estimator='Head-mean conditionals + head-mean M5c'")
 s=s.replace("evidence=c.evidence_mode,","evidence='single_qk_fused_grid',")
 s=s.replace('min(c.window_groups,s.groups)*q.shape[0]*4*s.budget*4 if self.layer<23 else 0','min(c.window_groups,s.groups)*q.shape[0]*4*s.budget*4 if self.layer<23 else 0')
 s=s.replace("history_cache_dtype='positive_fp16_u14_edges_float32_grid_statistics'","history_cache_dtype='headmean_positive_fp16_u14_edges_float32_grid_statistics'")
 s=s.replace("history='Prepared preceding-layer conditionals; FP16 edge cache; no old-QK reconstruction; M5c beta(1,2)'","history='Previous conditional probabilities averaged over heads before caching; current grid probabilities averaged before confidence; M5c beta(1,2); no head matching'")
 s=s.replace("('second_qk','saved_logits')","('second_qk','saved_logits','fused_grid')")
 s=s.replace("if c.history_mode!='dense_m5c' or","if c.capture:raise RuntimeError('Fused-grid capture exposes grid summaries, not per-edge masses; capture mode unsupported')\n        if c.history_mode!='dense_m5c' or")
 p.write_text(s)
 config=a.destination/'configs/evaluation/reuse_zdrop_history_states.json'
 values=json.loads(config.read_text());values['evidence_mode']='fused_grid';config.write_text(json.dumps(values,indent=2)+'\n')
 evaluator=a.destination/'scripts/evaluate_zdrop_history_states.py'
 text=evaluator.read_text().replace('names=set(sources)|{','names=set(sources)|{'+','.join(repr(n) for n in names)+',')
 evaluator.write_text(text)
 print(a.destination)
if __name__=='__main__':main()
