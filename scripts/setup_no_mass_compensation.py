"""Single-factor ablation of the evaluated head-mean/single-QK method."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
 if a.destination.exists():raise ValueError('Fresh destination required')
 previous=ROOT/'outputs/headmean_single_qk_quality_20261102'
 m=json.loads((previous/'manifest.json').read_text());base=Path(m['source_root'])
 assert all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in m['source_sha256'].items())
 shutil.copytree(base,a.destination,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 p=a.destination/'saf3r/patch/reuse_residual_exploration.py';s=p.read_text()
 assert s.count('if c.mass_transfer:')==1
 s=s.replace('if c.mass_transfer:','if c.mass_transfer and not c.disable_population_compensation:')
 s=s.replace('pops=(ids>=0).to(dtype=pops.dtype)','pops=(ids>=0).float()')
 s=s.replace('mass_transfer="estimated assigned mass / own prior mass" if c.state_loss=="cosine_distortion" and c.mass_transfer else "original population weights",','mass_transfer="disabled: unit weight on each selected Key",mass_compensation_enabled=False,')
 p.write_text(s)
 p=a.destination/'saf3r/patch/zdrop_history_cleanup.py';s=p.read_text()
 assert ",history_mass_transfer_error=context['mass_error']" in s
 s=s.replace(",history_mass_transfer_error=context['mass_error']",'')
 p.write_text(s)
 p=a.destination/'configs/evaluation/reuse_zdrop_history_states.json';v=json.loads(p.read_text());assert not v['disable_population_compensation'];v['disable_population_compensation']=True;p.write_text(json.dumps(v,indent=2)+'\n')
 audit=dict(parent=str(previous),changed_existing_files=['saf3r/patch/reuse_residual_exploration.py','saf3r/patch/zdrop_history_cleanup.py','configs/evaluation/reuse_zdrop_history_states.json'],config_change={'disable_population_compensation':[False,True]},semantics='Bypass mass transfer, use weight 1 for every selected Key including L0. Keep selection losses, History qualification, clustering, budget, head-mean estimation and single-QK backend unchanged. Remove inapplicable transfer-conservation diagnostic rather than reporting a fictitious zero.')
 (a.destination/'no_compensation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(a.destination)
if __name__=='__main__':main()
