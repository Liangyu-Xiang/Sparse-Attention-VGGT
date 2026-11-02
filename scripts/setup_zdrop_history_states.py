"""Fork the EXACT repaired-Drop snapshot; patch history decisions only."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/tmp/zdrop_eval_20261031');SNAP=Path('/tmp/zdrop_history_states_20261102')
BASE_RESULT=ROOT/'outputs/zdrop_shared_rms_20261031/eth3d/courtyard/rep_combined_l2_direction_adaptive/result.json'

def main():
    old=json.loads(BASE_RESULT.read_text())
    assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h for n,h in old['source_sha256'].items())
    if SNAP.exists():raise ValueError('Fresh fork required')
    shutil.copytree(BASE,SNAP,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['saf3r/patch/zdrop_history_states.py','saf3r/triton/zdrop_history_states.py']:
        shutil.copy2(ROOT/name,SNAP/name)
    name='saf3r/patch/reuse_residual_exploration.py';s=(BASE/name).read_text()
    s='from . import zdrop_history_states as history_rule\n'+s
    needle="                        prior,pi=m5c.build_prior(s,start,old_ids,m5c_observation)\n"
    assert needle in s;s=s.replace(needle,needle+"                        history_context=history_rule.gate(prior,pi,population,old_ids,s.fine.labels,s.fine.reps)\n")
    oldcall='errors,costs=cosine_states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier,geometry=geometry)'
    assert oldcall in s;s=s.replace(oldcall,'errors,costs=history_rule.states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier,geometry,history_context)')
    s=s.replace('compact.pack(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)','history_rule.pack(selected,hids,s.fine.reps,s.fine.labels,population,s.budget,history_context)')
    s=s.replace('compact.transfer(selected,s.fine.labels,s.fine.reps,hids,prior,pi,ids,geometry)','history_rule.transfer(selected,s.fine.labels,s.fine.reps,hids,prior,pi,ids,geometry,history_context)')
    s=s.replace('compact.diagnostics(errors,selected,hids,old_ids,s.fine.labels,s.fine.reps)','history_rule.diagnostics(errors,selected,hids,old_ids,s.fine.labels,s.fine.reps,history_context)')
    s=s.replace("record['history_cap_discarded']=discarded", "record['history_cap_discarded']=torch.zeros_like(discarded)")
    (SNAP/name).write_text(s)
    config=old['config'];(SNAP/'configs/evaluation/reuse_zdrop_history_states.json').write_text(json.dumps(config,indent=2)+'\n')
    (ROOT/'configs/evaluation/reuse_zdrop_history_states.json').write_text(json.dumps(config,indent=2)+'\n')
    s=(BASE/'scripts/evaluate_reuse_exploration.py').read_text().replace('import traceback\n','import traceback\nimport io,tarfile\n')
    s=s.replace("dataset=MV_REGISTRY.get('7scenes')()", "dataset_name=args.dataset\n    dataset=MV_REGISTRY.get(dataset_name)()")
    s=s.replace("if args.method=='reuse_explore':","if args.method=='zdrop_history_states':")
    a=s.index('    sources=[');b=s.index('    samples=[]',a)
    s=s[:a]+'''    sources=json.loads((ROOT/'baseline_source_hashes.json').read_text())
    names=set(sources)|{'saf3r/patch/zdrop_history_states.py','saf3r/triton/zdrop_history_states.py','scripts/evaluate_zdrop_history_states.py','configs/evaluation/reuse_zdrop_history_states.json'}
    hashes={};buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name in sorted(names):
            data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    (out/'source_snapshot.tar').write_bytes(buffer.getvalue())
    save(out/'source_sha256.json',hashes)
'''+s[b:]
    s=s.replace("dataset='7scenes'","dataset=dataset_name").replace("load_scene_data(dataset,'7scenes',","load_scene_data(dataset,dataset_name,")
    s=s.replace("choices=['dense','saf3r','history80','reuse_explore']", "choices=['zdrop_history_states']")
    s=s.replace("p.add_argument('--method',", "p.add_argument('--dataset',choices=['eth3d','7scenes'],default='eth3d')\n    p.add_argument('--method',")
    (SNAP/'scripts/evaluate_zdrop_history_states.py').write_text(s)
    (ROOT/'scripts/evaluate_zdrop_history_states.py').write_text(s)
    (SNAP/'baseline_source_hashes.json').write_text(json.dumps(old['source_sha256'],indent=2)+'\n')
    changed=[n for n,h in old['source_sha256'].items() if hashlib.sha256((SNAP/n).read_bytes()).hexdigest()!=h]
    assert changed==['saf3r/patch/reuse_residual_exploration.py'],changed
    (SNAP/'baseline_fork_audit.json').write_text(json.dumps(dict(baseline=str(BASE_RESULT),changed_existing_files=changed,configuration_identical=True,query_group_size=32,retention=.2,shared_rms_drop=True,squared_l2=True,hierarchical=False,mandatory_keys_added=False),indent=2)+'\n')
    print(SNAP)
if __name__=='__main__':main()
