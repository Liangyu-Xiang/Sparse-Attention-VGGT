"""Aligned actual-observation probability and old critical-Key coverage audit."""
import argparse,csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output;selected=json.loads((root/'current_probe_selection.json').read_text());name=selected['candidates'][0]['name'];folder=root/'current_probe_key_impact_exact_target'/name
 import numpy as np
 new=json.loads((folder/'result.json').read_text());old=json.loads((root/'key_impact_aligned/current/result.json').read_text());assert new['input']==old['input'];bits=np.load(folder/'selected_key_bits.npz');positive=[r for r in old['keys'] if r['fixed_budget_weakest_swap_gain']>0];rows=[]
 for r in positive:
  retained=bool(np.unpackbits(bits[f"L{r['layer']}_G{r['query_group']}"])[r['key_id']]);path=old['input']['image_paths'][r['key_frame']];rows.append(dict(**r,source_image_path=path,source_frame_id=Path(path).stem,retained_by_current_probe=retained))
 covered=[r for r in rows if r['retained_by_current_probe']];errors=[abs(r['estimated_prior_probability']-r['true_dense_probability_observed_mean'])/max(r['true_dense_probability_observed_mean'],1e-30) for r in new['keys'] if r['estimated_prior_probability'] is not None]
 assert max(errors)<.005,'Actual probe target/probability mismatch'
 summary=dict(old_helpful_candidate_edges=len(rows),retained_by_new=len(covered),fraction=len(covered)/len(rows),single_edge_gain_weighted_coverage=sum(r['fixed_budget_weakest_swap_gain'] for r in covered)/sum(r['fixed_budget_weakest_swap_gain'] for r in rows),maximum_relative_probability_error=max(errors),mean_relative_probability_error=statistics.mean(errors),probability_target='Exact same observed4 clamped Probe IDs including repeated tail and special queries,head mean',interpretation='Finite8-per-group candidate pool,support overlap across trajectories;individual counterfactual gains are not additive final geometry effects.',quality=new['quality'])
 prior=root/'current_probe_key_analysis.json'
 if prior.exists() and not (root/'current_probe_key_analysis_unaligned.json').exists():prior.rename(root/'current_probe_key_analysis_unaligned.json')
 save(prior,summary)
 with (root/'critical_keys_current_probe.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(root/'current_probe_key_impact/diagnostic_target_correction.json',dict(reason='Previous group_mean used deduplicated Patch-only diagnostic rows,not the4 clamped actual observed Probe IDs;not comparable at tails/specials. Kernel and production method unchanged. Corrected results in current_probe_key_impact_exact_target.',comparison_file='current_probe_key_analysis.json'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
