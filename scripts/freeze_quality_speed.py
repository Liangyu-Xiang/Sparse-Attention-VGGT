"""Freeze Pareto endpoints using independent RGB evidence, never target GT."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
 groups=[('validation_phase1',['c12_head_sketch_lowflat','c06_head_sketch_flat','c10_head_mean_lowflat']),('validation_phase2_retry',['p2_c8_population_exact1.0_middle']),('validation_phase3',['p3_sketch10_mid21','p3_sketch10_mid20','p3_sketch10_mid19'])]
 rows=[];configs={}
 for stage,names in groups:
  for name in names:
   parent=root/stage/name;monitor=json.loads((parent.parent/(name+'.monitor.json')).read_text());assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
   entry=dict(name=name,phase=stage,per_sequence={},physical_gpu=monitor['gpu_uuid'])
   for ds in ['chess_seq03','tum_walking128']:
    new=json.loads((parent/name/ds/'result.json').read_text());ref=json.loads((parent/'current'/ds/'result.json').read_text());assert new['input']==ref['input'] and new['checkpoint_sha256']==ref['checkpoint_sha256']
    assert new['timing']['warmup']==1 and new['timing']['repeats']==3;assert not new['test_gt_read'] and not new['quality']['complete']
    entry['per_sequence'][ds]=dict(seconds=new['timing']['median_seconds'],reference_seconds=ref['timing']['median_seconds'],time_ratio=new['timing']['median_seconds']/ref['timing']['median_seconds'],score=new['fidelity']['score'],reference_score=ref['fidelity']['score'],error_ratio=new['fidelity']['score']/ref['fidelity']['score'])
    configs[name]=new['config']
   entry.update(score=statistics.mean(r['score'] for r in entry['per_sequence'].values()),speed_pass=all(r['time_ratio']<=1 for r in entry['per_sequence'].values()),fidelity_pass=all(r['error_ratio']<1 for r in entry['per_sequence'].values()))
   rows.append(entry)
 eligible=[r for r in rows if r['speed_pass'] and r['fidelity_pass']];assert eligible,'No validated no-slower,more-faithful candidate'
 best=min(eligible,key=lambda r:(r['score'],r['name']));fast=min(eligible,key=lambda r:(r['per_sequence']['tum_walking128']['time_ratio'],r['name']));chosen=[best]+([] if fast['name']==best['name'] else [fast])
 equivalence=json.loads((root/'dead_observation/result.json').read_text());assert equivalence['passed'];configs_out=[]
 for r in chosen:
  name=r['name'];cfg=dict(configs[name],skip_dead_exact_observation=configs[name].get('candidate_fraction')==1.);p=ROOT/'configs/evaluation/quality_speed_final_20261106'/(name+'.json');save(p,cfg)
  configs_out.append(dict(name=name,config=str(p.relative_to(ROOT)),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='primary;lowest independent RGB error' if r is best else 'alternative;fastest independently improved error',validation=r,engineering_change='Remove only unread all-eligible residual observations; real8-frame numerical comparison passed'))
 frozen=dict(frozen_before_target_quality=True,candidates=configs_out,validation_rows=rows,selection='All independent validation sequences must have lower sampled local error and no higher warmed1+3 median time;freeze lowest mean error and fastest128-frame Pareto endpoints;no target geometry read',global_optimum_claim=False,excluded='GPU7 lost validation;retry on GPU0 only',dense_control_rerun=False,saf3r_control_rerun=False)
 save(root/'final_selection.json',frozen);print(json.dumps(frozen,indent=2))
if __name__=='__main__':main()
