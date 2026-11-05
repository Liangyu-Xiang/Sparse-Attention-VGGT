"""Freeze one new probability-backend candidate using independent RGB only."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',type=int,choices=[5,6],default=5);a=p.parse_args();root=a.output;finalists=json.loads((root/f'phase{a.phase}/finalists.json').read_text());rows=[];configs={}
 for c in finalists['candidates']:
  name=c['name'];parent=root/f'validation_phase{a.phase}'/name;m=json.loads((parent.parent/(name+'.monitor.json')).read_text());assert m['returncode']==0 and not m['same_gpu_overlap_detected'];entry=dict(name=name,gpu_uuid=m['gpu_uuid'],per_sequence={})
  for ds in ['chess_seq03','tum_walking128']:
   new=json.loads((parent/name/ds/'result.json').read_text());old=json.loads((parent/'current'/ds/'result.json').read_text());assert new['input']==old['input'];assert new['fidelity']['heldout_queries'] and old['fidelity']['heldout_queries'];assert new['timing']['repeats']==3 and new['timing']['warmup']==1
   entry['per_sequence'][ds]=dict(seconds=new['timing']['median_seconds'],reference_seconds=old['timing']['median_seconds'],time_ratio=new['timing']['median_seconds']/old['timing']['median_seconds'],score=new['fidelity']['score'],reference_score=old['fidelity']['score'],error_ratio=new['fidelity']['score']/old['fidelity']['score']);configs[name]=new['config']
  entry.update(score=statistics.mean(v['score'] for v in entry['per_sequence'].values()),speed_pass=all(v['time_ratio']<=1 for v in entry['per_sequence'].values()),fidelity_pass=all(v['error_ratio']<1 for v in entry['per_sequence'].values()));rows.append(entry)
 eligible=[r for r in rows if r['speed_pass'] and r['fidelity_pass']];assert eligible,'No no-slower,lower-error validated current-probe candidate'
 best=min(eligible,key=lambda r:(r['score'],r['name']));name=best['name'];p=ROOT/('configs/evaluation/current_probe_final_20261106' if a.phase==5 else 'configs/evaluation/current_probe_budget_final_20261106')/(name+'.json');save(p,configs[name])
 frozen=dict(frozen_before_target_quality=True,family='current exact sampled Dense probabilities+History A',candidates=[dict(name=name,config=str(p.relative_to(ROOT)),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='Primary:lowest independent held-out-query error under both validation latency limits',validation=best)],validation_rows=rows,selection='No target test quality used;no slower on each independent sequence,lower held-out-query local error on both,lowest equal-sequence mean error',global_optimum_claim=False,dense_rerun=False,saf3r_rerun=False)
 save(root/('current_probe_selection.json' if a.phase==5 else 'current_probe_budget_selection.json'),frozen);print(json.dumps(frozen,indent=2))
if __name__=='__main__':main()
