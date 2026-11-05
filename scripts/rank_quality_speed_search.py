"""Rank RGB-only screening; never open target quality metrics."""
import argparse,concurrent.futures,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def rank(root):
 plan=json.loads((root/'plan.json').read_text());rows=[];missing=[]
 paths=[]
 for i,c in enumerate(plan['candidates']):
  for ds in plan['calibration']:paths.append((c['name'],ds,0 if i==0 else i%3,root/'screen'/f'shard{0 if i==0 else i%3}'/c['name']/ds/'result.json'))
 def load(row):
  name,ds,shard,p=row
  try:return row,json.loads(p.read_text())
  except FileNotFoundError:return row,None
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:loaded=list(pool.map(load,paths))
 data={(name,ds):v for (name,ds,_,_),v in loaded if v}
 baselines={}
 for shard in range(3):
  for ds in plan['calibration']:
   p=root/'screen'/f'shard{shard}'/'current'/ds/'result.json'
   if p.exists():baselines[shard,ds]=json.loads(p.read_text())
 for i,c in enumerate(plan['candidates']):
  name=c['name'];shard=0 if i==0 else i%3
  if any((name,ds) not in data or (shard,ds) not in baselines for ds in plan['calibration']):missing.append(name);continue
  rs=[data[name,ds] for ds in plan['calibration']];ratios={ds:data[name,ds]['timing']['median_seconds']/baselines[shard,ds]['timing']['median_seconds'] for ds in plan['calibration']}
  score=statistics.mean(r['fidelity']['score'] for r in rs)
  rows.append(dict(name=name,score=score,per_sequence_error={ds:data[name,ds]['fidelity']['score'] for ds in plan['calibration']},time_ratio_vs_current=ratios,speed_feasible=all(r<=1 for r in ratios.values()),mean_retention=statistics.mean(c['config']['layer_retentions']),cluster_fraction=c['config']['cluster_fraction'],features=c['config']['shared_value_features']))
 rows.sort(key=lambda r:(r['score'],r['name']))
 result=dict(complete=not missing,missing=missing,rows=rows,selection_uses_target_gt=False,criterion=plan['ranking']);save(root/'screen_rank.json',result)
 if not missing:
  eligible=[r for r in rows if r['name']!='current' and r['speed_feasible']];assert eligible
  finalists=eligible[:2]
  low=next((r for r in eligible if r['mean_retention']<=.12+1e-9 and r not in finalists),None)
  if low:finalists.append(low)
  if len(finalists)<3:finalists+=eligible[len(finalists):3]
  save(root/'finalists.json',dict(candidates=finalists,rule='Best2 score among no-slower screen candidates plus best distinct <=12% mean retention candidate; else next score; validation still no GT'))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=rank(a.output);print(json.dumps(dict(complete=r['complete'],finished=len(r['rows']),missing=len(r['missing']),best=r['rows'][:8]),indent=2))
