"""Summarize diagnostic counterfactuals without opening target test quality."""
import argparse,csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output;folder=root/'key_impact_aligned';current=json.loads((folder/'current/result.json').read_text());selection=json.loads((root/'final_selection.json').read_text());name=selection['candidates'][0]['name'];new=json.loads((folder/name/'result.json').read_text());assert current['input']==new['input']
 import numpy as np
 newbits=np.load(folder/name/'selected_key_bits.npz');oldbits=np.load(folder/'current/selected_key_bits.npz');summary={};rows=[]
 for method,record in [('current',current),(name,new)]:
  keys=record['keys'];assert len(keys)==1536;positive=[r for r in keys if r['fixed_budget_weakest_swap_gain']>0];ratios=[r['true_dense_probability_probe0']/r['estimated_prior_probability'] for r in positive if r['estimated_prior_probability'] and r['layer']>0];dist=[r['head_mean_value_distance_sq']/r['per_head_value_distance_sq'] for r in positive if r['per_head_value_distance_sq']>0]
  summary[method]=dict(candidate_edges=len(keys),positive_single_edge_swaps=len(positive),positive_approximate_cluster=sum(r['omitted_reason']=='approximate_cluster' for r in positive),positive_dropped_cluster=sum(r['omitted_reason']=='dropped_cluster' for r in positive),positive_never_previous_history=sum(not r['was_history'] for r in positive),median_probe0_true_over_prior=statistics.median(ratios),fraction_probe0_underestimated_gt10=sum(x>10 for x in ratios)/len(ratios),median_head_mean_distance_over_headwise_distance=statistics.median(dist),mean_local_relative_mse=statistics.mean(r['projected_relative_mse'] for r in record['layers']))
  if method=='current':
   for r in positive:
    slot=f"L{r['layer']}_G{r['query_group']}";key=r['key_id'];assert not bool(np.unpackbits(oldbits[slot])[key]);retained=bool(np.unpackbits(newbits[slot])[key]);rows.append(dict(**r,retained_by_frozen_primary=retained))
 total=sum(r['fixed_budget_weakest_swap_gain'] for r in rows);covered=[r for r in rows if r['retained_by_frozen_primary']];coverage=dict(old_positive_candidate_edges=len(rows),retained_by_new=len(covered),fraction=len(covered)/len(rows),single_edge_gain_weighted_coverage=sum(r['fixed_budget_weakest_swap_gain'] for r in covered)/total,interpretation='Conditional coverage of8-per-group candidate pool;single-edge gains are not additive joint output improvement. Trajectories differ, so this is support overlap, not an isolated causal estimate of final geometry.')
 result=dict(complete=True,diagnostic=True,quality=current['quality'],summary=summary,coverage=coverage,top_current_examples=sorted(rows,key=lambda r:-r['fixed_budget_weakest_swap_gain'])[:20],target_gt_read=False,definition='Same-QKV full-key Dense probabilities,Probe0 matched with predicted Probe0 prior. Four-query group mean reported separately. Counterfactual changes one key at fixed edge count,not guaranteed legal under four cluster states.')
 save(root/'key_impact_analysis.json',result)
 with (root/'critical_keys.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# 关键Key：独立RGB上的固定边数反事实','',json.dumps(summary,indent=2,ensure_ascii=False),'',json.dumps(coverage,indent=2,ensure_ascii=False),'', '此分析固定ScanNet64帧、24层、8组×4Patch Query；每组仅考察8条高概率遗漏Key。真实概率是同一稀疏轨迹QKV下的完整Key Softmax，不是独立Dense模型轨迹。估计prior针对Probe0，因此只与Probe0真实概率对齐；四Probe均值另存。','', '单Key替换保持边数不变，精确重算当前补偿加权Softmax输出及Wo投影误差。它可能超出原四状态可表达的Mask；收益是当前层局部的，不能当作最终相机/F1因果收益，也不能累加当作多Key共同收益。','', 'head_mean距离小于逐Head距离，说明跨Head抵消可能掩盖Value差异；新Head草图保留Head分离方向。遗漏原因区分为：整个簇被Drop，或簇仍有Rep但漏掉具体Key。后者说明仅增加簇数/只确保每簇一Rep不足。无历史边的新峰值也不可能仅靠历史概率校准找回。','', 'critical_keys.csv记录层、Query组、原帧与Patch坐标、所属簇/Rep、是否上一层历史Key、Probe0真概率与估计、固定预算替换收益及新方案是否实际保留。所有六项几何指标N/A，未读取GT。']
 (root/'key_impact_report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summary,coverage=coverage),indent=2))
if __name__=='__main__':main()
