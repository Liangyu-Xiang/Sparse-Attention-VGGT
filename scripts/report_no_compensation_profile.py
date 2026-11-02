"""Group exclusive scope slices once; never add inclusive and child times."""
import collections,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/no_mass_compensation_profile_20261102'
def main():
 d=json.loads((OUT/'profile.json').read_text());events={r['id']:r for r in d['events']};totals=collections.defaultdict(float)
 mapping={'attention_core_fused_grid':'稀疏Attention、探针导出与grid累积：融合内核','attention_grid_normalize_reduce':'Attention grid统计归一化与归约','attention_wrapper':'Attention Key分组及执行准备','value_update':'当前Value聚类：含特征平均及全部子操作','produce_history':'本层历史生产：归一化、grid校准、head-mean及缓存','consume_history':'历史消费：结合当前grid迁移、展开prior并汇总簇概率','state_states':'Rep/History等状态表示损失与组装','state_gate':'历史候选资格和Rep身份判断','state_pack':'支持集打包与合法性检查','state_diagnostics':'状态诊断统计','coarse_observation':'旧coarse观测','residual_fine_probe':'残差与Exact候选筛选','m5c_coarse_observation':'当前M5c grid及特殊Token观测','pooling':'池化与簇几何准备','reuse_and_budget':'Drop代价、预算求解及其他分配准备','history_summary':'旧grid证据汇总与历史ID缓存','execution_and_evidence':'Attention调用其余开销','global_attention':'其余全局Attention：含QKV、RoPE、投影和调度'}
 for r in events.values():
  cur=r
  while cur['name'] not in mapping:
   if cur['parent'] is None:raise AssertionError(cur['name'])
   cur=events[cur['parent']]
  totals[mapping[cur['name']]]+=r['exclusive_seconds']
 assert abs(sum(totals.values())-d['global_attention_seconds'])<1e-7
 core=d['groups']['attention_core_fused_grid']['exclusive_seconds'];rows=[dict(operation=k,seconds=v,times_core=v/core) for k,v in sorted(totals.items(),key=lambda kv:-kv[1])]
 with (OUT/'exclusive_operations.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text=['# 取消补偿版 heads128 独占计时','',f"插桩完整前向 {d['wall_seconds']:.6f}s；24层全局Attention {d['global_attention_seconds']:.6f}s。一次预热后一次profile；不是三次完整前向中位数。",'', '每个scope扣除直接子scope后再分类，每个独占时间片只归一项；融合内核不拆成假定的独立时间。本表包含全局Attention里的QKV/RoPE/投影，均只在最后的其余项中统计，不属于mask构建。', '', '|互斥操作|秒|核心融合内核倍数|','|---|---:|---:|']
 text += [f"|{r['operation']}|{r['seconds']:.6f}|{r['times_core']:.3f}×|" for r in rows]
 text+=['', f"互斥分项总和 {sum(totals.values()):.9f}s；差额 {sum(totals.values())-d['global_attention_seconds']:.12g}s；父子重复时间已扣除。",'', '聚类细分仅用于解释聚类父项，不能再次加到上表：邻簇图0.446518s、两轮局部分配0.400050s、L0两轮全中心搜索0.168358s、空簇修复0.014831s、中心归约0.009864s，其余为特征平均和控制等。', '', '历史生产：L0–22，各66个窗口，共1518次；历史消费：L1–23，各66个窗口，共1518次。前者得到本层条件统计，后者结合新一层观测估计新一层概率。它们是不同的真实执行段，不能将同一生产调用再记入后一层消费。', '', '六项质量指标均N/A：本次为计时诊断，未评估GT。Dense相对加速比N/A：未运行不变对照。']
 (OUT/'report.md').write_text('\n'.join(text)+'\n')
 print('Exclusive groups sum:',sum(totals.values()),'core:',core)
if __name__=='__main__':main()
