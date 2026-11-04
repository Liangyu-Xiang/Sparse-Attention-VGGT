"""Closed operation ledgers for new clustering, with historical scope reference."""
import csv,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import report_deep_saf3r_query256 as ledger
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source_clustering_detail_20261105'
SCENES=('heads','chess','stairs');VARIANTS=('source_sequential_w32_t1','source_spatiotemporal_w32_t2')

def getnode(node,name):
 if node.name==name:return node
 for c in node.children:
  found=getnode(c,name)
  if found:return found

def newtree(d,p):
 tree=ledger.build_tree(d,p)
 ev={k:v['cuda_interval_seconds'] for k,v in p['intervals'].items()};get=lambda k:ev.get(k,0.)
 old=getnode(tree,'Value聚类')
 replacement=ledger.parent(old.name,old.seconds,[ledger.leaf(label,get(key)) for label,key in (
  ('各头V均值与patch特征','cluster/head_mean_value_features'),
  ('特征中心化与padding','source_cluster/feature_centering_padding'),
  ('簇中心与人口重算','source_cluster/centers_population'),
  ('source内成员距离、分配与写回','source_cluster/member_assignment'),
  ('source内空簇修复','source_cluster/empty_repair'),
  ('最终真实代表点更新','source_cluster/final_real_representatives'))])
 old.children=replacement.children
 other=getnode(tree,'模型其余：初始化、布局、未细分区间')
 other.children=ledger.parent(other.name,other.seconds,[ledger.leaf('一次性固定source布局构建',get('source_cluster/initial_source_layout'))]).children
 return tree

def serialize(n):return dict(name=n.name,seconds_raw=n.seconds,integer_microseconds=n.us,kind=n.kind,children=[serialize(c) for c in n.children])
rows=[];means={};trees={};monitors={};missing=[]
for variant in VARIANTS:
 trees[variant]={}
 for scene in SCENES:
  path=OUT/scene/variant/'result.json';monitorpath=OUT/f'{scene}_{variant}.monitor.json'
  if not path.exists() or not monitorpath.exists():missing.append(f'{scene}/{variant}');continue
  d=json.loads(path.read_text());monitor=json.loads(monitorpath.read_text())
  assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
  assert d['deep_validation']['source_clustering_scopes_preserve_outputs_cpu']
  qualityref=json.loads((ROOT/'outputs/source_local_quality_20261105'/scene/variant/'result.json').read_text())
  for k in ('image_paths','frame_indices','shape'):assert d['input'][k]==qualityref['input'][k]
  assert Path(d['input']['checkpoint']).resolve()==Path(qualityref['input']['checkpoint']).resolve()
  idx=sorted(range(3),key=lambda i:d['event_profiles'][i]['wall_seconds'])[1]
  p=d['event_profiles'][idx];tree=ledger.quantize(newtree(d,p));assert sum(us for _,us,_ in ledger.leaves(tree))==tree.us
  trees[variant][scene]=tree;cluster=getnode(tree,'Value聚类');initial=getnode(tree,'一次性固定source布局构建')
  row=dict(scene=scene,variant=variant,selected_profile_index=idx,cluster_seconds=cluster.us/1e6,cluster_suboperations={n.name:n.us/1e6 for n in cluster.children},initial_source_layout_seconds=initial.us/1e6,
   profiled_full_forward_seconds=tree.us/1e6,normal_warmed_median_seconds=d['normal_warmed_median_seconds'],
   all_profile_cluster_seconds=[r['intervals']['shared_stage/value_update']['cuda_interval_seconds'] for r in d['event_profiles']],
   quality={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},quality_unavailable_reason='Explicit operation profiling diagnostic; quality from prior completed campaign, not measured here',
   dense_speedup=None,dense_speedup_reason='No paired Dense timing; unchanged control not rerun')
  rows.append(row);monitors[f'{scene}/{variant}']=monitor
  leafrows=ledger.leaves(tree)
  with (OUT/f'ledger_{scene}_{variant}.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['operation','microseconds','kind']);writer.writerows(leafrows)
  table=['| 操作（互斥叶子项） | 秒 |','|---|---|']+[f'| {name} | {us/1e6:.6f} |' for name,us,_ in leafrows]+[f'| **完整前向＝以上叶子项之和** | **{tree.us/1e6:.6f}** |']
  (OUT/f'ledger_{scene}_{variant}.md').write_text('\n'.join(table)+'\n')
 if len(trees[variant])==3:
  mean=ledger.quantize(ledger.combine([trees[variant][scene] for scene in SCENES]));assert sum(us for _,us,_ in ledger.leaves(mean))==mean.us
  means[variant]=mean
  leafrows=ledger.leaves(mean)
  with (OUT/f'ledger_mean_{variant}.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['operation','microseconds','kind']);writer.writerows(leafrows)
  lines=['| 操作（互斥叶子项） | 秒 |','|---|---|']+[f'| {name} | {us/1e6:.6f} |' for name,us,_ in leafrows]+[f'| **完整前向＝以上叶子项之和** | **{mean.us/1e6:.6f}** |']
  (OUT/f'ledger_mean_{variant}.md').write_text('\n'.join(lines)+'\n')
reference=2.098010
summary=dict(complete=not missing,missing=missing,per_scene=rows,three_scene_means={v:dict(cluster_seconds=getnode(t,'Value聚类').us/1e6,initial_source_layout_seconds=getnode(t,'一次性固定source布局构建').us/1e6,profiled_full_forward_seconds=t.us/1e6,cluster_suboperations={c.name:c.us/1e6 for c in getnode(t,'Value聚类').children},tree=serialize(t),leaf_sum_us=sum(us for _,us,_ in ledger.leaves(t)),wall_us=t.us) for v,t in means.items()},monitors=monitors,
 historical_reference=dict(cluster_seconds=reference,batch='deep_saf3r_query256_20261105_final',paired=False,note='Historical instrumented scope reference; not contemporaneous speedup'),
 quality={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},quality_unavailable_reason='Explicit operation profiling; do not present as new quality evaluation',dense_speedup=None,dense_speedup_reason='No paired Dense full-forward timing')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 新聚类的 Value 聚类操作耗时','',f'有效完成 {len(rows)}/6 个诊断。只测 heads、chess、stairs 各128帧，Query group=256，source候选簇上限32，迭代2次。','',
 '沿用历史深度统计方法：每序列3次统计完整前向，选择整个wall时间居中的那一次并使用其全部阶段数据，最后做三个序列等权平均。没有逐项混合不同运行的中位数。CUDA Event区间包含CPU提交间隙，插桩会扰动运行；不是纯kernel self time。','',
 '原来的2.098010秒是同口径历史统计参照，不是本批配对基线；不要把比值称为同期加速。该阶段统计24层各头V均值、聚类更新与真实代表点选择，不含reset时的一次性初始布局。新source布局构建另列，并包含在完整前向账本中。','',
 '| 序列 | 连续编号 source32：Value聚类秒 | 时空 source32、2帧分箱：Value聚类秒 |','|---|---|---|']
for scene in SCENES:
 vals=[]
 for v in VARIANTS:
  r=next((r for r in rows if r['scene']==scene and r['variant']==v),None);vals.append(f"{r['cluster_seconds']:.6f}" if r else 'pending')
 lines.append('| '+' | '.join([scene]+vals)+' |')
if len(means)==2:
 lines.append('| **三序列均值** | '+' | '.join(f"**{getnode(means[v],'Value聚类').us/1e6:.6f}**" for v in VARIANTS)+' |')
 lines+=['','## Value聚类内部：互斥子项精确相加','', '| 操作 | 连续编号source32秒 | 时空source32/2帧秒 |','|---|---|---|']
 for ca,cb in zip(getnode(means[VARIANTS[0]],'Value聚类').children,getnode(means[VARIANTS[1]],'Value聚类').children):lines.append(f'| {ca.name} | {ca.us/1e6:.6f} | {cb.us/1e6:.6f} |')
 lines+=['','全局中心邻接图、全局L0分配和认证边界计算已从新算法移除；保留固定source内的分配与局部空簇修复。这是算法候选范围变化，不是仅替换一个等价kernel。','',
 '| 单列操作 | 连续编号source32秒 | 时空source32/2帧秒 |','|---|---|---|', '| 一次性固定source布局构建（不属于上表Value聚类） | '+' | '.join(f"{getnode(means[v],'一次性固定source布局构建').us/1e6:.6f}" for v in VARIANTS)+' |']
lines+=['','每序列完整账本见 ledger_*.md/csv；所有互斥叶子项之和恰好等于对应那一次完整前向wall时间。不能把这些统计阶段拼到上一轮正常测速11.071秒上。','',
 '这些是诊断运行，AUC@3、AUC@30、ACC、Comp、CD、F1全部N/A：本次未运行pose/recon_unposed。完整六指标质量结果在 ../source_local_quality_20261105/report_zh.md，本次不重复当作新测质量。Dense加速比N/A：没有本批配对Dense完整前向。','',
 '本次没有重跑未改动的SAF3R或Dense。每个纳入结果都要求严格GPU监控无其他进程重叠。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=not missing,valid_runs=len(rows),means={v:r['cluster_seconds'] for v,r in summary['three_scene_means'].items()})))
