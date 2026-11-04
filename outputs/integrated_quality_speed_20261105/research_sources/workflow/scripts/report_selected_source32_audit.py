"""Current selected-method closed ledger vs explicitly historical references."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_deep_saf3r_query256 as ledger
OUT=ROOT/'outputs/selected_source32_audit_20261105';OLD=Path('/data_NVME1/mmc_lyxiang/3D/SAF3R-query-group-scaling/outputs/deep_saf3r_query256_20261105_final')
SCENES=('heads','chess','stairs');VARIANT='source_sequential_w32_t1'

def node(root,name):
 if root.name==name:return root
 for child in root.children:
  result=node(child,name)
  if result:return result

def deserialize(d):return ledger.Node(d['name'],d['seconds_raw'],[deserialize(c) for c in d['children']],d['kind'])
def serialize(n):return dict(name=n.name,seconds_raw=n.seconds,integer_microseconds=n.us,kind=n.kind,children=[serialize(c) for c in n.children])
def selected_tree(d,p):
 tree=ledger.build_tree(d,p);get=lambda name:p['intervals'].get(name,{}).get('cuda_interval_seconds',0.)
 cluster=node(tree,'Value聚类')
 cluster.children=ledger.parent(cluster.name,cluster.seconds,[ledger.leaf(label,get(key)) for label,key in (
  ('各头V均值与patch特征','cluster/head_mean_value_features'),('特征中心化与padding','source_cluster/feature_centering_padding'),
  ('簇中心与人口重算','source_cluster/centers_population'),('source内成员距离、分配与写回','source_cluster/member_assignment'),
  ('source内空簇修复','source_cluster/empty_repair'),('最终真实代表点更新','source_cluster/final_real_representatives'))]).children
 other=node(tree,'模型其余：初始化、布局、未细分区间')
 other.children=ledger.parent(other.name,other.seconds,[ledger.leaf('一次性固定source布局构建',get('source_cluster/initial_source_layout'))]).children
 return tree

def top_rows(tree):
 attention=node(tree,'Global attention');rows=[]
 for name in ('支持集构建','支持集上的attention执行','历史概率统计与补全','QKV与输出投影'):rows.append((name,node(attention,name).us))
 rows.append(('Global attention其余',attention.us-sum(us for _,us in rows)))
 rows += [(c.name,c.us) for c in tree.children if c.name!='Global attention']
 assert sum(us for _,us in rows)==tree.us
 return rows

def leaf_report(path,trees,names):
 maps=[{n:us for n,us,_ in ledger.leaves(t)} for t in trees];keys=sorted(set().union(*(set(m) for m in maps)))
 rows=[[key]+[m.get(key,0) for m in maps] for key in keys]
 assert all(sum(row[i+1] for row in rows)==t.us for i,t in enumerate(trees))
 with path.with_suffix('.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['operation']+[n+'_integer_us' for n in names]);w.writerows(rows)
 lines=['以下均为互斥叶子项，各列相加恰好等于该列完整前向wall时间。历史列不是本批配对测速。','', '| 操作 | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
 for row in rows:lines.append('| '+row[0]+' | '+' | '.join(f'{v/1e6:.6f}' for v in row[1:])+' |')
 lines.append('| **全部叶子合计＝完整前向** | '+' | '.join(f'**{t.us/1e6:.6f}**' for t in trees)+' |')
 path.with_suffix('.md').write_text('\n'.join(lines)+'\n')

old=json.loads((OLD/'summary.json').read_text())
old_sf=ledger.quantize(deserialize(old['mean_tree']['saf3r']));old_adaptive=ledger.quantize(deserialize(old['mean_tree']['shared_value_q256']))
newtrees={};runs={};selected={};monitors={};missing=[]
for scene in SCENES:
 p=OUT/scene/VARIANT/'result.json';mp=OUT/f'{scene}_{VARIANT}.monitor.json'
 if not p.exists() or not mp.exists():missing.append(scene);continue
 d=json.loads(p.read_text());m=json.loads(mp.read_text());assert m['returncode']==0 and not m['same_gpu_overlap_detected']
 assert d['deep_validation']['source_clustering_scopes_preserve_outputs_cpu']
 config=json.loads((p.parent/'config.json').read_text())['shared_config']
 assert config==json.loads((OUT/'selected_config.json').read_text())
 for method in ('saf3r','shared_value_q256'):
  reference=json.loads((OLD/scene/method/'result.json').read_text())
  for key in ('image_paths','frame_indices','shape'):assert d['input'][key]==reference['input'][key],(scene,key)
  assert Path(d['input']['checkpoint']).resolve()==Path(reference['input']['checkpoint']).resolve()
 idx=sorted(range(3),key=lambda i:d['event_profiles'][i]['wall_seconds'])[1]
 tree=ledger.quantize(selected_tree(d,d['event_profiles'][idx]));assert sum(us for _,us,_ in ledger.leaves(tree))==tree.us
 newtrees[scene]=tree;runs[scene]=d;selected[scene]=idx;monitors[scene]=m
 oldper=[ledger.quantize(deserialize(old['per_scene'][scene][method])) for method in ('saf3r','shared_value_q256')]
 leaf_report(OUT/f'leaf_ledger_{scene}',oldper+[tree],['历史SAF3R','历史adaptive','本批连续source32'])
if missing:
 print(json.dumps(dict(complete=False,missing=missing)));sys.exit(0)
mean=ledger.quantize(ledger.combine([newtrees[s] for s in SCENES]));assert sum(us for _,us,_ in ledger.leaves(mean))==mean.us
leaf_report(OUT/'leaf_ledger_mean',[old_sf,old_adaptive,mean],['历史SAF3R','历史adaptive','本批连续source32'])
normal={scene:runs[scene]['normal_warmed_median_seconds'] for scene in SCENES};normal_mean=statistics.mean(normal.values())
quality_rows=[]
for scene in SCENES:
 p=ROOT/'outputs/source_local_quality_20261105'/scene/VARIANT/'result.json';q=json.loads(p.read_text())
 for key in ('image_paths','frame_indices','shape'):assert runs[scene]['input'][key]==q['input'][key]
 assert Path(runs[scene]['input']['checkpoint']).resolve()==Path(q['input']['checkpoint']).resolve()
 quality_rows.append(dict(scene=scene,**q['quality']['values']))
fields=('auc03','auc30','acc','comp','cd','f1');quality_mean={k:statistics.mean(r[k] for r in quality_rows) for k in fields}
summary=dict(complete=True,selected_config=json.loads((OUT/'selected_config.json').read_text()),scenes=SCENES,frames=128,
 current_normal_warmed_median_seconds=normal,current_normal_three_scene_mean_seconds=normal_mean,selected_profile_index=selected,
 current_mean_tree=serialize(mean),current_per_scene_trees={s:serialize(t) for s,t in newtrees.items()},
 historical_reference_trees=dict(saf3r=serialize(old_sf),adaptive=serialize(old_adaptive)),historical_reference_batch=str(OLD),comparison_is_contemporaneously_paired=False,
 monitors=monitors,closure=dict(wall_us=mean.us,leaf_sum_us=sum(us for _,us,_ in ledger.leaves(mean)),difference_us=0),
 quality_evaluated=False,quality={k:None for k in fields},quality_unavailable='Explicit detailed timing diagnostic; no pose/recon_unposed evaluated in this batch',
 historical_selected_quality=dict(batch='source_local_quality_20261105',per_scene=quality_rows,three_scene_mean=quality_mean),
 dense_speedup=None,dense_speedup_reason='Unchanged Dense not rerun under AGENTS.md; no current paired full-forward timing')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 已选定：连续编号source32、无持久锚点，128帧完整耗时账本','',
 '当前工作区默认 configs/evaluation/reuse_shared_value_loss_q256.json 已指向连续编号 source32、无持久锚点版本；旧 adaptive_local 配置保存为 reuse_shared_value_loss_q256_adaptive_reference.json。Query group=256、Key上限20%、fine簇26009、source宽度32簇、聚类迭代2、历史采样间隔256，其余保持不变。','',
 '只测 heads/chess/stairs 各128帧，不是完整七场景宏平均。每个纳入运行的严格GPU监控均无外来任务重叠。','',
 f'**本批正常测速完整前向均值：{normal_mean:.6f}秒。** 每序列1次预热+3次完整前向，取热运行中位数后对三个序列等权平均。包含初始化、布局、聚类、mask、history、attention和输出头，不含图片读取或后续质量评估。','',
 f'**以下细分账本总时间：{mean.us/1e6:.6f}秒。** 细分插桩会扰动运行；每序列在另外3次统计前向中选择整个wall时间居中的一次并取全部明细，最后等权平均。不得与正常测速的总时间拼接或缩放。','',
 'SAF3R和旧adaptive列均复用历史 deep_saf3r_query256_20261105_final 带统计结果，只作历史参照；没有重跑未改动的对照，不能称为本批配对加速或将所有差值归因于聚类。输入帧列表、checkpoint、预处理形状已核对一致；历史SAF3R为lazy首个对应头Top-4，不是论文DINO Top-K。','',
 '## 完整总账：每一列各行直接相加','', '| 操作 | 历史SAF3R秒 | 历史adaptive秒 | 本批连续source32秒 |','|---|---:|---:|---:|']
top=[top_rows(t) for t in (old_sf,old_adaptive,mean)]
assert all([x[0] for x in rows]==[x[0] for x in top[0]] for rows in top)
for group in zip(*top):lines.append('| '+group[0][0]+' | '+' | '.join(f'{v/1e6:.6f}' for _,v in group)+' |')
lines += ['| **完整前向＝以上各行之和** | '+' | '.join(f'**{t.us/1e6:.6f}**' for t in (old_sf,old_adaptive,mean))+' |','',
 '自动校验：三列互斥明细与各自完整wall时间差均为0微秒。末尾包含显式剩余/边界项；整数微秒取整差保留在剩余中。','',
 '## 本批支持集构建展开（已包含在上表，不要重复相加）','', '| 操作 | 秒 |','|---|---:|']
support=node(mean,'支持集构建')
for c in support.children:lines.append(f'| {c.name} | {c.us/1e6:.6f} |')
lines.append(f'| **支持集构建合计** | **{support.us/1e6:.6f}** |')
def expand(parent,depth=0):
 lines=[]
 for c in parent.children:
  lines.append(f'| {"　"*depth}{c.name} | {c.us/1e6:.6f} | {"含下级子项" if c.children else c.kind} |')
  if c.children:lines+=expand(c,depth+1)
 return lines
for name in ('Value聚类','池化与共享几何','粗感知','残差候选筛选','prior、损失、预算与打包'):
 n=node(support,name);lines+=['',f'## {name}：{n.us/1e6:.6f}秒','', '| 操作 | 秒 | 包含关系 |','|---|---:|---|']+expand(n)
lines+=['','一次性source布局构建位于“模型其余”内，未计入Value聚类；该子项为 '+f"{node(mean,'一次性固定source布局构建').us/1e6:.6f}"+' 秒。已删除全局邻接图、L0全局分配和认证边界计算。','',
 '## 每序列正常测速与带统计总时间','', '| 序列 | 正常热中位数秒 | 带统计完整wall秒 | Value聚类秒 | 峰值allocated GiB（热正常运行最大） |','|---|---:|---:|---:|---:|']
for scene in SCENES:
 peak=max(r['peak_allocated_gib'] for r in runs[scene]['normal_samples'][1:])
 lines.append(f"| {scene} | {normal[scene]:.6f} | {newtrees[scene].us/1e6:.6f} | {node(newtrees[scene],'Value聚类').us/1e6:.6f} | {peak:.3f} |")
lines+=['','## 六项质量及Dense速度比','',
 '本批3次运行明确为细粒度计时诊断，未执行pose/recon_unposed；每个运行AUC@3、AUC@30、ACC、Comp、CD、F1均N/A，原因在各自quality.json/quality.md。Dense加速比均N/A：未重跑未改动的Dense，没有本批配对完整前向。','',
 '| 本批诊断 | AUC@3 | AUC@30 | ACC | Comp | CD | F1 | vs Dense |','|---|---|---|---|---|---|---|---|']
for scene in SCENES:lines.append(f'| {scene} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
lines+=['','以下单列复用上一轮同配置完整质量结果，批次 source_local_quality_20261105；不是本批新测。AUC/F1为百分数，距离米，CD=(ACC+Comp)/2非平方距离；重建F1阈值0.05m，下采样0.0078125m。质量来自每次末轮输出，浮点原子归约和离散预算存在非确定性，单次差值不能视为稳定提升。','',
 '| 质量历史记录 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
for q in quality_rows+[dict(scene='三序列等权平均',**quality_mean)]:lines.append('| '+q['scene']+' | '+' | '.join(f"{q[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in fields)+' |')
lines+=['','## 原始文件','', 'leaf_ledger_mean.md/csv 与 leaf_ledger_{scene}.md/csv 只列互斥叶子项，每列相加等于该列完整wall时间。summary.json包含完整树、原始精度引用与监控。每个scene/source_sequential_w32_t1目录保存逐次计时、逐层事件、输入列表、配置、代码快照和明确N/A的六指标诊断记录。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=True,normal_mean_seconds=normal_mean,profiled_mean_seconds=mean.us/1e6,cluster_seconds=node(mean,'Value聚类').us/1e6,closure_us=0)))
