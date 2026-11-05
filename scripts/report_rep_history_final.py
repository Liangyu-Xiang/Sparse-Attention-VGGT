"""Final six-metric, timing, state-coverage and same-QKV delivery for History A/B."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_loss_20261105';FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def mean(v):return statistics.mean(v)
def ratio(n,d):return n/d if d>0 else None
def profiles():
 rows=[];mask=[]
 for method in ['baseline','A','B']:
  d=read(OUT/'profiles'/f'{method}.json');samples=d['samples']
  for layer in range(24):
   records=[s['layers'][layer] for s in samples]
   def avg(k):return mean(r.get(k,0.) for r in records)
   row=dict(method=method,layer=layer,scope='heads128; two instrumented diagnostic forwards, equal Query-group means',actual_key_density=avg('actual_unique_density'))
   if layer==0:
    row.update(unavailable='L0 is native seed initialization; four-state History-loss accounting not applicable')
   else:
    clusters=avg('fine_clusters');h=avg('history_chosen_clusters_mean');pi=avg('history_selected_cluster_probability_mean')
    for state,name in [(0,'drop'),(1,'rep'),(5,'history'),(10,'exact')]:row[name+'_cluster_fraction']=avg(f'state_{state}_clusters_mean')/clusters
    row.update(history_mean_keys=ratio(avg('history_support_keys_mean'),h),history_input_mean_keys=ratio(avg('history_input_keys_mean'),h),history_input_contains_rep_fraction=ratio(avg('history_input_rep_clusters_mean'),h),history_input_only_rep_fraction=ratio(avg('history_input_only_rep_clusters_mean'),h),history_support_contains_rep_fraction=ratio(avg('history_support_rep_clusters_mean'),h),history_support_only_rep_fraction=ratio(avg('history_support_only_rep_clusters_mean'),h),history_input_probability=avg('history_selected_input_probability_mean'),history_support_probability=avg('history_selected_support_probability_mean'),history_cluster_probability=pi,history_input_probability_coverage=ratio(avg('history_selected_input_probability_mean'),pi),history_support_probability_coverage=ratio(avg('history_selected_support_probability_mean'),pi),drop_key_cost=avg('drop_key_cost_mean'),rep_key_cost=avg('rep_key_cost_mean'),history_key_cost=avg('history_key_cost_mean'),exact_key_cost=avg('exact_key_cost_mean'),actual_key_count=avg('counts_mean'),patch_budget=avg('patch_budget'))
    assert abs(sum(row[n+'_cluster_fraction'] for n in ['drop','rep','history','exact'])-1)<1e-5
    assert abs(sum(row[n+'_key_cost'] for n in ['drop','rep','history','exact'])-row['actual_key_count'])<.01
   rows.append(row)
  production=read(OUT/'profiles_quiet'/f'{method}.json')
  assert not production['config']['config'].get('rep_history_diagnostics',False)
  assert all(s['graph_creations']==0 and s['graph_evictions']==0 for s in production['samples'])
  mask.append(dict(method=method,mask_pipeline_gpu_ms=statistics.median(s['mask_pipeline_gpu_ms'] for s in production['samples']),selection_prior_budget_gpu_ms=statistics.median(s['selection_prior_budget_gpu_ms'] for s in production['samples']),mask_samples_ms=[s['mask_pipeline_gpu_ms'] for s in production['samples']],selection_samples_ms=[s['selection_prior_budget_gpu_ms'] for s in production['samples']]))
 save(OUT/'state_layers.json',dict(scope='heads128 diagnostic, not ETH3D dataset-level state averages; profile statistics flag only, validated against native selection on fixed priors. Probabilities are raw estimated retained-Key mass, not execution-weighted probability or true Dense mass. L0 seed accounting explicitly N/A.',rows=rows))
 names=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/'state_layers.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=names,lineterminator='\n');w.writeheader();w.writerows(rows)
 lines=['# 各层状态、History coverage与预算','',
 '范围：7Scenes heads128，两个带统计的完整前向取均值；每层是Query-group等权均值，不是ETH3D全数据集平均。状态统计开启rep_history_diagnostics；与原始基线固定先验Mask/权重等价已通过单测。生产精度与计时均关闭该统计开关。','',
 'H指原始合格历史集合；S指实际History support，A为H∪{rep}，其他为H。contains/only-rep比例以被选History簇为分母；coverage=这些簇保留Key的估计概率质量/这些簇总估计概率质量。不是执行权重乘后的质量，也不是真实Dense概率。L0仅做原生seed初始化，不使用四状态History损失，因此其状态会计项N/A，实际密度仍记录。','',
 '| Method | Layer | Drop % | Rep % | History % | Exact % | H mean Keys | S mean Keys | H contains Rep % | H only Rep % | S contains Rep % | H coverage % | S coverage % | Actual density % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 def fmt(v,scale=1):return 'N/A' if v is None else f'{v*scale:.4f}'
 for r in rows:
  keys=['drop_cluster_fraction','rep_cluster_fraction','history_cluster_fraction','exact_cluster_fraction','history_input_mean_keys','history_mean_keys','history_input_contains_rep_fraction','history_input_only_rep_fraction','history_support_contains_rep_fraction','history_input_probability_coverage','history_support_probability_coverage','actual_key_density']
  lines.append('| '+r['method']+' | '+str(r['layer'])+' | '+' | '.join(fmt(r.get(k),1 if 'mean_keys' in k else 100) for k in keys)+' |')
 lines+=['','## 各状态实际Key预算消耗（每Query group平均）','', '| Method | Layer | Drop | Rep | History | Exact | Packed count | Budget cap |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append('| '+r['method']+' | '+str(r['layer'])+' | '+' | '.join(fmt(r.get(k)) for k in ['drop_key_cost','rep_key_cost','history_key_cost','exact_key_cost','actual_key_count','patch_budget'])+' |')
 (OUT/'state_layers.md').write_text('\n'.join(lines)+'\n');return rows,mask

def main():
 from scripts.quality_metrics import quality_markdown
 full=read(OUT/'full/summary.json');calib=read(OUT/'calibration/summary.json');assert full['complete'] and full['scenes']==11
 manifest=read(OUT/'full/frozen_manifest.json');source=manifest['source_sha256'];speed_dir=OUT/'speed_heads128';assert read(speed_dir/'execution.json')['complete']
 monitor=read(OUT/'speed_gpu.json');assert monitor['gpu']==6 and monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
 runs={m:read(speed_dir/m/'timing_result.json') for m in ['baseline','A','B','dense','saf3r']};dense=runs['dense'];speed=[]
 assert runs['saf3r']['saf3r']['topk']==4 and runs['saf3r']['saf3r']['lazy_dino_topk']
 for method,d in runs.items():
  assert d['input']==dense['input'] and d['hardware']==dense['hardware']
  assert d['timing']['warmup']==2 and d['timing']['repeats']==5 and not d['timing']['capture']
  assert all(d['source_sha256'].get(k)==v for k,v in source.items() if k.startswith(('saf3r/','evaluation/')))
  seconds=d['timing']['median_seconds'];s=dense['timing']['median_seconds']/seconds
  d['quality']['speedup_vs_vggt']=s;d['quality']['speedup_unavailable']=None
  save(speed_dir/method/'timing_result.json',d);save(speed_dir/method/'quality.json',d['quality']);(speed_dir/method/'quality.md').write_text(quality_markdown(d['quality']))
  speed.append(dict(method=method,seconds=seconds,speedup_vs_dense=s,time_over_baseline=seconds/runs['baseline']['timing']['median_seconds'],time_over_saf3r=seconds/runs['saf3r']['timing']['median_seconds'],peak_allocated_gib=max(v['peak_allocated_gib'] for v in d['timing']['samples']),samples=d['timing']['samples']))
 states,mask=profiles();qkv=read(OUT/'common_qkv/result.json');assert len(qkv['rows'])==12
 for r in qkv['rows']:
  assert len(r['patch_keys'])==len(set(r['patch_keys']))
  if r['method']=='baseline':assert r['actual_kernel_vs_weighted_reference']['relative_l2']<.01
 error_summary={m:dict(mean_post_projection_relative_l2=mean(r['post_projection']['relative_l2'] for r in qkv['rows'] if r['method']==m),mean_post_projection_cosine=mean(r['post_projection']['cosine'] for r in qkv['rows'] if r['method']==m)) for m in ['baseline','A','B']}
 for name in ['validation/gpu.json','validation/native_gpu.json','profiles_gpu.json','profiles_production_gpu.json','common_qkv_gpu.json']:
  d=read(OUT/name);assert d['gpu'] in [4,5,6] and d['returncode']==0 and not d['same_gpu_overlap_detected']
 summary=dict(complete=True,quality=full,calibration=calib,speed=speed,mask_timing=mask,mask_timing_scope='Quiet production path; two CUDA-event profile forwards after2warm+5repeat normal timing on same GPU/model; pipeline includes clustering/shared pooling/coarse estimation/prior+gate+loss+solver+pack+weights; distinct from uninstrumented production latency',state_layers=states,common_qkv_error_summary=error_summary,common_qkv_scope=qkv['scope'],adopted=False)
 save(OUT/'summary.json',summary)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=['# 基于Rep Loss的History Loss：最终实验报告','',
 '只验证文档的两种公式，保留原始实现、独立配置与输出。A为完整union-support版本；B明确是|H|=1的受限消融，其余簇保持原逻辑。未引入新状态系统、额外Fine QK、Query-dependent全量Value距离、M5c或聚类改动、额外预算、Attention kernel改动。','',
 '## 实现与配置','',
 '- saf3r/patch/rep_history_loss.py：新模块，复用原共享Cluster×Grid Rep统计与概率残差；仅遍历历史边累计p_h*d_h。A新增O(Group×Cluster)的修正/union gate与计数权重，B只修正singleton。','- saf3r/patch/reuse_residual_exploration.py、reuse_window_graph.py：添加默认关闭的配置与dispatch。原始shared loss/kernel/packing、M5c、Value clustering及Attention kernel文件未修改，protected_source_files的SHA审计通过。','- full/configs/baseline.json、A.json、B.json：同一原始布局修复基线，仅A/B配置rep_history_loss；特殊Query沿用用户给出的camera-only基线，不混入all-special修正。','',
 'A：L_H=max(0,L_Rep−Σ_H p_h||V_h−V_r||²)。S=H∪{r}，c_H=|S|；非Rep历史Key权重1，Rep权重=|C|−(|S|−1)。H={Rep}时与Rep同Loss、Cost、Mask、权重；H=C时均为单位权重。该权重实现“其他成员由Rep表示”的计数代理，并不保证真实attention概率质量或softmax输出误差完全匹配。','',
 'B：仅|H|=1时L_H=max(0,L_Rep+(pi_C−2p_h)||V_h−V_r||²)，cost1，原H support与population补偿不变；多历史Key不套用单Key公式。','',
 '## 正确性与协议','',
 '7项初始CUDA测试通过；补充原始基线与诊断路径等价后5项测试通过。覆盖H={Rep}原始Loss恒等（不把inf门控当loss）、Cost/Mask/权重、A非负/单调/完整保留、B概率主导、无重复Key、packed count和cost一致、不超预算、可用输出有限、相同支持/权重的Attention输出一致，以及动态先验与历史的CUDA Graph重放。全部完整场景另通过预算/帧表/权重/配置/协议/源码/GPU隔离审计。','',
 'ETH3D完整11场景、377帧/方法；每场景上限128帧，uniform seed0；沿用既有terrace/meadow排除设置，不新增过滤。pose+recon_unposed均评测。阈值0.25m、downsample/voxel0.0390625m、TSDF trunc0.2m、mesh1M点；原unposed对齐/掩码/深度处理不变。AUC/F1为百分数，距离为米，CD=(ACC+Comp)/2非平方距离。均为场景等权宏平均。未使用7Scenes测试真值调参。','',
 '## 小规模验证（三场景，不是全数据集）','', '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for m,v in calib['macro'].items():lines.append('| '+m+' | '+cells(v)+' |')
 lines+=['','## 完整ETH3D比较（本批新测）','', '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for m,v in full['macro'].items():lines.append('| '+m+' | '+cells(v)+' |')
 lines+=['','用户提供的既有参考单独列示，不将新测分数平移/缩放到历史参考。','', '| Reference | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|','| historical layout-fixed | '+cells(full['provided_reference_macro'])+' |', '',
 '## 正常完整前向耗时','',
 '7Scenes heads128，非全7场景宏平均；GPU6 RTX4090，全部CPU质量评分结束后，同批每种方法2暖机+5重复，完整前向中位数，包含初始化/reset、Value clustering、routing/loss/packing、history、重排与特殊Query补算；不包含模型/图像磁盘加载。统计开关关闭，未使用下面的profiling耗时代替正常时间。','',
 '| Method | Median s | Dense speedup | Time / Baseline | Time / SAF3R | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
 for r in speed:lines.append(f"| {r['method']} | {r['seconds']:.6f} | {r['speedup_vs_dense']:.4f}x | {r['time_over_baseline']:.4f} | {r['time_over_saf3r']:.4f} | {r['peak_allocated_gib']:.4f} |")
 lines+=['','Dense与SAF3R只计时，六项几何质量为N/A；SAF3R使用原版Top-4 lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json，并非论文DINO Top-K或Top-128。未混用论文/H100结果。','',
 '## Mask构建耗时（独立生产路径CUDA事件）','',summary['mask_timing_scope'],'',
 '| Method | Mask pipeline ms | Prior/gate/loss/budget/pack/weights ms | Raw pipeline samples ms |','|---|---:|---:|---|']
 for r in mask:lines.append(f"| {r['method']} | {r['mask_pipeline_gpu_ms']:.4f} | {r['selection_prior_budget_gpu_ms']:.4f} | {r['mask_samples_ms']} |")
 lines+=['','以上Mask pipeline包含Value更新/聚类、共享统计、coarse估计和选择，不含稀疏Attention与历史写出；selection子项不含聚类。CUDA事件可能包含主机提交空隙，不等同纯kernel算力时间。两次额外profile均在正常计时之后、同模型/GPU/input上进行，未改变生产公式或增加状态统计；暖机后Graph创建/淘汰为0。','',
 '带概率coverage统计的profile、并行CPU评分时的早期profile分别保留在profiles/和profiles_production/，它们是诊断/初步观测，不替代quiet生产路径计时。','',
 '## 各层状态与History probability coverage','',
 '完整72行状态占比、原H与实际support均值、包含Rep/仅Rep比例、保留概率质量/coverage及各状态实际预算见state_layers.md、state_layers.csv和state_layers.json。明确区分H和A新增的Rep，不将union support算成原始历史证据。L0是原生seed初始化，会计项N/A；实际密度仍提供。状态诊断范围为heads128的两个profile，不伪装成ETH3D宏平均。','',
 '## 同Q/K/V对Dense输出误差','',qkv['scope'],'',
 '| Layer | Method | Post-Wo relative L2 % | Post-Wo cosine |','|---|---|---:|---:|']
 for r in qkv['rows']:lines.append(f"| {r['layer']} | {r['method']} | {r['post_projection']['relative_l2']*100:.6f} | {r['post_projection']['cosine']:.8f} |")
 lines+=['','抽样结果不显示A/B在所有层更接近Dense；A在部分层误差更大。该诊断不作为唯一选择目标，也不用于调整本轮参数。','',
 '## 解释与限制','',
 'A修正了Rep/History表示误差定义不一致及support缺Rep的问题，但残余成员仍以Rep的Key/Value和计数权重代替：其logit与真实成员不相同，不能保证保留真实概率质量，更不等于全局softmax Attention输出误差。共享head-mean/RMS Value代理和M5c估计误差仍存在。','',
 'B来自对其余成员Value接近Rep的近似；一般的单Key替代精确失真还包含其他Value与(V_h−V_r)的交叉项。公式没有建模这些项，且本轮多历史Key原逻辑完全未修复，因此收益或退化不应外推为完整History损失修复。','',
 '端到端AUC/F1/CD与固定QKV的少量局部Attention误差不是同一目标；后续层会改变QKV与状态轨迹。只以本批完整六指标、正常速度和逐场景退化判断，不把History使用率或代理损失降低当作成功证据。','',
 '完整逐场景六指标与冷前向记录见full/report.md和full/per_scene.csv；ETH3D冷耗时包含JIT，不能作为暖机速度比较。只报告7Scenes速度，不使用7Scenes真值调参。本报告保留所有候选与负面结果，未自动替换当前方案。']
 # Scene contributions are descriptive, not a post-hoc parameter selection.
 groups={}
 for r in full['per_scene']:groups.setdefault(r['scene'],{})[r['method']]=r
 lines+=['','## 逐场景变化与结论','', '| Scene | A ΔAUC@3 pp | A ΔF1 pp | A ΔCD m | B ΔAUC@3 pp | B ΔF1 pp | B ΔCD m |','|---|---:|---:|---:|---:|---:|---:|']
 for scene,g in groups.items():
  values=[(g[m][k]-g['baseline'][k])*(100 if k!='cd' else 1) for m in ['A','B'] for k in ['auc03','f1','cd']]
  lines.append('| '+scene+' | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
 lines+=['',
 'A的AUC@3在9/11场景提高，主要增益见courtyard、electro、facade、relief和delivery_area；F1整体略降主要受relief_2、delivery_area和electro拖累。relief_2同时出现F1降低而CD改善，说明阈值内命中率与均值距离不能互相替代。B的CD退化主要见delivery_area、electro和relief_2。','',
 'heads128诊断中，L1–L23等层平均History簇占比约为基线0.993%、A8.447%、B2.032%；A的Exact簇占比由约3.804%降至2.544%。这些只是该诊断输入上的预算重新分配证据，不是ETH3D宏平均或精度因果证明。A被选History的原H估计概率coverage约58.12%，低于基线约78.49%；插入Rep后的实际support另行记录，不能混作原历史证据。','',
 'A改善AUC和平均CD，但F1约下降0.52个百分点，正常完整前向约慢0.89%；只能称为有收益的部分修正，尚未同时满足恢复F1与完全不牺牲速度。B虽然AUC略升，但CD和F1均变差，不建议作为本轮改进方案。','',
 'A同时改变了公式所必需的support、cost和权重，本轮只能评价这组一致性修正，不能分离损失项与各项执行修正的独立贡献。所有参数均在小规模/完整评测前冻结，未根据这些结果重新调整。各方法选中簇集合不同，coverage差异不能直接解释为概率估计质量变坏。','',
 '本批中位数相对SAF3R：A慢约8.87%，B慢约7.77%，均在10%范围内；A相对原基线慢约0.89%，不满足严格的完全无速度代价要求。','',
 'Mask表仅累计已标注的阶段；未独立包裹的reset/layout初始化、QKV投影等不能从该表推出，均包含在正常完整前向耗时中。两个测试批次包含重叠案例，不将其宣称为12个独立测试。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');print('FINAL_REPORT_COMPLETE',json.dumps(dict(macro=full['macro'],speed=speed,mask_timing=mask)),flush=True)
if __name__=='__main__':main()
