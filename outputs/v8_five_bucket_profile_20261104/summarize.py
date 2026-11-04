"""Lossless accounting of the five algorithm stages and base-model remainder."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
raw=json.loads((ROOT/'result/profile.json').read_text())
mapping={
 'cluster':['value_update','cluster_centers_from_labels','cluster_full_assignment','cluster_repair_empty','cluster_center_graph','cluster_local_assignment','cluster_population'],
 'history':['history_key_pooling','m5c_coarse_observation','attention_grid_normalize_reduce','m5c_evidence_and_host_copy','local_normalizers','kernel_local_normalizers','local_eps_compatibility','kernel_local_compat','local_probability_headmean_cache','kernel_local_uniform','kernel_local_store_local','local_checks_and_coarse_store','history_summary','coarse_observation','history_prior_estimation'],
 'state':['pooling','selection_cluster_grid_overlap','residual_fine_probe'],
 'mask':['global_attention','attention_wrapper'],
 'attention':['qkv_rope_prepare','execution_and_evidence','attention_core_fused_grid','output_projection','output_dropout']}
labels={'cluster':'每层 Value 特征、Cluster 划分与更新','history':'历史概率估计、生成、归一化、缓存与消费',
 'state':'资格筛选、状态损失计算与预算选择','mask':'稀疏 Mask 构建、质量转移与执行准备','attention':'稀疏 Attention 执行（含 QKV 与输出投影）'}
groups={k:v['exclusive_seconds'] for k,v in raw['groups'].items()}
names=[k for v in mapping.values() for k in v]
assert len(names)==len(set(names))
assert set(groups)-set(names)=={'reuse_and_budget'}

def account(group,windows):
    totals={key:sum(group.get(k,0.) for k in value) for key,value in mapping.items()}
    graph={k:sum(w[k+'_seconds'] for w in windows) for k in ['gate','loss','selection','pack','transfer','diagnostics','total']}
    overhead=group.get('reuse_and_budget',0.)-graph['total']
    assert overhead>=-1e-5,overhead
    totals['state']+=graph['gate']+graph['loss']+graph['selection']
    totals['mask']+=graph['pack']+graph['transfer']+graph['diagnostics']+overhead
    assert abs(sum(totals.values())-sum(group.values()))<1e-7
    return totals,graph,overhead

totals,graph,overhead=account(groups,raw['graph_windows'])
assert abs(sum(totals.values())-raw['global_attention_seconds'])<1e-7
other=raw['wall_seconds']-sum(totals.values())
assert other>=0
rows=[dict(category=key,label=labels[key],seconds=value,
           percent_sparse_chain=100*value/raw['global_attention_seconds'],
           percent_full_forward=100*value/raw['wall_seconds'],mean_per_global_layer_ms=1000*value/24)
      for key,value in totals.items()]
layers=[]
by_id={e['id']:e for e in raw['events']}
for layer in range(24):
    group={}
    for event in raw['events']:
        ancestor=event
        while ancestor['parent'] is not None:
            ancestor=by_id[ancestor['parent']]
        if ancestor['layer']==layer:
            group[event['name']]=group.get(event['name'],0.)+event['exclusive_seconds']
    windows=[w for w in raw['graph_windows'] if w['layer']==layer]
    buckets,_,_=account(group,windows)
    layers.append(dict(layer=layer,**{k+'_ms':1000*v for k,v in buckets.items()},
                       total_ms=1000*sum(buckets.values()),selection_marker_runs=len(windows)))
for key in totals:
    assert abs(sum(x[key+'_ms'] for x in layers)/1000-totals[key])<1e-7
monitor=json.loads((ROOT/'monitor.json').read_text())
assert not monitor['auxiliary_cuda_context'] and not monitor['same_gpu_overlap_detected']
assert not monitor['foreign_gpu_pids'] and monitor['returncode']==0 and monitor['max_simultaneous_gpu_applications']==1
hashes=json.loads((ROOT/'original_source_sha256.json').read_text())
unchanged=all(hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
assert unchanged
summary=dict(method='v8',input_scope='7Scenes heads128,24 global layers,RTX5090 D GPU7',
 timing_scope=raw['timing_scope'],full_forward_seconds=raw['wall_seconds'],
 sparse_chain_seconds=sum(totals.values()),base_model_and_outside_global_seconds=other,
 base_model_percent_full_forward=100*other/raw['wall_seconds'],categories=rows,
 graph_subphases=graph,graph_framework_seconds=overhead,
 graph_selection_marker_runs=raw['graph_marker_count'],logical_selection_windows=23*66,
 extra_eager_selection_runs=raw['graph_marker_count']-23*66,
 operation_mapping=mapping,per_layer=layers,source_unchanged=unchanged,
 monitor_passed=True,full_accounting_error_seconds=sum(totals.values())+other-raw['wall_seconds'],
 quality=raw['quality'],quality_reason=raw['quality_reason'],dense_speedup=None,dense_reason=raw['dense_reason'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
for name,data in [('buckets.csv',rows),('per_layer.csv',layers)]:
    with (ROOT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
text=['# v8 五类耗时统计','',
 '输入：7Scenes heads128，24 个全局 Attention 层，RTX5090 D GPU7。CPU 文件锁及监控不创建辅助 CUDA context；最大同时 CUDA 应用数1，无外部 PID，无重叠。来源为 efficiency_quality_v8_20261103/source 的独立副本，原模型源码未改；仅当前诊断进程内插入时间戳和事件范围。','',
 f"本次一次预热后一次带插桩完整前向 **{raw['wall_seconds']:.9f}s**；全局稀疏链路累计 **{sum(totals.values()):.9f}s**。历史无插桩 warmed median19.981s来自20261103另一计时批次，不能用它作为本次分项百分比的分母。加载 checkpoint/图像、H2D 输入搬运及 GT 评估不在前向计时内。",'',
 '| 部分 | 耗时 s | 占稀疏链路 % | 占完整前向 % | 均摊每个全局层 ms |','|---|---:|---:|---:|---:|']
for row in rows:
    text.append(f"| {row['label']} | {row['seconds']:.6f} | {row['percent_sparse_chain']:.2f} | {row['percent_full_forward']:.2f} | {row['mean_per_global_layer_ms']:.3f} |")
text += [f"| 五类稀疏链路合计 | {sum(totals.values()):.6f} | 100.00 | {100*sum(totals.values())/raw['wall_seconds']:.2f} | — |",
 f"| 基础模型及全局模块外开销 | {other:.6f} | — | {100*other/raw['wall_seconds']:.2f} | — |",
 f"| 完整前向 | {raw['wall_seconds']:.6f} | — | 100.00 | — |",'',
 '归类边界：','',
 '- Cluster 包含本层 Value 特征、中心/Rep 更新、局部及完整成员分配、空簇修复、population 计数；24层都计入。',
 '- 历史概率包含当前 Coarse 观测、Key Grid 聚合、历史 prior 消费，以及上一层证据的归一化、head-mean、压缩缓存和 Grid 统计；生产和消费各计一次。',
 '- 状态部分包含规范化状态几何、Cluster/Grid 关系、Exact 候选筛选、历史 gate、状态损失及16轮预算搜索。',
 '- Mask 部分包含最终 Key 列表打包、概率质量转移/补偿、诊断与合法性检查、排序与索引、Graph 输入刷新/常量更新/输出收集及未单独命名的全局路由框架开销。',
 '- Attention 包含该模块 QKV/Norm/RoPE、稀疏 QK/softmax/AV、执行发射和输出投影。与正式 QK/AV 融合产生的历史证据属于共享内核，统一计入 Attention，不再重复分摊到历史。',
 '- 基础模型部分包括局部 Attention、FFN、输出头、前向初始化和全局模块外未插桩开销；它不属于这五类稀疏算法步骤，因此单列，避免把它伪装为某一步。',
 '- CUDA Graph GPU 时间戳在 capture 前安装，直接测量 gate/loss/selection/pack/transfer/diagnostics；外层使用 CUDA Event 并扣除所有子项。Graph 包含1518个逻辑选择窗口，另有4次本次图缓存 miss 初始化时的 eager 选择，全部计入。时间戳已与 CUDA Event 做量级校验，见 result/clock_validation.json。',
 '- Graph 外层减去图内全部阶段的差额计入 Mask 框架开销；这是一条明确的记账约定，不宣称这部分全部是打包 kernel。百分比四舍五入可有0.01个百分点误差。','',
 '| Graph 内部子步骤 | 累计 s | 归入五类 |','|---|---:|---|']
for key in ['gate','loss','selection','pack','transfer','diagnostics']:
    text.append(f"| {key} | {graph[key]:.6f} | {'状态损失与选择' if key in ['gate','loss','selection'] else 'Mask 构建'} |")
text += [f'| Graph 数据刷新/框架差额 | {overhead:.6f} | Mask 构建 |','',
 '逐层真实时间（不是平均数复制；L0没有历史选择）：','',
 '| 层 | Cluster ms | 历史 ms | 损失/选择 ms | Mask ms | Attention ms | 合计 ms |','|---|---:|---:|---:|---:|---:|---:|']
for layer in layers:
    text.append('| '+str(layer['layer'])+' | '+' | '.join(f'{layer[k+"_ms"]:.3f}' for k in ['cluster','history','state','mask','attention'])+f" | {layer['total_ms']:.3f} |")
text += ['',
 '本次只做耗时诊断，未执行 pose/recon_unposed；不能当作新的完整质量评估。原 v8 完整质量见 efficiency_final_v8_20261103/report.md。','',
 '| AUC@3 | AUC@30 | ACC | Comp | CD | F1 | Dense配对加速 |','|---|---|---|---|---|---|---|',
 '| N/A | N/A | N/A | N/A | N/A | N/A | N/A |',
 '原因：本次仅插桩计时，没有重跑 GT 质量；按要求没有重跑未改动的 Dense，因此无同批配对分母。','',
 f"完整账目误差：{summary['full_accounting_error_seconds']:.12f}s；原始模型源码哈希一致：{unchanged}。原始事件和窗口时间戳：result/profile.json；逐层表：per_layer.csv；分类表：buckets.csv；完整映射及验证：summary.json。"]
(ROOT/'report.md').write_text('\n'.join(text)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ['full_forward_seconds','sparse_chain_seconds','base_model_and_outside_global_seconds','base_model_percent_full_forward','categories','graph_framework_seconds','source_unchanged','full_accounting_error_seconds']},indent=2,ensure_ascii=False))
