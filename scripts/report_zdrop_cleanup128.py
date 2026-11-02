"""Disjoint latency distribution; keep inclusive function detail separate."""
import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/zdrop_cleanup_heads128_20261102'

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvsave(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def partition(t,total):
 get=lambda name:t.get(name,0.)
 rows={
 'DINO图像特征':get('patch_embed_dino'),
 '逐帧Transformer块':get('frame_block'),
 '全局QKV与输出投影':get('global_qkv')+get('global_proj'),
 '全局MLP':get('global_mlp'),
 '全局块归一化和残差':get('global_block')-get('global_attention')-get('global_mlp'),
 '当前Value聚类更新':get('value_update'),
 '几何缓存':get('geometry_prepare'),
 '池化与簇网格关系':get('pooling')-get('geometry_prepare'),
 'M5c当前粗观测':get('m5c_coarse_observation'),
 '原coarse观测与Exact候选':get('coarse_observation')+get('residual_fine_probe'),
 '历史概率估计':get('history_probability'),
 '历史筛选及Rep身份':get('history_gate'),
 'RMS Drop向量汇总':get('drop_vector_sum'),
 'Rep平方L2损失':get('rep_loss'),
 'History损失与最近Key':get('history_loss_and_owners'),
 '状态损失组装':get('state_losses')-get('rep_loss')-get('history_loss_and_owners'),
 '四状态16轮预算求解':get('budget_solve4'),
 '状态ID映射':get('state_decode'),
 '支持集打包':get('support_pack'),
 '概率质量转移':get('mass_transfer'),
 '状态统计':get('state_diagnostics'),
 '分配其他准备':get('reuse_and_budget')-sum(get(n) for n in ['history_probability','history_gate','drop_vector_sum','state_losses','budget_solve4','state_decode','support_pack','mass_transfer','state_diagnostics']),
 '稀疏Attention前向':get('sparse_attention_kernel'),
 'Attention概率第二次QK':get('attention_mass_second_qk'),
 'Attention执行准备':get('execution_and_evidence')-get('sparse_attention_kernel')-get('attention_mass_second_qk'),
 '当前M5c探针QK':get('current_history_probe_qk'),
 'M5c校准统计及CPU拷贝':get('m5c_evidence_and_host_copy')-get('current_history_probe_qk'),
 '历史网格汇总':get('history_summary'),
 '相机头':get('camera_head'),
 '深度头':get('depth_head'),
 '其他输出头':get('point_head')+get('track_head'),
 }
 stages=['value_update','pooling','m5c_coarse_observation','coarse_observation','residual_fine_probe','reuse_and_budget','execution_and_evidence','m5c_evidence_and_host_copy','history_summary']
 rows['全局Attention其余工作']=get('global_attention')-sum(get(n) for n in stages)-get('global_qkv')-get('global_proj')
 rows['其余模型工作']=total-get('patch_embed_dino')-get('frame_block')-get('global_block')-sum(get(n) for n in ['camera_head','depth_head','point_head','track_head'])
 return rows

def main():
 before=json.loads((OUT/'before/forward.json').read_text());after=json.loads((OUT/'after/forward.json').read_text());prof=json.loads((OUT/'after/profile.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
 frozen={v:all(hashlib.sha256((Path(d['root'])/n).read_bytes()).hexdigest()==h for n,h in d['sha256'].items()) for v,d in m['source'].items()}
 t=defaultdict(float);host=defaultdict(float);count=defaultdict(int);perlayer=defaultdict(lambda:defaultdict(float))
 for ev in prof['events']:
  t[ev['name']]+=ev['seconds'];host[ev['name']]+=ev['cpu_seconds'];count[ev['name']]+=1;perlayer[ev['layer']][ev['name']]+=ev['seconds']
 parts=partition(t,prof['cuda_interval_seconds']);parts_rows=[dict(stage=k,seconds=v,percent=100*v/prof['cuda_interval_seconds']) for k,v in sorted(parts.items(),key=lambda x:-x[1])]
 details=[dict(name=k,inclusive_cuda_seconds=v,inclusive_host_seconds=host[k],calls=count[k]) for k,v in sorted(t.items(),key=lambda x:-x[1])]
 layer_rows=[]
 for layer in range(24):
  p=partition(perlayer[layer],perlayer[layer].get('frame_block',0.)+perlayer[layer].get('global_block',0.))
  layer_rows.append(dict(layer=layer,frame_plus_global_seconds=perlayer[layer].get('frame_block',0.)+perlayer[layer].get('global_block',0.),**p))
 jitter={}
 for variant,data in [('before',before),('after',after)]:
  keys=set(k for entry in data['same_variant_repeat_jitter'] for k in entry)
  jitter[variant]={k:max(entry[k]['relative_l2'] for entry in data['same_variant_repeat_jitter']) for k in keys}
 comparisons={k:max(entry[k]['relative_l2'] for entry in after['before_after_output_differences']) for k in after['before_after_output_differences'][0]}
 profile_diffs={k:d['relative_l2'] for k,d in prof['profile_vs_uninstrumented_difference'].items()}
 numeric={k:dict(before_self_jitter=jitter['before'][k],after_self_jitter=jitter['after'][k],max_before_after_relative_l2=v,within_0_005=v<=.005,within_observed_jitter=v<=max(jitter['before'][k],jitter['after'][k])+1e-7,profile_relative_l2=profile_diffs[k]) for k,v in comparisons.items()}
 newlayers=json.loads((OUT/'after/profile_layers.json').read_text());checks=dict(sources_unchanged=all(frozen.values()),same_config=before['config']==after['config'],all24=len(newlayers)==24,edge_cap20=all(l['actual_unique_density']<=.2+1e-7 for l in newlayers),costs=all(l.get('native_cost_count_error_mean',0)==0 for l in newlayers),transfer=all(l['history_mass_transfer_error_mean']<=1e-5 for l in newlayers[1:]),partition_sum=abs(sum(parts.values())-prof['cuda_interval_seconds'])<1e-5,nonnegative_parts=min(parts.values())>=-1e-4)
 result=dict(complete=True,scene='heads',dataset='7scenes',frames=128,uninstrumented_before=before['timing'],uninstrumented_after=after['timing'],cleanup_speedup=before['timing']['median_seconds']/after['timing']['median_seconds'],profile_wall_seconds=prof['wall_seconds'],profile_cuda_interval_seconds=prof['cuda_interval_seconds'],instrumentation_wall_ratio=prof['wall_seconds']/after['timing']['median_seconds'],stages=parts_rows,inclusive_detail=details,numerical_comparison=numeric,checks=checks,source_frozen=frozen,quality_evaluated=False,quality_six_fields={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_unavailable_reason='Implementation/timing diagnostic only; no GT quality evaluation',dense_speedup=None,dense_speedup_unavailable_reason='Dense/SAF3R not rerun',notes=prof['notes'])
 save(OUT/'summary.json',result);save(OUT/'numerical_comparison.json',numeric);csvsave(OUT/'stage_distribution.csv',parts_rows);csvsave(OUT/'inclusive_detail.csv',details);csvsave(OUT/'per_layer.csv',layer_rows)
 text=['# heads128：当前方法清理无用计算后的耗时分布','',
 '仅分析当前“修正RMS Drop基线+历史状态”。相同checkpoint、固定heads128输入、配置、20%预算、原簇数、32query组、RMS Drop及平方L2/历史规则。没有运行Dense、SAF3R或GT质量评测。不是七场景宏平均。',
 f'无插桩完整前向：清理前中位数{before["timing"]["median_seconds"]:.5f}s，清理后{after["timing"]["median_seconds"]:.5f}s，同GPU同批当前方法清理加速{result["cleanup_speedup"]:.4f}×；各一次预热+三次测量。',
 f'清理后另一次插桩诊断wall={prof["wall_seconds"]:.5f}s，CUDA区间={prof["cuda_interval_seconds"]:.5f}s。下表来自该诊断，不能将百分比乘无插桩时间当精确阶段秒数。CUDA事件区间包含CPU发射/排队空隙，非纯GPU内核活跃时间；无逐层新增同步。',
 '移除旧Top4参考筛选、废弃前缀/混合损失、无用的平方范数Drop中间量；求解4个有效状态并保留原同分顺序；移除未读取的Value均值、被M5c覆盖的coarse pi、未读取的half历史权重/对应gather、L0无用几何/RMS/overlap及空gain占位统计。保留原Drop向量汇总、Exact候选、完整History表示损失、16轮预算、质量转移、第二QK以及原历史/聚类更新。','',
 '| 阶段（互不重叠） | 秒 | 占CUDA完整区间 % |','|---|---:|---:|']
 for row in parts_rows:text.append(f'|{row["stage"]}|{row["seconds"]:.6f}|{row["percent"]:.3f}|')
 text+=['','## 数值一致性检查','', '|输出|原实现自身重复相对L2上限|清理后自身重复上限|清理前后上限|插桩相对无插桩|','|---|---:|---:|---:|---:|']
 for k,v in numeric.items():text.append(f'|{k}|{v["before_self_jitter"]:.8g}|{v["after_self_jitter"]:.8g}|{v["max_before_after_relative_l2"]:.8g}|{v["profile_relative_l2"]:.8g}|')
 text+=['','合成同输入验证：保留损失/目标/成本在FP32容差1e-5内一致；8种预算下选中状态、实际Key列表、权重完全一致，含同分、无历史、单元素簇和Exact不可用。全模型存在GPU原子归约非确定性；数值校验不等于六项GT精度验收，不声称未测质量分数。',f'结构与归档检查：{json.dumps(checks,ensure_ascii=False)}','',f'峰值allocated（无插桩各热前向最大）：前{max(s["peak_allocated_gib"] for s in before["timing"]["samples"]):.5f}GiB，后{max(s["peak_allocated_gib"] for s in after["timing"]["samples"]):.5f}GiB。','',
 '|AUC@3|AUC@30|ACC m|Comp m|CD m|F1|同批Dense加速|','|---|---|---|---|---|---|---|','|N/A|N/A|N/A|N/A|N/A|N/A|N/A|','',
 '原因：本次只做实现清理与耗时诊断，不计算pose/recon_unposed质量，也不重跑未改动Dense/SAF3R。','',
 'stage_distribution.csv为互斥耗时分布；inclusive_detail.csv为嵌套函数/模块包含式时间，禁止与父项重复相加；per_layer.csv为24层分解；after/instrumented_forward.py保存仅增加计时scope的forward；before/after_source.tar及manifest.json保存清理前后隔离源码与哈希。']
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
