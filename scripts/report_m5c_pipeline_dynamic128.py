"""Report only measured implementation timing; never synthesize quality scores."""
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/m5c_pipeline_dynamic_heads128_20261102'

sys.path.insert(0,str(ROOT))
from scripts.report_zdrop_cleanup128 import partition

def read(path):return json.loads(path.read_text())
def main():
 before=read(OUT/'before/forward.json');after=read(OUT/'after/forward.json');profile=read(OUT/'after/profile.json')
 manifest=read(OUT/'manifest.json');micro=read(ROOT/'outputs/m5c_pipeline_dynamic_20261102/drop_microbenchmark.json');extended=read(ROOT/'outputs/m5c_pipeline_dynamic_20261102/extended_checks.json');real=read(ROOT/'outputs/m5c_pipeline_dynamic_real128_20261102/result.json')
 frozen={variant:all(hashlib.sha256((Path(info['root'])/name).read_bytes()).hexdigest()==digest for name,digest in info['sha256'].items()) for variant,info in manifest['source'].items()}
 assert all(frozen.values()) and before['config']==after['config']
 layers=read(OUT/'after/profile_layers.json')
 assert len(layers)==24 and all(r['actual_unique_density']<=.2+1e-6 for r in layers)
 original=before['timing']['median_seconds'];optimized=after['timing']['median_seconds']
 totals={}
 for event in profile['events']:totals[event['name']]=totals.get(event['name'],0)+event['seconds']
 numerical={}
 for key in after['before_after_output_differences'][0]:
  numerical[key]=dict(max_before_after_relative_l2=max(r[key]['relative_l2'] for r in after['before_after_output_differences']),before_self_jitter=max(r[key]['relative_l2'] for r in before['same_variant_repeat_jitter']),after_self_jitter=max(r[key]['relative_l2'] for r in after['same_variant_repeat_jitter']))
 result=dict(max_drop_relative_l2=max(r.get('drop',{}).get('relative_l2',0) for r in real['records']),reference_timing_reused_from=manifest.get('reused_reference_timing'),complete=True,dataset='7scenes',scene='heads',frames=128,hardware=after['hardware'],config_identical=True,source_frozen=frozen,before=before['timing'],after=after['timing'],speedup_vs_preoptimization=original/optimized,time_reduction_fraction=1-optimized/original,profile_wall_seconds=profile['wall_seconds'],profile_inclusive_seconds=totals,output_differences=numerical,real_same_input_max_prior_rel_l2=max(r.get('prior',{}).get('relative_l2',0) for r in real['records']),real_history_qualified_mismatches=sum(r.get('gate_mismatches',{}).get('qualified',0) for r in real['records']),real_pi_max_rel_l2=max(r.get('pi',{}).get('relative_l2',0) for r in real['records']),peak_calibration_gpu_bytes=max(r['history_calibration_gpu_bytes'] for r in layers),quality_evaluated=False,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Implementation/numerical/timing diagnostic only; no pose or recon_unposed GT evaluation',dense_speedup=None,dense_reason='Unchanged Dense/SAF3R not rerun',controls_executed=0)
 by_layer=defaultdict(lambda:defaultdict(float))
 for event in profile['events']:by_layer[event['layer']][event['name']]+=event['seconds']
 def adjusted(values):
  values['M5c校准统计（GPU常驻）']=values.pop('M5c校准统计及CPU拷贝');return values
 parts=adjusted(partition(totals,profile['cuda_interval_seconds']))
 assert min(parts.values())>=-1e-4 and abs(sum(parts.values())-profile['cuda_interval_seconds'])<1e-5
 rows=[dict(stage=k,seconds=v,percent=100*v/profile['cuda_interval_seconds']) for k,v in sorted(parts.items(),key=lambda kv:-kv[1])]
 result['stages_disjoint']=rows
 def csvsave(path,data):
  with path.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 csvsave(OUT/'stage_distribution.csv',rows)
 per_layer=[]
 for i in range(24):
  t=by_layer[i];total=t['global_block']+t['frame_block'];per_layer.append(dict(layer=i,frame_plus_global_seconds=total,**adjusted(partition(t,total))))
 csvsave(OUT/'per_layer.csv',per_layer)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Drop与统计流水线实现优化：heads128','',
 '范围：7Scenes heads 单序列、128帧，不是全七场景质量宏平均。相同checkpoint、输入列表、预处理、20%预算、32-query组、64组窗口、每层聚类、16轮预算和历史/Rep/Exact规则；无Dense或SAF3R重跑。',
 '', '## 完整前向计时','',
 '同批次同一GPU独占测量上一版和新优化版；参考计时复用自本批次已完成阶段；各1次预热+3次无插桩完整前向取中位数。初始化/聚类/历史/分配/Attention/全部输出头均在forward内。模型加载及计时外数值检查不计入。','',
 '|实现|中位数 s|峰值 allocated GiB|峰值 reserved GiB|相对上一版M5c融合|','|---|---:|---:|---:|---:|']
 for label,data,speed in [('上一版M5c融合+GPU缓存',before,1.),('新Drop与统计流水线',after,original/optimized)]:
  t=data['timing'];lines.append(f"|{label}|{t['median_seconds']:.5f}|{max(r['peak_allocated_gib'] for r in t['samples']):.5f}|{max(r['peak_reserved_gib'] for r in t['samples']):.5f}|{speed:.4f}×|")
 lines+=['',f"耗时减少 {100*(1-optimized/original):.2f}%。这个加速比的分母是上一版M5c融合，不是Dense。",'', '## 实现变化','',
 '- 按现有簇的成员数分桶，直接求sum(p_i*z_i)的L2范数；不再逐query group建立大型三维向量库。分桶只改变kernel调度，不改变簇归属、Rep或状态。',
 '- 窗口内QK计数保持在GPU，合法性检查及计数同步集中到层末；检查未删除，异常输入仍在返回该层结果前报错。',
 '- 桶长度COUNT为不参与JIT特化的运行时参数，避免簇成员数量轻微变化造成反复编译；M5c概率公式和GPU FP32校准缓存保持上一版。',
 '- 第二次Attention QK保留，网格证据/Exact资格继续使用其结果；没有调整预算、状态或聚类。',
 '',f"缓存GPU占用最大 {result['peak_calibration_gpu_bytes']/2**30:.5f} GiB；实际总峰值见上表。新版逐层记录标明GPU校准缓存；history_cpu_bytes=0。",'',
 '## 数值验证','',
 f"扩展合成测试{len(extended['checks'])}例，包含FP32/BF16、尾query、无历史组和全Key观测；校准、概率、历史筛选、4个预算下的状态选择通过。",
 f"真实128帧抽查每层首尾窗口，共{len(real['records'])}项prior/cache/Drop检查；prior最大相对L2={result['real_same_input_max_prior_rel_l2']:.9g}，pi最大相对L2={result['real_pi_max_rel_l2']:.9g}。",
 f"Drop最大相对L2={max(r.get('drop',{}).get('relative_l2',0) for r in real['records']):.9g}。严格概率阈值附近有{result['real_history_qualified_mismatches']}条qualified历史候选翻转；Rep移除身份未变。公式等价不等于逐位等价，不能声称完整状态轨迹或GT精度完全不变。完整输出自身也存在重复运行波动。",'',
 '|输出|原版自身重复相对L2上限|新版自身重复上限|原新版相对L2上限|','|---|---:|---:|---:|']
 for key,value in numerical.items():lines.append(f"|{key}|{value['before_self_jitter']:.8f}|{value['after_self_jitter']:.8f}|{value['max_before_after_relative_l2']:.8f}|")
 lines+=['','## 新版独立插桩','',f"完整wall={profile['wall_seconds']:.5f}s；以下是包含式区间，不能互相全部累加；含CPU发射/排队空隙，非纯GPU活跃时间。",'', '|区间|秒|','|---|---:|']
 for key in ['history_probability','m5c_evidence_statistics','drop_vector_sum','sparse_attention_kernel','attention_mass_second_qk','mass_transfer','value_update','budget_solve4']:
  lines.append(f"|{key}|{totals.get(key,0):.6f}|")
 lines+=['','## 六项质量及Dense速度','', '|实现|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|同批Dense速度|','|---|---|---|---|---|---|---|---|','|上一版M5c融合|N/A|N/A|N/A|N/A|N/A|N/A|N/A|','|新Drop与统计流水线|N/A|N/A|N/A|N/A|N/A|N/A|N/A|','',result['quality_reason']+'; '+result['dense_reason']+'.', '', '原始JSON、逐层统计、源码归档及数值测试保存在本目录、outputs/m5c_pipeline_dynamic_20261102和outputs/m5c_pipeline_dynamic_real128_20261102。旧constexpr COUNT候选因反复编译中止，原始诊断保留在outputs/m5c_pipeline_heads128_20261102；其完整参考计时复用于本次比较，避免重跑未改动参考。不使用中止候选的速度作为最终结果。未根据GT选择参数。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:result[k] for k in ['speedup_vs_preoptimization','time_reduction_fraction','real_history_qualified_mismatches']},indent=2))
if __name__=='__main__':main()
