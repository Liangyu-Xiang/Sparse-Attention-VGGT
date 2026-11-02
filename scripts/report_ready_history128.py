"""Describe ready-history resource/precision tradeoff without inventing quality."""
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_zdrop_cleanup128 import partition
OUT=ROOT/'outputs/ready_history_v5_heads128_20261102'

def read(p):return json.loads(p.read_text())
def main():
 m=read(OUT/'manifest.json');b=read(OUT/'before/forward.json');a=read(OUT/'after/forward.json');p=read(OUT/'after/profile.json');layers=read(OUT/'after/profile_layers.json');real=read(ROOT/'outputs/ready_history_v5_real128_20261102/result.json');unit=read(ROOT/'outputs/ready_history_20261102/validation.json')
 frozen={v:all(hashlib.sha256((Path(d['root'])/n).read_bytes()).hexdigest()==h for n,h in d['sha256'].items()) for v,d in m['source'].items()}
 assert all(frozen.values()) and b['config']==a['config'] and len(layers)==24
 assert all(r['actual_unique_density']<=.2+1e-6 for r in layers)
 assert all(r['extra_previous_layer_probe_qk_per_head']==0 and r['extra_known_probe_qk_per_head']==0 and r['history_gpu_reconstruction_bytes']==0 for r in layers)
 monitor=read(ROOT/'outputs/ready_history_v5_heads128_20261102_monitor.json');assert not monitor['same_gpu_overlap_detected'] and monitor['returncode']==0
 t=defaultdict(float);perlayer=defaultdict(lambda:defaultdict(float))
 for e in p['events']:t[e['name']]+=e['seconds'];perlayer[e['layer']][e['name']]+=e['seconds']
 parts=partition(t,p['cuda_interval_seconds']);parts['上层证据校准及分层缓存写入']=parts.pop('M5c校准统计及CPU拷贝');parts['稀疏Attention与探针logit导出']=parts.pop('稀疏Attention前向')
 assert abs(sum(parts.values())-p['cuda_interval_seconds'])<1e-5 and min(parts.values())>=-1e-4
 rows=[dict(stage=k,seconds=v,percent=100*v/p['cuda_interval_seconds']) for k,v in sorted(parts.items(),key=lambda kv:-kv[1])]
 with (OUT/'stage_distribution.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 layer_rows=[]
 for i in range(24):
  x=perlayer[i];total=x['frame_block']+x['global_block'];values=partition(x,total);layer_rows.append(dict(layer=i,frame_plus_global_seconds=total,**values))
 with (OUT/'per_layer.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(layer_rows[0]));w.writeheader();w.writerows(layer_rows)
 conditional_error=max(r.get('conditional',{}).get('relative_l2',0) for r in real['records']);prior_error=max(r.get('prior',{}).get('relative_l2',0) for r in real['records']);pi_error=max(r.get('pi',{}).get('relative_l2',0) for r in real['records'])
 result=dict(complete=True,quality_evaluated=False,scope='7Scenes heads128 single-sequence diagnostic',source_frozen=frozen,config_identical=True,current=a['timing'],historical_reference=b['timing'],historical_time_ratio=b['timing']['median_seconds']/a['timing']['median_seconds'],paired_speedup=None,paired_reason='Unchanged reference reused from historical run; allocator/cache behavior also differs',dense_speedup=None,dense_reason='Unchanged Dense/SAF3R not rerun',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Numerical/timing diagnostic only; pose/recon_unposed quality not evaluated',gpu_cache_bytes=max(r['history_calibration_gpu_bytes'] for r in layers),host_pinned_cache_bytes=max(r['history_cpu_bytes'] for r in layers),gpu_edge_cache_bytes=max(r['ready_edge_cache_bytes'] for r in layers),conditional_relative_l2=conditional_error,prior_relative_l2=prior_error,pi_relative_l2=pi_error,qualified_mismatches=real['qualified_mismatches'],half_underflow_count=sum(r.get('half_underflow_count',0) for r in real['records']),profile_wall_seconds=p['wall_seconds'],profile_inclusive_seconds=dict(t),profile_disjoint=rows,remaining_second_qk_seconds=t['attention_mass_second_qk'],extra_historical_probe_qk_per_head=0,extra_current_probe_qk_per_head=0,codec='FP16 rounded once, then bitwise-lossless positive-u14 packing',cache_gpu_limit_gib=8,allocator='expandable_segments:True, pressure-only trim',allocator_trim_calls=max(r['ready_allocator_trims'] for r in layers))
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 m.update(cache_encoding='positive FP16 losslessly packed into u14',gpu_edge_cache_cap_gib=8,host_spill='pinned, asynchronous two-buffer H2D prefetch/D2H writeback',precision_change='FP32 to FP16 edge conditionals, no extra packing quantization',comparison='Historical reference only; not contemporaneous paired speedup')
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 text=['# 上一层准备历史条件概率：heads128','',
 '本轮是7Scenes heads单序列、128帧实现/数值/速度诊断，不是七场景质量宏平均。使用原checkpoint、固定输入列表、预处理、20%预算、32-query分组、64组窗口、每层聚类和16轮预算。保留所有head/4探针，canonical grid继续使用整个grid的池化Key；未改为剩余成员均值。', '',
 '首次稀疏Attention直接导出所需探针原始logit，本层结束前校准并缓存条件概率；下一层仅使用当前grid观测和缓存，重建旧QK及校准专用探针QK均为零。原Attention第二次QK仍保留，用于原grid/Exact观测，不声称所有重复QK已消除。','',
 '缓存先做FP16量化，再对0~1正FP16数去掉恒零的两位，14位打包本身逐位无损。单层原地覆盖、GPU缓存上限8GiB、尾窗口pinned CPU缓存、双缓冲异步预取/回写；窗口流事件保护旧数据读取及原地覆盖，不同时保留两整层。最后一层不再准备无人使用的历史缓存。','',
 '|范围|完整前向中位数 s|峰值 allocated GiB|峰值 reserved GiB|','|---|---:|---:|---:|']
 for label,data in [('旧版历史记录',b),('新版本次实测',a)]:
  q=data['timing'];text.append(f"|{label}|{q['median_seconds']:.5f}|{max(r['peak_allocated_gib'] for r in q['samples']):.5f}|{max(r['peak_reserved_gib'] for r in q['samples']):.5f}|")
 text+=['',f"当前1次预热+3次完整前向，包含初始化、分配、聚类、DMA等待、校准、重排及输出头。冷前向{a['timing']['cold_samples'][0]['seconds']:.5f}s。旧时间仅作历史参考，不能当同批配对验收；Dense速度N/A（未重跑）。GPU独占监控无同卡外部重叠。",'',
 f"GPU算法缓存 {result['gpu_cache_bytes']/2**30:.5f} GiB，其中GPU边缓存 {result['gpu_edge_cache_bytes']/2**30:.5f} GiB，另含grid统计与双缓冲。CPU固定内存缓存 {result['host_pinned_cache_bytes']/2**30:.5f} GiB；这不是总CPU峰值。相对于旧重算方案，整体显存上升；相对于原样FP32逐边缓存约26GiB，压缩后的逻辑边缓存约11.3GiB，分布在GPU与CPU。",'',
 '## 验证','',f"原子/归约FP32路径、FP16缓存路径、无历史、全观测、尾query、全部正FP16编码打包解包及四种logsumexp长度测试通过；三层强制CPU溢出测试通过，prior相对L2={unit['spill_validation']['prior_relative_l2']:.8g}。",
 f"真实输入每层首尾窗口各抽一个query group，共{len(real['records'])}项producer/consumer检查。导出logit相同；条件概率相对L2≤{conditional_error:.8g}，prior≤{prior_error:.8g}，簇质量pi≤{pi_error:.8g}。",
 f"FP16相对FP32会量化，抽查有{real['qualified_mismatches']}条历史候选阈值翻转，另有{result['half_underflow_count']}个极小逐头/探针系数下溢为零。不能声称逐位等价或最终质量不降。缓存生命周期/顺序校验、budget上限及源哈希校验通过。",'',
 '## 已有新版profile','',f"独立插桩wall={p['wall_seconds']:.5f}s。下列是包含式主流区间；DMA在复制流上重叠执行，不能额外把DMA持续时间加到这些区间上。主流等待和最终全设备同步计入完整前向。CUDA区间含CPU发射/排队空隙。",'', '|区间|秒|','|---|---:|']
 for k in ['sparse_attention_kernel','m5c_evidence_statistics','history_probability','attention_mass_second_qk','ready_allocator_trim']:text.append(f"|{k}|{t[k]:.6f}|")
 text+=['','|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|','|---|---|---|---|---|---|','|N/A|N/A|N/A|N/A|N/A|N/A|','',result['quality_reason']+'. No GT parameter tuning.', '',
 '纯GPU缓存原型的OOM原始记录保存在outputs/ready_history_heads128_20261102以及ready_history_v2/v3/v4_heads128_20261102；它们不是完成的质量评测，也没有被用作最终速度。最终模型源码归档与manifest位于本目录。复现脚本为scripts/setup_ready_history.py，计时脚本为scripts/run_ready_history_v5_128.py。']
 (OUT/'report.md').write_text('\n'.join(text)+'\n')
 print(json.dumps({k:result[k] for k in ['gpu_cache_bytes','host_pinned_cache_bytes','conditional_relative_l2','prior_relative_l2','qualified_mismatches']},indent=2))
if __name__=='__main__':main()
