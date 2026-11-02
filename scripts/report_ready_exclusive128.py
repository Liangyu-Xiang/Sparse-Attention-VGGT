"""Audit exclusive main-stream intervals; DMA activity is non-additive."""
import collections,csv,gzip,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/ready_history_exclusive128_20261102'

def union(intervals):
 result=[]
 for a,b in sorted(intervals):
  if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(result[-1][1],b))
  else:result.append((a,b))
 return result

def length(intervals):return sum(b-a for a,b in intervals)
def overlap(a,b):
 a=union(a);b=union(b);i=j=0;total=0.
 while i<len(a) and j<len(b):
  total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<=b[j][1]:i+=1
  else:j+=1
 return total

def main():
 data=json.loads((OUT/'timeline.json').read_text());events={e['id']:e for e in data['events']};assert len(events)==len(data['events'])
 children=collections.defaultdict(list)
 for e in events.values():
  if e['lane']!='main':continue
  parent=e['parent_id']
  while parent is not None and events[parent]['lane']!='main':parent=events[parent]['parent_id']
  if parent is not None:children[parent].append(e)
 def lineage(e):
  names=[]
  while e is not None:
   names.append(e['name']);e=events.get(e['parent_id'])
  return names
 def included(e):
  names=lineage(e)
  return e['lane']=='main' and 'global_attention' in names and not any(n in names for n in ['global_qkv','global_proj'])
 def category(e):
  n=e['name'];names=set(lineage(e))
  direct={
   'sparse_attention_kernel':'稀疏Attention与探针logit导出（融合内核）',
   'attention_mass_second_qk':'第二次选中边QK与概率聚合（融合内核）',
   'kernel_grid_exp_segment_reduce':'校准：逐边指数与grid分段累加（融合内核）',
   'kernel_conditional_quantize_pack':'校准：条件概率计算、FP16量化与u14打包（融合内核）',
   'kernel_selected_logsumexp':'校准：保留边logsumexp归约',
   'coarse_grid_qk':'当前canonical grid QK',
   'coarse_special_qk':'当前特殊Token QK',
   'cache_read_or_h2d_wait':'缓存读取、缓冲区就绪等待与控制',
   'cache_staging_write_wait':'缓存读取、缓冲区就绪等待与控制',
   'cache_prefetch_host_submit':'异步缓存预取/回写提交与控制',
   'cache_writeback_host_submit':'异步缓存预取/回写提交与控制',
   'prior_prefetch_enqueue':'异步缓存预取/回写提交与控制',
   'calib_writeback_enqueue':'异步缓存预取/回写提交与控制',
   'state_diagnostics':'诊断统计与概率质量检查',
   'prior_mass_validation':'诊断统计与概率质量检查',
   'calib_mass_validation':'诊断统计与概率质量检查',
   'global_attention':'未细分全局Attention准备与收尾（含Q/K归一化、RoPE等）',
   'reuse_and_budget':'分配阶段其余准备',
   'execution_and_evidence':'Attention工作区初始化与执行准备',
   'history_summary':'下一层grid/Exact证据汇总',
   'ready_allocator_trim':'缓存分配与显存回收',
   'ready_cache_allocate':'缓存分配与显存回收',
  }
  if n in direct:return direct[n]
  for ancestor,label in [
   ('mass_transfer','概率质量转移与补偿权重'),
   ('support_pack','支持集打包及合法性检查'),
   ('history_gate','历史候选筛选与Rep身份'),
   ('drop_vector_sum','RMS Drop加权向量范数'),
   ('history_loss_and_owners','History状态最近Key归属与表示损失'),
   ('state_losses','Rep损失与其他状态损失组装'),
   ('budget_solve4','16轮预算求解与状态ID映射'),
   ('state_decode','16轮预算求解与状态ID映射'),
   ('value_update','当前Value聚类更新'),
   ('pooling','池化、簇几何及grid映射准备'),
   ('prior_history_edge_update','缓存历史边概率更新、head平均与回填'),
   ('history_probability','网格概率迁移、prior展开与簇质量汇总'),
   ('m5c_coarse_observation','当前粗观测：探针准备、归一化及调度'),
   ('coarse_observation','原coarse观测与Exact候选筛选'),
   ('residual_fine_probe','原coarse观测与Exact候选筛选'),
   ('m5c_evidence_and_host_copy','校准其余：计数、网格归一化及缓存准备'),
  ]:
   if ancestor in names:return label
  raise AssertionError(('Unclassified',n,lineage(e)))
 roots=[e for e in events.values() if e['lane']=='main' and e['name']=='global_attention'];assert len(roots)==24
 root_spans=[(e['start_seconds'],e['end_seconds']) for e in roots];assert abs(length(root_spans)-length(union(root_spans)))<1e-7
 projections=[(e['start_seconds'],e['end_seconds']) for e in events.values() if e['lane']=='main' and e['name'] in ['global_qkv','global_proj'] and 'global_attention' in lineage(e)]
 expected=length(root_spans)-length(union(projections));segments=[];scope_rows=[];sums=collections.defaultdict(float);counts=collections.Counter()
 for e in events.values():
  if not included(e):continue
  cursor=e['start_seconds'];end=e['end_seconds'];pieces=[]
  for child in sorted(children[e['id']],key=lambda x:x['start_seconds']):
   a,b=child['start_seconds'],child['end_seconds']
   assert a>=cursor-1e-7 and b<=end+1e-7,(e['name'],child['name'],cursor,a,b,end)
   if a>cursor:pieces.append((cursor,a))
   cursor=b
  if end>cursor:pieces.append((cursor,end))
  label=category(e);self_time=length(pieces);sums[label]+=self_time;counts[label]+=1
  scope_rows.append(dict(scope_id=e['id'],parent_id=e['parent_id'],name=e['name'],category=label,layer=e['layer'],inclusive_seconds=end-e['start_seconds'],exclusive_seconds=self_time))
  segments.extend(dict(start_seconds=a,end_seconds=b,seconds=b-a,scope_id=e['id'],name=e['name'],category=label,layer=e['layer']) for a,b in pieces)
 segments.sort(key=lambda x:x['start_seconds']);duplicates=0.;cursor=-1.
 for seg in segments:
  duplicates+=max(0.,cursor-seg['start_seconds']);cursor=max(cursor,seg['end_seconds'])
 actual=sum(sums.values());assert duplicates<1e-7 and abs(actual-expected)<1e-7,(actual,expected,duplicates)
 core=sums['稀疏Attention与探针logit导出（融合内核）']
 rows=[dict(operation=k,seconds=v,percent=100*v/expected,times_core=v/core) for k,v in sorted(sums.items(),key=lambda x:-x[1])]
 def csvsave(path,rows):
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 csvsave(OUT/'exclusive_operations.csv',rows);csvsave(OUT/'scope_exclusive.csv',scope_rows)
 with gzip.open(OUT/'exclusive_intervals.csv.gz','wt',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(segments[0]));w.writeheader();w.writerows(segments)
 dma=[]
 for name in ['dma_h2d','dma_d2h']:
  entries=[e for e in events.values() if e['name']==name];spans=[(e['start_seconds'],e['end_seconds']) for e in entries]
  assert abs(length(spans)-length(union(spans)))<1e-7
  dma.append(dict(operation=name,calls=len(entries),seconds=length(spans),bytes=sum(e['nbytes'] for e in entries),inside_global_attention_seconds=overlap(spans,root_spans)))
 dma_spans=[(e['start_seconds'],e['end_seconds']) for e in events.values() if e['lane']=='copy'];dma_sum=sum(e['seconds'] for e in events.values() if e['lane']=='copy');assert abs(dma_sum-length(union(dma_spans)))<1e-7
 monitor=json.loads((ROOT/'outputs/ready_history_exclusive128_20261102_monitor.json').read_text());assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
 layers=json.loads((OUT/'layers.json').read_text());expected_bytes=max(r['history_cpu_bytes'] for r in layers)*23
 assert all(d['bytes']==expected_bytes and d['calls']==460 for d in dma)
 audit=dict(scope='24 global attention modules, excluding QKV Linear and output Linear; residual includes Q/K normalization, RoPE and other preparation',whole_profile_wall_seconds=data['wall_seconds'],whole_profile_cuda_interval_seconds=data['cuda_interval_seconds'],global_attention_inclusive_seconds=length(root_spans),excluded_qkv_projection_seconds=length(union(projections)),target_exclusive_seconds=expected,sum_of_exclusive_rows_seconds=actual,duplicate_seconds=duplicates,partition_error_seconds=actual-expected,number_of_exclusive_segments=len(segments),global_layers=24,source_unchanged=data['source_unchanged'],same_gpu_overlap_detected=False,dma_not_added_to_main_total=True,dma=dma,operations=rows,quality_evaluated=False,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Exclusive timing / DMA timeline diagnostic only',dense_speedup=None,dense_reason='Unchanged Dense/SAF3R not rerun')
 (OUT/'exclusive_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
 text=['# 最新ready-history：全局Attention互斥耗时审计','',
 '7Scenes heads128，当前方法，原checkpoint/输入/配置不变，只增加透明计时scope。未重跑Dense/SAF3R，未做GT质量评测。正常无插桩完整前向中位数23.54043s是已有记录，本次只做1次预热和1次细粒度profile。', '',
 f"本次插桩完整前向wall={data['wall_seconds']:.6f}s；24层global Attention包含式区间={length(root_spans):.6f}s，扣除QKV Linear/输出Linear={length(union(projections)):.6f}s，下面表格分母为剩余{expected:.6f}s。未细分项仍含Q/K归一化、RoPE等准备工作，不能全算作mask构建。不是用23.54秒乘比例倒推阶段时间。",'',
 '统计方法：每个scope有唯一ID和父ID、CUDA起止时间。父scope只计扣除直接子scope区间后的空隙；每个主流时间段只归一个scope和一个分类。融合内核作为一项，不虚拆成多个重复计时。复制流另表，不与主流相加。CUDA事件区间包含CPU发射/排队及等待，不能称为纯GPU内核活跃时间。', '',
 '|互斥操作|秒|占所分析范围 %|相对核心Attention|','|---|---:|---:|---:|']
 for r in rows:text.append(f"|{r['operation']}|{r['seconds']:.6f}|{r['percent']:.3f}|{r['times_core']:.3f}×|")
 text+=['','复制流实际活动（与主流重叠，禁止加到上表）：','', '|方向|次数|GiB|活动秒|其中位于global Attention区间内的秒|','|---|---:|---:|---:|---:|']
 for d in dma:text.append(f"|{d['operation']}|{d['calls']}|{d['bytes']/2**30:.5f}|{d['seconds']:.6f}|{d['inside_global_attention_seconds']:.6f}|")
 text+=['', '缓存读取/缓冲就绪行已包含对应主流区间，不能再把复制流活动加一次。该行也有少量提交/读取控制，不能把它全称为纯等待；复制对带宽竞争的影响已经包含在实测计算时段里，不能由两个区间简单反推净增耗时。', '',
 f"审计：互斥行合计={actual:.9f}s，目标={expected:.9f}s，重复区间={duplicates:.9f}s，差额={actual-expected:.12g}s；共{len(segments)}个互斥时间片、24个global Attention根节点。", '',
 'exclusive_operations.csv为互斥分类；scope_exclusive.csv同时保留包含式和独占时间以便审计，不能将两列相加；exclusive_intervals.csv.gz列出每一时间片的唯一归属；timeline.json保留原事件。', '',
 '|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|同批Dense速度|','|---|---|---|---|---|---|---|','|N/A|N/A|N/A|N/A|N/A|N/A|N/A|','',audit['quality_reason']+'; '+audit['dense_reason']+'.']
 (OUT/'exclusive_report.md').write_text('\n'.join(text)+'\n')
 print(json.dumps({k:audit[k] for k in ['whole_profile_wall_seconds','target_exclusive_seconds','sum_of_exclusive_rows_seconds','duplicate_seconds','partition_error_seconds']},indent=2))
 for r in rows:print(f"{r['operation']}: {r['seconds']:.5f}s {r['percent']:.2f}% {r['times_core']:.3f}x")
if __name__=='__main__':main()
