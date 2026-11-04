"""Fixed validation subset; every control is re-averaged on identical scenes."""
import argparse,csv,hashlib,json,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];KEYS=['auc03','auc30','acc','comp','cd','f1']
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def csvwrite(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
def table(rows,keys):
 def fmt(k,v):
  if v is None:return 'N/A'
  return f'{100*v:.3f}%' if k in ['auc03','auc30','f1'] else f'{v:.6f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(fmt(k,r.get(k)) for k in keys)+' |' for r in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=ROOT/'outputs/cluster_count_quality_20261104');a=p.parse_args();out=a.campaign/'validation_subset';out.mkdir(exist_ok=True,parents=True)
 scope=json.loads((ROOT/'configs/evaluation/validation_subset.json').read_text());save(out/'scope.json',scope);ref=ROOT/'outputs/sampled_history_quality_20261104';rows=[];audits=[];layers=[]
 controls=['dense','saf3r','v8','direct_legacy','probability4','sampled32_baseline'];methods=controls+['clusters15','clusters10','clusters5']
 for ds,scenes in scope['scenes'].items():
  for scene in scenes:
   base=json.loads((ref/f'quality/32/{ds}/{scene}/result.json').read_text())
   for method in methods:
    if method=='sampled32_baseline':src=ref/f'quality/32/{ds}/{scene}/result.json'
    elif method in controls:src=ref/f'references/{method}/{ds}/{scene}.json'
    else:src=a.campaign/f'quality/{method.removeprefix("clusters")}/{ds}/{scene}/result.json'
    d=json.loads(src.read_text());dest=out/f'raw/{method}/{ds}/{scene}.json';save(dest,d);q=d['quality'];v=q['values']
    checks=dict(complete=q['complete'],images=d['input']['image_paths']==base['input']['image_paths'],frame_indices=d['input']['frame_indices']==base['input']['frame_indices'],shape=d['input']['shape']==base['input']['shape'],checkpoint=Path(d['input']['checkpoint']).resolve()==Path(base['input']['checkpoint']).resolve(),protocol=d['quality_protocol']==base['quality_protocol'])
    if method not in controls:
     mp=a.campaign/f'monitors/{method.removeprefix("clusters")}_{ds}_{scene}.json';m=json.loads(mp.read_text());checks['exclusive']=m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1
     ll=json.loads((src.parent/'layers.json').read_text());layers.extend(dict(dataset=ds,scene=scene,method=method,layer=x['layer'],density=x['actual_unique_density'],geometry_width=x['geometry_width'],fine_clusters=x['fine_clusters'],patch_budget=x['patch_budget']) for x in ll)
    assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
    audits.append(dict(dataset=ds,scene=scene,method=method,checks=checks,passed=all(checks.values())))
    rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],**v,complete=q['complete'],reused_control=method in controls,dense_speedup=1. if method=='dense' else None))
 subset=[];delta=[]
 for ds,scenes in scope['scenes'].items():
  for method in methods:
   r=[x for x in rows if x['dataset']==ds and x['method']==method];assert len(r)==len(scenes)
   subset.append(dict(dataset=ds,method=method,scenes=len(r),scope='fixed validation subset',**{k:statistics.mean(x[k] for x in r) for k in KEYS},dense_speedup=1. if method=='dense' else None))
  for method in ['clusters15','clusters10','clusters5']:
   c=next(x for x in subset if x['dataset']==ds and x['method']==method)
   for control in ['dense','sampled32_baseline']:
    b=next(x for x in subset if x['dataset']==ds and x['method']==control)
    delta.append(dict(dataset=ds,method=method,reference=control,**{k+'_absolute_change':c[k]-b[k] for k in KEYS},**{k+'_percentage_point_change':100*(c[k]-b[k]) for k in ['auc03','auc30','f1']}))
 timing=[];profiles=[]
 for v in [15,10,5]:
  d=json.loads((a.campaign/f'timing/{v}/result.json').read_text());m=json.loads((a.campaign/f'monitors/timing_{v}.json').read_text());assert not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['returncode']==0
  timing.append(dict(cluster_percent=v,median_seconds=d['median_seconds'],peak_allocated_gib=max(x['peak_allocated_gib'] for x in d['samples']),gpu=m['gpu'],dense_speedup=None))
  pr=d['profile'];profiles.extend(dict(cluster_percent=v,stage=k,seconds=s,percent=100*s/pr['wall_seconds']) for k,s in pr['exclusive_seconds'].items());profiles.append(dict(cluster_percent=v,stage='outside_named_stages',seconds=pr['unattributed_seconds'],percent=100*pr['unattributed_seconds']/pr['wall_seconds']))
 hashes=json.loads((a.campaign/'source_sha256.json').read_text());runtime=[k for k in hashes if k.startswith(('saf3r/','evaluation/'))];runtime_unchanged=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==hashes[k] for k in runtime)
 complete=all(x['passed'] for x in audits) and runtime_unchanged
 summary=dict(complete=complete,scope=scope,subset_means=subset,per_scene=rows,deltas=delta,timing=timing,profiles=profiles,audits=audits,frozen_runtime_sources_unchanged=runtime_unchanged,quality_runs_in_subset=12,baseline_runs_reexecuted=0,dense_speedup=None,dense_reason='Unchanged Dense was not rerun;no paired warmed full-forward timing',full_dataset_claim=False)
 save(out/'summary.json',summary)
 for name,data in [('subset_means.csv',subset),('per_scene.csv',rows),('deltas.csv',delta),('timing.csv',timing),('profile.csv',profiles),('per_layer.csv',layers)]:csvwrite(out/name,data)
 text=['# 固定困难序列验证：32 Query 历史采样、20% Attention 保留预算','',f'验证完整性：{complete}。质量范围仅为4个固定序列，所有候选和baseline使用同一范围。以下均为子集结果，不是完整数据集指标。','',
 '固定序列：7Scenes **stairs、pumpkin**；ETH3D **relief、relief_2**。依据此前 sampled32 基线在各数据集中最低的两个 F1 选定，不参考本轮候选结果。后续保持固定。','',
 '本轮完整评估在用户缩小范围时已完成；已有原始记录保留，主比较只重新汇总这4个序列，不再执行额外推理。今后的默认验证入口只运行这4个序列。Dense、SAF3R、v8、Direct-legacy、Probability4 和 sampled32 均复用已有匹配的逐场景记录，重新按相同子集等权平均；没有使用它们的完整数据集平均数。','',
 '## 完整前向速度','',table(timing,['cluster_percent','median_seconds','peak_allocated_gib','gpu','dense_speedup']),'',
 '三档均在同一物理 GPU2 RTX5090D、同一 heads128 输入下测量，预热1次、正式3次取中位数，另做一次profile；包含forward内初始化、聚类、路由、历史、重排和预测头。heads128仅作为固定计时输入，不计入上述质量子集平均。每张卡单CUDA进程，监控记录排除了同卡重叠。Dense加速比N/A：遵照只重跑变更方法的要求，没有重跑Dense配对计时；Dense自身定义为1.00x。旧 sampled32 约20%簇数15.222148秒来自 sampled_history_quality_20261104 历史批次，不能作为本轮配对加速比。','',
 '## 相同子集的六项质量均值','',
 '每个数据集仅对选定的2个序列等权平均。AUC/F1为百分数，原始JSON为fraction；距离单位米，CD=(ACC+Comp)/2，非平方距离。7Scenes阈值/下采样0.05/0.0078125m，ETH3D0.25/0.0390625m；每序列相同checkpoint、精确帧列表、预处理、pose协议与无位姿重建设置均已审核。7Scenes每序列128帧；ETH3D使用既有固定输入的实际帧数，详见逐序列表。SAF3R为本地lazy first-correspondence-head Top4/ETH3D profile，不是论文DINO Top-K，也不是Top128。','',table(subset,['dataset','method','scenes',*KEYS]),'',
 '## 逐序列结果','',table(rows,['dataset','scene','method','frames',*KEYS]),'',
 '## 耗时明细','',
 '下表来自独立profile前向，百分比使用各自profile总时间，不与正式中位数混算。Attention包含稀疏计算和历史概率输出；outside_named_stages包含原始共享操作及未分类开销，不能全部称为不可避免的模型成本。','',table(profiles,['cluster_percent','stage','seconds','percent']),'',
 '## 实现与验证记录','',
 '仅cluster_fraction设为15%/10%/5%，历史间隔32、Attention预算20%、状态损失和16轮预算搜索保持固定。首层每簇至少一个代表，以确定性真实成员补足原预算，并从代表权重中扣除这些成员的单位质量，避免重复Key/质量。后续离散选择的实际密度可能低于20%，详见per_layer.csv。','',
 '5%大簇触发旧kernel共享内存超限，以及原256成员几何限制。修复保持width<=128的计算分块不变，对更宽簇缩小history-loss工作块，并将精确几何/drop分桶支持扩展到512；未重分簇或改变状态公式。合成width32–512验证归属完全一致、损失符合FP32归约容差。首轮37个无重叠且width<=128的成功结果被复用，其余17项补测成功；失败与同卡重叠作废记录在attempt1批次中，未混入最终结果。','',
 '所有原始六项指标、源代码快照、显存、帧列表和GPU监控已保留。数值/冒烟/计时/失败实验六项质量字段均标N/A并注明原因；它们不是完整质量评测。未根据测试真值调参或自动晋升新默认配置。冻结的推理与评估实现hash复核通过；报告与后续默认调度入口在用户缩小范围后更新。']
 (out/'report.md').write_text('\n'.join(text)+'\n');shutil.copyfile(out/'report.md',a.campaign/'report.md');print(json.dumps(dict(complete=complete,subset_means=subset,timing=timing),indent=2))
if __name__=='__main__':main()
