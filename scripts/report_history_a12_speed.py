"""Audit corrected12% History-A runs and report six metrics without retuning."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs');FIELDS=('auc03','auc30','acc','comp','cd','f1');METHOD='history_A_special5_cluster12_fast'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;plan=read(out/'plan.json');done=read(out/'completion.json');assert done['complete'] and len(done['runs'])==36
 rows=[];times=[];audits=[];layerrows=[];changes=[]
 for ds,scenes in plan['datasets'].items():
  for sc in scenes:
   folder=out/'runs'/ds/sc;new=read(folder/METHOD/'result.json');sf=read(folder/'saf3r/timing_result.json');q=new['quality'];assert q['complete']
   assert new['input']==sf['input'] and new['quality_protocol']==sf['quality_protocol']
   assert new['input']['checkpoint_sha256']==plan['checkpoint_sha256']
   config=new['config'];assert config['cluster_fraction']==.12 and config['dense_special_queries']==5 and config['rep_history_loss']=='A'
   assert config==plan['effective_config'];assert abs(q['values']['cd']-(q['values']['acc']+q['values']['comp'])/2)<1e-9
   monitors={name:read(out/'monitors'/f'{ds}_{sc}_{name}.json') for name in ('saf3r',METHOD)}
   assert monitors['saf3r']['gpu_uuid']==monitors[METHOD]['gpu_uuid']
   for name,raw in [('saf3r',sf),(METHOD,new)]:
    m=monitors[name];assert not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1 and m['returncode']==0
    log=(out/'logs'/f'{ds}_{sc}_{name}.log').read_text();assert log.count('FORWARD ')==4
    if name=='saf3r':assert log.count('Computing QK TopK indices')==4
    t=raw['timing'];assert t['warmup']==1 and t['repeats']==3 and len(t['samples'])==3
    assert abs(t['median_seconds']-statistics.median(x['seconds'] for x in t['samples']))<1e-10
    for i,sample in enumerate(t['cold_samples']+t['samples']):times.append(dict(dataset=ds,scene=sc,method=name,iteration=i,cold=i<len(t['cold_samples']),**{k:sample[k] for k in ('seconds','peak_allocated_gib','peak_reserved_gib')}))
   t=new['timing'];st=sf['timing'];ratio=st['median_seconds']/t['median_seconds']
   rows.append(dict(dataset=ds,scene=sc,method=METHOD,frames=new['frames'],**q['values'],seconds=t['median_seconds'],peak_warm_allocated_gib=max(x['peak_allocated_gib'] for x in t['samples']),peak_warm_reserved_gib=max(x['peak_reserved_gib'] for x in t['samples']),peak_all_allocated_gib=max(x['peak_allocated_gib'] for x in t['samples']+t['cold_samples']),cold_seconds=t['cold_samples'][0]['seconds'],saf3r_seconds=st['median_seconds'],speedup_vs_saf3r=ratio,speedup_vs_dense=None))
   layers=read(folder/METHOD/'layers.json');assert len(layers)==24
   shape=new['input']['shape'];patch=new['frames']*(shape[-2]//14)*(shape[-1]//14);expected=math.floor(.12*patch)
   for r in layers:
    assert r['fine_clusters']==expected and r['dense_special_query_count']==new['frames']*5
    assert r['query_group_size']==256 and r['actual_unique_density']<=.25+1e-6
    layerrows.append(dict(dataset=ds,scene=sc,layer=r['layer'],fine_clusters=r['fine_clusters'],patch_budget=r['patch_budget'],special_keys=r['special_keys'],dense_special_query_count=r['dense_special_query_count'],actual_unique_density=r['actual_unique_density']))
   audits.append(dict(dataset=ds,scene=sc,checkpoint_equal=True,inputs_equal=True,protocol_equal=True,physical_gpu=monitors[METHOD]['gpu_uuid'],correct_history_A=True,all_special_queries=True,clusters12=True,speed_pass=ratio>=1))
   if ds=='eth3d':
    ref=read(ROOT/'outputs/rep_history_A_all_specials_20261105/quality/A_all_specials/runs'/sc/'result.json');assert ref['quality']['complete'] and ref['input']['checkpoint_sha256']==plan['checkpoint_sha256']
    for key in ('image_paths','frame_indices','shape'):assert ref['input'][key]==new['input'][key]
    change={k:100*(q['values'][k]/ref['quality']['values'][k]-1) for k in FIELDS}
    changes.append(dict(scene=sc,**change,pointcloud_abs_change_below2pct=all(abs(change[k])<2 for k in ('acc','comp','cd','f1'))))
 macros=[];references=[];gates={}
 for ds,scenes in plan['datasets'].items():
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==len(scenes)
  macro=dict(dataset=ds,method=METHOD,scenes=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in FIELDS},seconds=statistics.mean(r['seconds'] for r in rr),saf3r_seconds=statistics.mean(r['saf3r_seconds'] for r in rr),peak_warm_allocated_gib=max(r['peak_warm_allocated_gib'] for r in rr),peak_warm_reserved_gib=max(r['peak_warm_reserved_gib'] for r in rr),peak_all_allocated_gib=max(r['peak_all_allocated_gib'] for r in rr),mean_cold_seconds=statistics.mean(r['cold_seconds'] for r in rr),speedup_vs_dense=None)
  macro['speedup_vs_saf3r']=macro['saf3r_seconds']/macro['seconds'];macros.append(macro)
  gates[ds]=dict(all_scene_speed_pass=all(r['speedup_vs_saf3r']>=1 for r in rr),macro_speed_pass=macro['speedup_vs_saf3r']>=1,all128_below9=all(r['seconds']<9 for r in rr if r['frames']==128))
  for method in ('dense','saf3r'):
   records=[read(BASE/'combined_reuse_20261031'/ds/sc/method/'result.json') for sc in scenes];assert all(r['quality']['complete'] for r in records)
   for record in records:
    matched=next(r for r in rows if r['dataset']==ds and r['scene']==record['scene']);assert record['frames']==matched['frames']
   references.append(dict(dataset=ds,method=method+'_historical',scenes=len(scenes),**{k:statistics.mean(r['quality']['values'][k] for r in records) for k in FIELDS},batch='combined_reuse_20261031; historical quality, not rerun',checkpoint_note='Old control records lack file SHA; original checkpoint path plus current active-tensor identity audit available, no retrospective byte-hash assertion'))
  if ds=='eth3d':
   records=[read(ROOT/'outputs/rep_history_A_all_specials_20261105/quality/A_all_specials/runs'/sc/'result.json') for sc in scenes]
   ref=dict(dataset=ds,method='A_all_specials_cluster12_historical',scenes=len(scenes),**{k:statistics.mean(r['quality']['values'][k] for r in records) for k in FIELDS},batch='rep_history_A_all_specials_20261105; RTX4090/Torch2.8')
   references.append(ref);delta={k:100*(macro[k]/ref[k]-1) for k in FIELDS};gates[ds].update(relative_change_percent=delta,pointcloud_macro_abs_change_below2pct=all(abs(delta[k])<2 for k in ('acc','comp','cd','f1')),pointcloud_all_scene_abs_change_below2pct=all(x['pointcloud_abs_change_below2pct'] for x in changes),interpretation='Historical same-checkpoint reproduction drift, not an isolated same-hardware engineering-quality delta')
  else:gates[ds].update(pointcloud_macro_abs_change_below2pct=None,reason='No complete matched A_all_specials 12% seven-scene reference; legacy-main scores are not substituted')
 for rel,sha in plan['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
 summary=dict(complete=True,macro=macros,historical_quality=references,acceptance=gates,per_scene_changes_eth3d=changes,source_unchanged=True,reference_7scenes=dict(values={k:None for k in FIELDS},reason='No existing complete matched A_all_specials12% seven-scene quality record'),dense_speedup_reason='Dense rerun not authorized; no paired current Dense timing')
 write(out/'summary.json',summary);write(out/'audit.json',dict(passed=True,runs=36,quality_runs=18,timing_only_runs=18,source_unchanged=True,per_scene=audits));csvwrite(out/'per_scene.csv',rows);csvwrite(out/'macro.csv',macros);csvwrite(out/'timing.csv',times);csvwrite(out/'per_layer.csv',layerrows);csvwrite(out/'quality_changes_eth3d.csv',changes)
 lines=['# 修正基线后的 History Loss A＋全特殊 Query 修正＋12%簇：完整评测','', '本轮采用正确的A_all_specials基线，非上一轮误用的legacy main，也不将旧6.161秒挪作本方案成绩。History support=H∪{Rep}；非Rep历史Key权重1，Rep权重population−(union_count−1)；全部5个特殊Query/帧做全Key输出修正，特殊Key常驻；Exact候选按实际stride读取。簇比例12%，Query组256，当前Value每层更新，16轮预算，层保留率10%/22%/10%未改。工程选项与参考配置差异见plan.json。','', '完整18场景新方法pose＋recon_unposed均完成；7Scenes七场景各128帧，ETH3D既有11场景14–76帧，共377帧。宏平均等场景权重。AUC/F1为百分数，距离米；CD=(ACC+Comp)/2未平方。7Scenes重建阈值/下采样0.05/0.0078125m，ETH3D0.25/0.0390625m，TSDF融合/对齐和输入精确帧表保持原协议。','', '使用官方tracker-fixed文件SHA256 b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969。与之前d15文件的逐Tensor比较发现非tracking参数完全一致；tracking在本轮未调用，不能将之前质量差简单归因于文件SHA不同。历史A12来自RTX4090/Torch2.8，本轮为RTX5090D/Torch2.9，历史分数差异不等于隔离后的工程优化损失。','', '## 六项质量','']
 qualityrows=[]
 for ds in plan['datasets']:
  for r in [x for x in references if x['dataset']==ds]+[x for x in macros if x['dataset']==ds]:qualityrows.append([ds,r['method'],r['scenes'],*[f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS]])
 lines += [table(['数据集','方法','场景','AUC@3 %','AUC@30 %','ACC m','Comp m','CD m','F1 %'],qualityrows),'', 'Dense/SAF3R质量复用历史完整记录，未重跑；旧控制记录缺少checkpoint文件SHA，保留历史参考标签。7Scenes缺少同一A_all_specials12%方案的全七场景原质量，不能用上一轮legacy-main结果替代并声称<2%退化。','', '## 当前配对速度和显存','', 'GPU4/5/6，逐场景同卡独占；SAF3R仅做获准的计时，不重跑质量。每方法1预热＋3次完整前向中位数，再按场景等权平均；包含reset、聚类、路由、历史、重排、所有内部检查和预测头，不含模型/图像磁盘加载及后续几何评估。SAF3R为仓库lazy首个对应头Top4/frame，非论文DINO TopK。每次TopK重置在计时内。Dense没有重跑，正式Dense加速比N/A。','',table(['数据集','新方法 s','配对SAF3R s','vs SAF3R','热峰值allocated GiB','平均冷前向 s','含冷前向峰值 GiB'],[[r['dataset'],f"{r['seconds']:.6f}",f"{r['saf3r_seconds']:.6f}",f"{r['speedup_vs_saf3r']:.4f}×",f"{r['peak_warm_allocated_gib']:.3f}",f"{r['mean_cold_seconds']:.3f}",f"{r['peak_all_allocated_gib']:.3f}"] for r in macros]),'', 'cuDNN自动选核的冷前向代价与热耗时分开记录；不能把热运行数字当首次冷启动时间。reserved和每次样本见timing.csv。共同路径融合仅对新方案启用，SAF3R保留原执行，因此这是实现端到端比较，不证明选择器算法本身更快。','', '## 验收与复现边界','',json.dumps(gates,indent=2,ensure_ascii=False),'', 'ETH3D变化按相对变化绝对值<2%逐项报告，改善超2%也如实标记；不改门槛、不依据测试真值调预算、不挑选最佳重复。由于历史硬件/框架不同，这些是历史复现差异。','', '## 新方法逐场景完整结果','',table(['数据集','场景','帧','AUC3 %','AUC30 %','ACC m','Comp m','CD m','F1 %','热前向 s','vs SAF3R'],[[r['dataset'],r['scene'],r['frames'],*[f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS],f"{r['seconds']:.6f}",f"{r['speedup_vs_saf3r']:.4f}×"] for r in rows]),'', '全量原始JSON、输入列表、配置、源码哈希、GPU监控、逐层簇数/预算/密度和冷/热计时全部保存；旧实验未覆盖。']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(macro=macros,acceptance=gates),indent=2))
if __name__=='__main__':main()
