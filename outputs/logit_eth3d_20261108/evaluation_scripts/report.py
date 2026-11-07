"""Frozen-formula ETH3D quality report; unchanged fast control reused."""
import json,hashlib,shutil,csv,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/logit_eth3d_20261108';MAIN=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT')
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def csvsave(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(rows):
 keys=['method','scene']+FIELDS
 lines=['| method | scene | AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ |','|---|---|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  lines.append('| '+' | '.join(str(row[k]) if k in ['method','scene'] else f'{row[k]*(100 if k in ["auc03","auc30","f1"] else 1):.4f}' for k in keys)+' |')
 return '\n'.join(lines)
def main():
 camp=json.loads((OUT/'eth3d/campaign_complete.json').read_text());assert camp['complete'] and len(camp['runs'])==11
 hashes=json.loads((OUT/'source_hashes.json').read_text())
 for rel,sha in hashes.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
 records=[];runtime=[];states=[];references={};raw={}
 for expectedfile in sorted((OUT/'eth3d/expected_inputs').glob('*.json')):
  scene=expectedfile.stem;expected=json.loads(expectedfile.read_text());p=OUT/'eth3d/runs'/scene;d=json.loads((p/'result.json').read_text());assert d['quality']['complete'] and d['input']==expected
  ll=json.loads((p/'layers.json').read_text());assert len(ll)==24
  for l in ll:
   assert l['history_source_layer']==l['layer']-1 and l['current_fine_patch_qk_before_mask']==0 and not l['unobserved_fine_history_inherited']
   assert l['actual_block_min']==l['actual_block_max']==925
   states.append(dict(scene=scene,layer=l['layer'],**{f'state_{x}_percent':100*l[f'state_{x}_clusters_mean']/l['fine_clusters'] for x in [0,1,5,10]}))
  mon=json.loads((OUT/'eth3d/monitors'/f'{scene}.json').read_text());assert not mon['same_gpu_overlap_detected'] and mon['returncode']==0
  samples=d['timing']['samples'];runtime.append(dict(scene=scene,frames=d['frames'],warm_median_full_forward_seconds=d['timing']['median_seconds'],warm_peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in samples),cold_peak_allocated_GiB=d['timing']['cold_samples'][0]['peak_allocated_GiB'],speedup_vs_dense='N/A',reason='No paired Dense timing; unchanged controls not rerun'))
  raw[scene]=d['quality'];records.append(dict(method='logit_centered050_new',scene=scene,**d['quality']['values']))
  fast=MAIN/'outputs/quality_speed_search_20261106/current_probe_quality/runs/cp4_c10_prior_residual_middle18/eth3d'/scene/'result.json';ref=json.loads(fast.read_text());assert ref['quality']['complete']
  for key in ['image_paths','frame_indices','shape','checkpoint_sha256']:assert ref['input'][key]==expected[key],(scene,key)
  for rel in ['evaluation/evaluate_da3bench.py','evaluation/benchmarks/da3/da3bench_utils.py','evaluation/benchmarks/da3/bench/utils.py','evaluation/benchmarks/da3/bench/datasets/eth3d.py','evaluation/benchmarks/da3/bench/constants.py']:
   assert ref['source_sha256'][rel]==hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),rel
  assert ref['quality_protocol']['f1_threshold_m']==d['quality_protocol']['f1_threshold_m'] and ref['quality_protocol']['downsample_m']==d['quality_protocol']['downsample_m']
  references[scene]=dict(path=str(fast),sha256=hashlib.sha256(fast.read_bytes()).hexdigest(),matched_checkpoint_input_preprocessing_eval_code=True)
  dst=OUT/'reused_fast'/scene/'result.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fast,dst)
  records.append(dict(method='fast_historical_20261106',scene=scene,**ref['quality']['values']))
 macro=[]
 for method in ['fast_historical_20261106','logit_centered050_new']:
  rows=[r for r in records if r['method']==method];assert len(rows)==11
  macro.append(dict(method=method,scene='macro11',**{f:statistics.mean(r[f] for r in rows) for f in FIELDS}))
 diff={f:macro[1][f]-macro[0][f] for f in FIELDS}
 csvsave(OUT/'per_scene_quality.csv',records);csvsave(OUT/'equal_scene_macro_quality.csv',macro);csvsave(OUT/'runtime_memory.csv',runtime);csvsave(OUT/'per_scene_layer_states.csv',states)
 save(OUT/'quality_raw_by_scene.json',raw);save(OUT/'reference_index.json',references);save(OUT/'quality_difference_vs_fast.json',diff)
 (OUT/'six_metrics.md').write_text(table(macro)+'\n\n'+table(records)+'\n\nSpeedup vs original VGGT Dense: N/A for candidate (no paired timing; unchanged Dense not rerun). Dense reference definition 1.00x.\n')
 text='''# ETH3D：旧 logit 居中收缩公式完整质量评估

固定公式：历史实际执行边的 delta = 上一层真实 logit − 上一层对应块均值 logit。当前历史边 logit = 当前块均值 + 0.5 × (delta − 上一层同块实际观测支持上的 delta 均值)。非历史边只采用当前块均值。逐 head 重建 softmax 后平均全部16头与4个普通Query probe，用于状态损失。历史缓存只含紧邻上一层实际 sparse mask 边，不保存未执行边估计，不滚动继承旧值；没有当前层选边前细粒度QK校验。正式稀疏AV计算当前所选边，并导出已有logit作为下一层证据。特殊token排除于普通估计，正式AV仍保留原特殊token处理。

Query group256；Key配置目标1024，ETH3D实际925/块，每帧一块。保持四状态、Value聚类、10%/18%/10%层段保留率及sqrt(population residual × estimated prior residual)补偿。不增加新操作，不按本次ETH3D分数调整模型。

全部11场景，精确输入帧14–76、同一checkpoint SHA、350×518预处理、pose和recon_unposed，和既有快速版输入及评估代码SHA逐项匹配。快速版是20261106本地历史结果，不是本轮重跑或论文值。两者除历史建模外，还存在快速版当前4probe全Key细粒度概率与本方法粗块估计的区别，因此不是只改变一个系数的消融。

## 等场景宏平均

'''+table(macro)+'''\n
AUC与F1为百分数；ACC/Comp/CD为米；CD=(ACC+Comp)/2，未平方。完整逐场景表见six_metrics.md和per_scene_quality.csv，原始任务指标在各run/result.json。

重建阈值0.25m；下采样/TSDF voxel0.0390625m；sdf_trunc0.2m；mesh采样1000000点；原unposed Umeyama/RANSAC尺度及位姿对齐、random_state42，原深度最近邻上采样及GT有效mask。两方法均非SAF3R，Top-K/profile=N/A。

## 时间和显存

每场景1次冷/预热、3次热完整前向中位数，包括初始化、聚类、路由、粗估计、历史导出更新、重排和预测头。CPU质量评估不计入完整前向。11场景热前向中位数等权均值为'''+f'{statistics.mean(r["warm_median_full_forward_seconds"] for r in runtime):.4f}s'+'''，逐场景时间/峰值见runtime_memory.csv。GPU4/5/6每卡单任务，有全程独占监测。

相对原始VGGT Dense端到端加速比=N/A：没有同期同硬件配对热完整前向，不为计时重跑未修改Dense。Dense基准定义1.00x。历史快速版时间不能用于本轮严格速度比较。没有7Scenes128帧速度验收，也没有混用ScanNet局部注意力输出误差。
'''
 (OUT/'REPORT.md').write_text(text);save(OUT/'audit.json',dict(complete=True,scenes=11,quality_fields=6,all24layers_per_scene=True,controls_reused_not_rerun=True,strict_fast_input_checkpoint_and_eval_code_match=True,source_hashes_match=True,speedup_vs_dense=None))
 print(table(macro));print('DIFFERENCE',diff)
if __name__=='__main__':main()
