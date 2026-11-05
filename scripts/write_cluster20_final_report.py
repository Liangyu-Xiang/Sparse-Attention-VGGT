"""Human-readable final acceptance report; never changes raw scores or gates."""
import json,statistics
from pathlib import Path
BASE=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs')
A=BASE/'cluster20_speed_20261105';B=BASE/'cluster20_speed_conservative_20261105'
a=json.loads((A/'summary.json').read_text());b=json.loads((B/'summary.json').read_text());FIELDS=('auc03','auc30','acc','comp','cd','f1')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def macro(data,ds,method):return next(x for x in data['macro'] if x['dataset']==ds and x['method']==method)
lines=['# VGGT 20%簇：推理优化最终报告','', '**结论：速度目标通过，严格的2%质量变化约束尚未全部通过，不能宣称两项硬约束同时达成。** 两轮结果完整保留，没有放宽阈值、挑选最佳重复、修改预算或依据测试真值调参。','', '固定基线为integrated main的20% patch簇版本：cluster_fraction=0.20；L0–5保留10%、L6–17保留22%、L18–23保留10%；query group256；每层更新当前Value聚类；cluster_population权重；History Top4；16轮预算求解；每帧Camera query的既有完整Key修正保留。模型为VGGT，不是Pi3。','', 'A为第一工程加速候选optimized.json；B为保守候选conservative.json。A包含GPU掩码写回、两级Attention流水、内存复用、RoPE/残差融合、cuDNN自动选核；B仅保留内存复用和已逐位验证的RoPE/残差融合，恢复原Attention流水、原聚类写回和原卷积选择。没有改概率公式、四状态损失、状态选择规则、预算或Attention归一化。','', '## 同卡完整前向速度','', '每场景1次预热+3次完整前向，取中位数；数据集按场景等权平均。每个候选均与本批次SAF3R在同一物理GPU上配对，GPU4/5/6独占，未发现同卡其他CUDA程序。计时包括每次前向的初始化、聚类、路由、概率、历史、重排、内部检查和预测头；模型加载、图像解码及后续TSDF几何评估在计时外。原VGGT Dense未重跑，正式Dense加速比为N/A。','']
rows=[]
for ds in ('7scenes','eth3d'):
 for label,data in [('A',a),('B',b)]:
  m=macro(data,ds,'optimized');s=macro(data,ds,'saf3r');check=next(c for c in data['acceptance'] if c['dataset']==ds)
  rows.append([ds,label,f"{m['seconds']:.6f}",f"{s['seconds']:.6f}",f"{check['speedup_vs_saf3r']:.4f}×",'全部通过' if check['all_scenes_speed_pass'] else '存在未通过',f"{m['peak_warm_allocated_gib']:.3f}",f"{m['mean_cold_seconds']:.3f}",f"{m['peak_all_allocated_gib']:.3f}"])
lines += [table(['数据集','候选','热前向 s','本批SAF3R s','相对SAF3R','逐场景速度','热峰值allocated GiB','平均冷前向 s','含冷前向峰值GiB'],rows),'', '7Scenes为七场景各128帧的现有序列条目，不是完整官方测试划分；ETH3D为11场景、14–76帧、共377帧。两轮所有18个场景的热运行中位数都不慢于各自配对SAF3R。B的7Scenes部分速度余量只有约2%，必须结合原始三次波动阅读。','', 'A的cuDNN自动选核导致很高的首次冷前向开销和瞬时显存；因此不能把6.16秒说成首次加载后即获得的耗时。B没有这项自动选核。表中显存为PyTorch allocated，reserved及逐次数据另见CSV；不把allocated当作全部驱动显存。','', '共同模型路径的内存复用、RoPE和残差融合只对候选启用，SAF3R保留原实现。本报告比较完整实现，不能据此声称我们的稀疏选择算法本身一定比所有优化过的SAF3R实现更快。SAF3R为仓库lazy first-correspondence-head Top4/frame，profile=eth3d-train-fltr-calib_cmpmse.json，不等于论文DINO TopK。SAF3R仅补计时，质量复用历史完整记录。','', '## 六项质量指标','', '所有质量行都完成pose和recon_unposed。AUC/F1在表中为百分数，JSON中为fraction；距离为米；CD=(ACC+Comp)/2，使用未平方距离。7Scenes阈值/下采样为0.05/0.0078125m，ETH3D为0.25/0.0390625m，checkpoint、逐帧列表、预处理、位姿、对齐和融合设置均保持一致。','']
rows=[]
for ds in ('7scenes','eth3d'):
 for label,data,method in [('20%簇参考',a,'reference'),('SAF3R历史质量',a,'saf3r'),('A',a,'optimized'),('B',b,'optimized')]:
  m=macro(data,ds,method);rows.append([ds,label,*[f"{m[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS]])
lines += [table(['数据集','方法','AUC@3 %','AUC@30 %','ACC m','Comp m','CD m','F1 %'],rows),'', '7Scenes的20%簇参考质量此前没有完整匹配记录，因此在A批次补全，B直接复用。ETH3D的参考质量复用既有20%簇11场景结果。旧SAF3R质量记录未保存checkpoint内容SHA；已核对其路径解析至同一现存文件，但不能追溯验证当时字节。当前所有配对计时和新质量运行均记录同一checkpoint SHA256。','', '## 严格2%验收','', '冻结规则为四项重建指标宏平均的**相对变化绝对值<2%**，不是2个百分点。同时额外报告每个场景是否都满足，避免宏平均掩盖个例。改善超过2%也按“绝对变化”规则如实标记，未事后改成只看退化。','']
rows=[]
for label,data in [('A',a),('B',b)]:
 for c in data['acceptance']:
  ds=c['dataset'];bad=[x['scene'] for x in data['per_scene_changes'] if x['dataset']==ds and not x['all_reconstruction_absolute_changes_below2pct']]
  rows.append([ds,label,*[f"{c['relative_quality_change_percent'][k]:+.4f}%" for k in ('acc','comp','cd','f1')],'通过' if c['reconstruction_macro_pass'] else '未通过',','.join(bad) if bad else '无'])
lines += [table(['数据集','候选','ΔACC','ΔComp','ΔCD','ΔF1','宏平均门槛','逐场景超限列表'],rows),'', 'A在7Scenes三项距离宏指标均在2%以内，F1提高2.3224%，因此不满足严格绝对变化门槛；此外部分场景存在实际退化，不能仅以F1平均提升宣称全部通过。B在7Scenes的Comp恶化2.7321%，直接超出2%门槛。两轮ETH3D宏平均都通过；B仍有逐场景超限。两轮失败结果保留，不选择某次重复分数替代，也不通过改预算补分。','', '## 哪些优化在起作用','', '单独数值验证：16组聚类轨迹标签/Rep逐项一致；18组残差dtype/形状逐位一致；7组RoPE尾部和非连续输入测试通过。初版残差融合有4.77e-7差异，已拒绝并改用明确FP32乘法/加法舍入后通过。Attention选参只用固定Q/K/V、Mask和中间量，未用GT；保守版最终恢复原流水参数。','', 'A的heads128独立Profile如下，24层累计。每个区间互不重叠，不能再次与其子步骤或CUDA Kernel self time相加；CUDA Event区间含调度间隙，并非纯硬件占用。此Profile不是B的实测分解。','']
r=json.loads((A/'final_profile/result.json').read_text());total=r['profile_cuda_interval_seconds'];core=r['stages_seconds']['execution_and_evidence'];rows=[]
for name,sec in r['stages_seconds'].items():rows.append([name,f'{sec:.6f}',f'{100*sec/total:.2f}%',f'{sec/core:.3f}×'])
rows.append(['未命名区间：其他模型前向、准备/收尾及间隙',f"{r['outside_named_stages']:.6f}",f"{100*r['outside_named_stages']/total:.2f}%",f"{r['outside_named_stages']/core:.3f}×"])
lines += [table(['操作','累计秒','该Profile占比','相对Attention+证据'],rows),'', '最大单项仍是Sparse Attention+在线证据，约1.596秒。聚类GPU写回优化把诊断区间约0.140秒降至0.091秒，但对全前向改善很小；主要全前向收益来自取消逐block清空CUDA缓存、融合RoPE并免除重复positions.max同步，以及融合残差。各中间Pilot使用不同卡/不同运行，不能把它们时间差当作严格配对的单项消融收益。','', '## 为什么没有承诺质量逐次一致','', '固定输入检查中，参考实现不改任何设置再次前向，Mask在L14开始不同，head-mean Value在L15开始不同，最终pose相对L2=0.001732。说明原流程自身存在运行间数值变化；离散状态选择可能放大微小变化。未将全部差异武断归因于某个原子归约或某项融合。卷积自动选核开/关的前端抽样在本诊断中逐位一致，不支持此前“前端卷积导致全部变化”的假设。完整Mask、特征及Rep比较见trajectory/result.json；不含GT、不是质量验收或速度计时。','', '严格质量条件未全部通过，因此两候选均未作为“已满足全部约束”的最终默认方案。所有原始JSON、逐场景CSV、冷/热计时、显存、GPU独占监控、配置和源码SHA均已保存。没有重跑Dense，也没有重跑SAF3R质量。','', '详细表：[A](report.md)、[B](../cluster20_speed_conservative_20261105/report.md)。数值稳定性：[诊断](numerical_stability.md)。']
(A/'final_report_zh.md').write_text('\n'.join(lines)+'\n')
print('WROTE',A/'final_report_zh.md')
