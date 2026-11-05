"""Assemble audited search endpoints without evaluating/tuning any model."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_quality_speed_final import FIELDS,BASE,table,fmt,csvwrite
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
 base=json.loads((BASE/'summary.json').read_text())
 fast=json.loads((root/'current_probe_summary.json').read_text())
 budget=json.loads((root/'current_probe_budget_summary.json').read_text())
 assert fast['complete'] and budget['complete']
 methods=[('原A12',base['macro']),('快速14%',fast['macro']),('预算扩展17%',budget['macro'])]
 rows=[];timings=[]
 for ds in ['7scenes','eth3d']:
  for label,macros in methods:
   r=next(r for r in macros if r['dataset']==ds);ref=next(r for r in base['macro'] if r['dataset']==ds)
   rows.append(dict(r,label=label))
   timings.append(dict(dataset=ds,method=label,seconds=r['seconds'],reference_batch_seconds=ref['seconds'],reference_batch_speedup=ref['seconds']/r['seconds'],warm_peak_allocated_gib=r.get('warm_peak_allocated_gib',r.get('peak_warm_allocated_gib')),speedup_vs_dense=None,reason='No paired Dense run authorized; original reference target timings are from the preceding batch'))
 recommended={r['dataset']:next(v['method'] for v in timings if v['dataset']==r['dataset'] and v['method']=='快速14%') for r in fast['macro']}
 save(root/'delivery_summary.json',dict(complete=True,methods=rows,timing=timings,default_recommendation=recommended,selection='Frozen independent RGB criteria;no target-truth retuning;fast14 retains primary low-latency role;budget17 is predeclared quality-oriented alternative',global_optimum=False))
 csvwrite(root/'delivery_macro.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]);csvwrite(root/'delivery_timing.csv',timings)
 lines=['# 最终交付：簇数、预算和概率证据的有限搜索','',
 '完成7Scenes全七场景128帧及ETH3D既有11场景（14–76帧）评测。下面是等场景宏平均，AUC/F1为%，距离m，CD为(ACC+Comp)/2。当前A12与两个新端点逐场景checkpoint SHA、RGB帧表、预处理、位姿、融合/对齐、阈值均通过审计。','',
 table(['数据集','方案','AUC3 %','AUC30 %','ACC m','Comp m','CD m','F1 %'],[[r['dataset'],r['label'],*[fmt(k,r[k]) for k in FIELDS]] for r in rows]),'',
 '两个新端点在评测前确定角色，默认推荐快速14%版；预算17%是独立RGB选择的质量优先替代项。不能按目标分数反复搜索预算，也不宣称全局最优。前期两个有退化的端点也完整保留，见report.md。','',
 table(['数据集','新方案','六项均不差于A12','重建退化不超过2%','不慢于上一批A12（仅历史参考）'],[[r['dataset'],r['label'],r['strict_all_six_not_worse'],r['no_more_than2pct_reconstruction_degradation'],r['faster_than_historical_current']] for r in rows if r['label']!='原A12']),'',
 '## 完整前向时间与显存','',
 table(['数据集','方案','完整前向 s','相对上一批A12参考','热allocated峰值 GiB','正式Dense配对加速'],[[r['dataset'],r['method'],f"{r['seconds']:.6f}",f"{r['reference_batch_speedup']:.4f}x",f"{r['warm_peak_allocated_gib']:.3f}",'N/A'] for r in timings]),'',
 '各场景1次预热、3次完整前向中位数，再等场景平均；逐卡监控禁止同卡重叠。全部初始化/聚类/观测/预算/打包/特殊Query/预测头开销包含。A12目标时间来自上一批，不能当作同期配对。独立RGB验证另外在同卡同进程进行，与目标数据集时间分别记录。Dense仅定义基准1.00x，未重跑；无法给出本轮正式Dense配对加速。首次前向、reserved显存和完整采样见两个详细报告。','',
 '## 固定参数与算法变化','',
 table(['参数','原A12','快速14%','预算扩展17%'],[['簇比例','12%','10%','10%'],['逐层保留上限 L0–5 / L6–17 / L18–23','10% / 22% / 10%','10% / 18% / 10%','13% / 21% / 13%'],['Query组','256','256','256'],['概率证据','旧历史估计','每组4个当前层完整Key探针','同快速版'],['Value特征','Head mean','每Head 4维正交草图','同快速版'],['History支持','H union Rep','H union Rep','同快速版'],['Rep补偿','population','预测剩余概率质量/Rep概率','同快速版'],['全簇Exact资格','旧候选筛选','全部可竞争','同快速版'],['预算求解','16轮','16轮','16轮']]),'',
 '新后端先逐Head归一化真实探针logit，再Head/Probe平均。两个流式QK扫描替代Grid/M5c/历史压缩和回填；正式所有Query稀疏QK/AV只执行一次，不导出大概率矩阵。每层更新当前Value聚类与状态，不冻结跨层Mask。Layer0也使用当前观测选状态；这些是显式建模调整，不能说数学等价。默认current_probability_probes=0保留旧后端，未自动覆盖原配置。','',
 '快速版主Mask16%→14%，但探针有额外QK约3.12%Dense；预算扩展平均17%。逐层总逻辑QK连同两遍观测及特殊Query重复工作都<=25%，见per_layer.csv；不把主保留率当作总FLOPs。','',
 '## 哪些Key重要及耗时原因','',
 '独立固定边数反事实中917条有益遗漏，876条位于已被表示的簇，799条未进入上一层历史；只加历史复用并不能找回它们。快速版保留709/917（77.32%），覆盖单边收益权重97.57%，不能把单边收益加总当作最终几何增益。具体层/Query组/原RGB帧/Patch位置见critical_keys_current_probe.csv。校正匹配实际4探针后，64帧24层候选池概率相对误差最大2.19e-6。','',
 '快速版独立128帧互斥profile：稀疏AV1.234s；概率观测0.234s、状态/预算0.178s、Value几何0.102s，三者合计0.514s；其余模型部分占主要时间。所有行互斥，没有把上一层产生历史与当前层接收历史重复计入。完整分解见current_probe_report.md。','',
 '21项CPU/CUDA回归通过。局部误差诊断尾部clamp曾有少量探针重合，已过滤实际重合点并补做冻结候选的严格独立RGB验证；源算法未改变，没有重跑质量选最好重复。更正前后诊断均保留。','',
 '## 历史控制与复现边界','',
 table(['数据集','历史控制','AUC3 %','AUC30 %','ACC m','Comp m','CD m','F1 %'],[[r['dataset'],r['method'],*[fmt(k,r[k]) for k in FIELDS]] for r in fast['historical_controls']]),'',
 'Dense/SAF3R未重跑。旧控制记录缺checkpoint SHA，保留其历史批次标签，不追认严格字节一致，也不用于选新参数。SAF3R速度历史参考7Scenes6.705480s、ETH3D1.603858s，lazy first-correspondence-head Top4/frame，不是论文DINO Top-K；与本轮不能当同期速度对比。','',
 '## 原始结果与复现','',
 '- current_probe_report.md / current_probe_summary.json：快速14%完整报告。',
 '- current_probe_budget_report.md / current_probe_budget_summary.json：预算17%完整报告。',
 '- 两个*_per_scene.csv及*_per_layer.csv：逐场景六指标、逐层状态/预算/QK工作量。',
 '- current_probe_quality/、current_probe_budget_quality/：原始JSON、配置、输入SHA、日志和GPU独占监控。',
 '- phase1–6、validation_phase*、frozen selections：无测试GT搜索与预先冻结证据。',
 '- source_snapshot_store.json / source_snapshot_store/：重复源码快照的SHA寻址无损压缩归档。',
 '', '7Scenes阈值/下采样0.05/0.0078125m；ETH3D0.25/0.0390625m。checkpoint SHA b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969。所有质量运行同时pose+recon_unposed；诊断、失败、kernel测试的六项指标均N/A并附具体原因。无额外数据集质量评测，无7Scenes/ETH3D真值调参。']
 (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(rows=rows,timing=timings),indent=2))

if __name__=='__main__':main()
