"""Chinese summary of the completed, validated clustering campaign."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source_local_quality_20261105'
s=json.loads((OUT/'summary.json').read_text());assert s['complete'],'Wait for all 42 valid runs'
g=json.loads((OUT/'graph_summary.json').read_text())
def label(name):
 if name=='adaptive_graph':return '原 adaptive_local（Graph）'
 if name=='adaptive_eager':return '原 adaptive_local（eager）'
 pieces=name.split('_')
 return ('连续编号' if pieces[1]=='sequential' else '时空')+' '+pieces[-2][1:]+'/'+pieces[-1][1:]
lines=['# 固定 source、无持久锚点聚类：速度与精度实验','',
 '完成 42/42 次完整质量评估：原聚类 Graph/eager 两个对照，加上预先固定的 12 组新聚类配置，每组都测 heads、chess、stairs 各 128 帧。所有纳入运行的严格 GPU 监控均未检测到其他任务重叠；同一序列各方法使用同一块物理 RTX4090。失败/中断的尝试在 excluded_attempts 中保留并排除。','',
 '所有配置都保持 Query group=256、Key 保留上限 20%、fine 簇数 26009、局部迭代 2 次、历史采样间隔 256，以及相同的损失函数、四状态、预算、历史规则、attention 内核。实际平均 Key 密度约 19.94%–19.96%。这里 source 宽度表示候选簇数，不是 Query 数。','',
 '相同 checkpoint、逐帧列表、预处理、pose 与 recon_unposed 协议。每次先预热 1 次，再测完整前向 3 次取中位数；包含初始化、布局构建、聚类、路由、历史、attention 和输出头。精度使用末轮输出，不是三次精度的平均。','',
 '下表是三个序列的等权平均，不是完整七场景结果。AUC/F1 为百分数，ACC/Comp/CD 为米；CD=(ACC+Comp)/2，使用非平方距离。重建 F1 阈值 0.05 m，下采样 0.0078125 m。','',
 '| 配置：source 宽度/时间分箱 | 完整前向 s ↓ | 相对原 Graph 加速 | AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ | Dense 加速比 |',
 '|---|---|---|---|---|---|---|---|---|---|']
for r in s['three_scene_means']:
 values=[label(r['variant']),f"{r['seconds']:.6f}",f"{r['relative_to_adaptive_graph']:.3f}x",f"{r['auc03']*100:.4f}",f"{r['auc30']*100:.4f}",f"{r['acc']:.6f}",f"{r['comp']:.6f}",f"{r['cd']:.6f}",f"{r['f1']*100:.4f}",'N/A']
 lines.append('| '+' | '.join(values)+' |')
lines+=['','Dense 加速比均为 N/A：按 AGENTS.md 未重跑未改动的 Dense/SAF3R，历史计时不能当作本批配对测速。匹配的历史本地 Dense/SAF3R 质量表及相对本地 Dense 的质量差异见 report.md；历史 SAF3R 为 lazy 首个对应头 Top-4，不代表论文 DINO Top-K。','',
 '连续编号行不使用时间分箱参数，其 source 可能跨相邻帧边界。所有新聚类行都启用原 Graph 路径，未叠加并行聊天中的 K/V、history 或静态 Graph 优化。时空布局的空间格子随时间分箱调整，以保持候选簇上限；时间分箱是跨度上限，不保证每个 source 含同样多帧。这些是联合布局变体，不能将变化全部归因于时间连续性。实际布局和分组数见 source_layouts.md。','',
 '## Graph 为什么整体慢了','',
 f"新测完整前向三序列均值：Graph {g['three_scene_mean_seconds']['graph']:.6f} s，eager {g['three_scene_mean_seconds']['eager']:.6f} s，启用当前 Graph 实现后整体慢 {(g['three_scene_mean_seconds']['graph']/g['three_scene_mean_seconds']['eager']-1)*100:.2f}%。两者调用同一选边数学流程；单纯重放在隔离窗口里反而更快。",'',
 '真实 heads 128 帧窗口，预热后测量 7 次取中位数：','', '| 操作 | 毫秒 |','|---|---|']
for name,text in [('eager','eager 选边'),('pure_graph_replay','纯 Graph 重放'),('graph_with_input_and_record_copies','重放＋窗口输入/记录拷贝'),('geometry_update','换层几何更新，另计')]:lines.append(f"| {text} | {g['window'][name]['median_seconds']*1000:.6f} |")
c=g['capture'];total=round(c['creation_and_capture_seconds']*1e6);stages={k:round(c['creation_stages_seconds'][k]*1e6) for k in ('gc_collect','synchronize','empty_cache')};remaining=total-sum(stages.values())
lines+=['','同一次创建/捕获的闭合账本，以下子项相加恰好等于创建总时间：','', '| 操作 | 秒 |','|---|---|',f"| Python gc.collect | {stages['gc_collect']/1e6:.6f} |",f"| 同步 | {stages['synchronize']/1e6:.6f} |",f"| 清理 CUDA 缓存 | {stages['empty_cache']/1e6:.6f} |",f'| 分配、拷贝、预运行、捕获及剩余边界 | {remaining/1e6:.6f} |',f'| **总计** | **{total/1e6:.6f}** |','',
 '当前 PyTorch 捕获入口显式执行 synchronize、gc.collect、empty_cache。当前 Graph 缓存键又依赖簇人口分桶的容量、几何宽度、窗口形状；簇变化或 24 项缓存淘汰会在热运行期间继续触发创建。一次重放节省的调度时间很小，多次重新捕获及垃圾回收容易抵消收益。另有输入/输出记录拷贝和换层几何更新。优先优化重复 capture 和缓存拓扑；不要把选边计算本身的重放耗时全部当作 Graph 新增开销。','',
 '单窗口诊断、捕获账本与完整前向属于不同测量范围，不能相互拼接或乘回去冒充完整前向的操作账本。graph_report.md 保存每序列正常计时、每轮缓存命中/创建数及原始诊断。','',
 '## 等价性与精度解释边界','',
 'Graph 没有有意改变 prior、损失、四状态或预算规则，但当前实现不能保证逐位等价。相同当前输入的全窗口检查中，Graph/eager 在 heads/chess/stairs 的 207 个窗口里分别有 2/3/7 个窗口出现支持集差异；连续两次 eager 也分别有 3/3/5 个窗口有差异。chess/stairs 已记录的集合差异为少量 Key（受影响窗口增删 1–10 个），打包位置变化不等于数千个 Key 发生变化。','',
 '代码中的历史损失修正使用 FP32 原子加法，归约次序可变；接近离散预算边界时，小误差可能改变状态或选边。重复 eager 的差异证明存在非确定性，但尚不能证明每一个 Graph/eager 差异都仅由这个因素引起。历史态与 exact 态还可能产生相同支持集而内部状态统计不同。','',
 '因此本表是本次输出的精度测量，不能把单次差值直接解释为稳定提升，尤其不能将 Graph/eager 两次完整运行的质量差异全部归因于 Graph。本轮没有根据测试集真值调整参数或选择最优方案。','',
 '## 文件','',
 '- report.md：全部三序列均值、逐序列六项质量、显存、完整前向时间及历史本地对照。',
 '- summary.json、per_scene.csv、three_scene_means.csv：机器可读汇总。',
 '- 每个 scene/variant/result.json：原始六项质量、全部热/冷计时、协议及配置。',
 '- 每个 run 的 input.json、source_snapshot.tar、source_sha256.json 和严格 GPU monitor：输入、代码和独占监控记录。',
 '- graph_report.md、graph_summary.json：Graph 对照、单窗口计时、捕获账本及等价性检查。']
(OUT/'report_zh.md').write_text('\n'.join(lines)+'\n');print('Chinese report saved')
