"""Report all six fields, matched scopes, attention fidelity, and paired timing."""
import csv
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/shared_value_accuracy_20261105'
KEYS=['auc03','auc30','acc','comp','cd','f1']
SCENES=['heads','chess','stairs','fire','office','pumpkin','redkitchen']


def table(rows,keys):
    def cell(k,v):
        if v is None:return 'N/A'
        if k in ('auc03','auc30','f1'):return f'{100*v:.4f}%'
        if isinstance(v,float):return f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
        ['| '+' | '.join(cell(k,row.get(k)) for k in keys)+' |' for row in rows])


def main():
    historical=json.loads((ROOT/'outputs/shared_value_cluster_scaling_20261104/summary.json').read_text())
    per_scene=[];references=[];checks=[];records=[]
    for dataset,scenes in [('7scenes',SCENES),('eth3d',['courtyard','kicker','playground'])]:
        for scene in scenes:
            path=OUT/f'quality/{dataset}/{scene}/result.json'
            result=json.loads(path.read_text()) if path.exists() else None
            row=dict(dataset=dataset,scene=scene,method='repaired',**{k:None for k in KEYS},complete=False,
                     warmed_seconds=None,peak_gib=None,speedup_vs_dense=None)
            if result:
                row.update(result['quality']['values'],complete=result['quality']['complete'],frames=result['frames'],
                    warmed_seconds=result['timing']['median_seconds'],peak_gib=max(s['peak_allocated_gib'] for s in result['timing']['samples']))
                records.append(result)
                checks.append(dict(dataset=dataset,scene=scene,cd_definition_ok=abs(row['cd']-(row['acc']+row['comp'])/2)<1e-8,
                    retention=result['config']['retention'],query_interval=result['config']['history_query_interval'],
                    config_frozen=result['config']['shared_value_weighting']=='cluster_population' and result['config']['shared_value_allow_observed_rep']))
            per_scene.append(row)
            dense_path=ROOT/f'outputs/sampled_history_quality_20261104/references/dense/{dataset}/{scene}.json'
            if dataset=='7scenes' and scene=='pumpkin':
                # Local source sequence has 773 available image/pose pairs,
                # whereas the imported Nov04 reference sampled all 1000.
                # Use the existing Oct21 control with the exact same 128 paths,
                # indices, image shape and full quality protocol; never select
                # a reference by its score or rerun an unchanged control.
                dense_path=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128/pumpkin/dense/result.json'
            dense=json.loads(dense_path.read_text())
            references.append(dict(dataset=dataset,scene=scene,method='historical Dense',**dense['quality']['values'],
                complete=dense['quality']['complete'],frames=dense['frames'],source=str(dense_path.relative_to(ROOT)),speedup_vs_dense=1.))
            if result:
                checks[-1].update(exact_frame_list=result['input']['image_paths']==dense['input']['image_paths'],
                    exact_frame_indices=result['input']['frame_indices']==dense['input']['frame_indices'],
                    same_shape=result['input']['shape']==dense['input']['shape'],
                    checkpoint_filename=Path(result['input']['checkpoint']).name==Path(dense['input']['checkpoint']).name,
                    quality_protocol=result['quality_protocol']==dense['quality_protocol'])
            old=next((s for s in historical[dataset]['20']['per_scene'] if s['scene']==scene),None)
            if old:references.append(dict(dataset=dataset,scene=scene,method='historical Shared Value',**old['metric'],complete=True,
                frames=old['frames'],source='outputs/shared_value_cluster_scaling_20261104/summary.json',speedup_vs_dense=None))
    macro=[]
    scopes=[('7scenes full7','7scenes',SCENES),('7scenes matched subset3','7scenes',['heads','chess','stairs']),
            ('eth3d subset3','eth3d',['courtyard','kicker','playground'])]
    for scope,dataset,scenes in scopes:
        for method in ['historical Dense','historical Shared Value','repaired']:
            rows=[r for r in per_scene+references if r['dataset']==dataset and r['scene'] in scenes and r['method']==method]
            complete=len(rows)==len(scenes) and all(r['complete'] for r in rows)
            macro.append(dict(scope=scope,method=method,scenes=f'{sum(r["complete"] for r in rows)}/{len(scenes)}',complete=complete,
                **{k:statistics.mean(r[k] for r in rows) if complete else None for k in KEYS},
                speedup_vs_dense=1. if method=='historical Dense' else None))
    diagnostics=[]
    for folder in sorted(OUT.iterdir()):
        p=folder/'audit.json'
        if not p.exists():p=folder/'audit_partial.json'
        if not p.exists():continue
        rows=json.loads(p.read_text());available=[r['errors'].get('executed',r['errors']['unit']) for r in rows]
        tr=folder/'dense_trajectory.json'
        trajectory=json.loads(tr.read_text()) if tr.exists() else []
        diagnostics.append(dict(run=folder.name,layers=len(set(r['layer'] for r in rows)),sampled_group_layers=len(rows),
            local_projected_nmse=statistics.mean(r['projected_nmse'] for r in available),
            counterfactual_cluster_nmse=statistics.mean(r['errors']['cluster_count']['projected_nmse'] for r in rows),
            trajectory_output_nmse=statistics.mean(r['output_nmse'] for r in trajectory) if trajectory else None,
            last_trajectory_output_nmse=trajectory[-1]['output_nmse'] if trajectory else None,
            kernel_vs_explicit_nmse_max=max(r['kernel_vs_explicit_nmse'] for r in rows),
            artifact_complete=(folder/'audit.json').exists()))
    timing_path=OUT/'timing128_lut/result.json'
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else None
    timing_rows=[]
    if timing:
        timing_rows=[dict(method=m,median_seconds=t,speedup_vs_dense=None,
            speedup_vs_original_shared=timing['medians']['original']/t) for m,t in timing['medians'].items()]
    all_runs=[]
    for p in OUT.rglob('quality.json'):
        q=json.loads(p.read_text())
        all_runs.append(dict(run=str(p.parent.relative_to(OUT)),**q['values'],complete=q['complete'],
            speedup_vs_dense=q.get('speedup_vs_vggt'),reason='; '.join(q.get('unavailable',{}).values()),
            speed_reason=q.get('speedup_unavailable')))
    def save(name,data):
        (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    def csvsave(name,data):
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(data)
    save('report_summary.json',dict(per_scene=per_scene,historical_references=references,macro=macro,checks=checks,
        diagnostics=diagnostics,timing=timing,all_runs=all_runs,full7_complete=all(r['complete'] for r in per_scene if r['dataset']=='7scenes'),
        unresolved='No all-layer Dense equivalence; pose and reconstruction do not improve uniformly; relaxed config absent from fetched remote refs'))
    csvsave('per_scene.csv',per_scene+references);csvsave('macro.csv',macro);csvsave('diagnostics.csv',diagnostics);csvsave('all_runs.csv',all_runs)
    text=['# Shared Value Loss · Query-256：精度修复与耗时核查','',
        '分支：`codex/shared-value-accuracy-repair-20261105`。基于 `d7f535ca`；原方案配置 `reuse_shared_value_loss_256.json`。日期按 Asia/Shanghai：2026-11-05。', '',
        '## 结论与限制','',
        '已确认并修正两个问题：代表簇的误差语义与单位执行权重不一致；历史中出现当前代表后不必要地禁用 Rep 候选。修复减少了逐层局部 attention 误差，但不保证跨层轨迹与 Dense 相同，也没有证明六项几何指标全部恢复。原默认配置没有被替换。', '',
        '保持 Query-256、20% 总 Key 上限（含 special tokens）、原簇数、原预处理、四状态和16轮预算求解。新配置显式设置 `shared_value_weighting=cluster_population` 与 `shared_value_allow_observed_rep=true`。', '',
        '## 原因与证据','',
        '1. 原先 Rep loss 假定一个代表近似整个簇，但实际每个 Key 只计单位权重，导致压缩后的 patch 质量相对未压缩 special tokens 偏小。对同一 Q/K/V、同一 mask，仅修改权重即可明显降低 Dense 输出误差；构造中所有簇成员 K/V 相同时，population bias 能精确恢复 Dense 的 softmax 分母与输出，单位权重不能。',
        '2. 四种状态互为替代，不会同时发出重复 Key，因此“当前代表已在历史中出现”不应禁止 Rep。已允许这种 Rep 候选，并继续检查支持集唯一性和预算上限。',
        '3. kernel 与显式稀疏 attention 的最大 NMSE 约为数个 1e-6，明显小于算法的 Dense 误差；共享统计的 CPU 代数与 CUDA/Graph 检查通过。当前证据不支持把主要退化归因于 Triton AV 数值错误。',
        '4. Query-256 的粗概率/历史先验仍有误差；跨层轨迹显示局部接近不等于累计接近。全 head 拼接 Value 和强制全代表 mask 的诊断均未稳定改善累计误差，未采用。', '',
        '## 修复如何执行','',
        '对每个已覆盖簇，令 n 为簇成员数、m 为本次所选成员数，对这些成员添加共享 log(n/m) bias。Rep 为 n 倍；History 为 n/m 倍；Exact 为1倍。完全 Drop 的簇仍丢弃，不向其他簇转移质量。首层继续使用已有 seed population。', '',
        '这是明确的执行语义变化：History 也承接所在簇的近似质量，而原 History loss 仍按未保留成员 Drop 的代理评价。因此它是经诊断支持的修复候选，不是与旧损失完全等价的重写。仍需继续改善状态代理与实际输出的一致性。', '',
        '补偿复用分配器已计算的 eligible_count/population，在一次有界 Triton 遍历中输出权重，不新增 QK、不保存稠密 attention，不扩大20% Key 预算。使用约1MiB的常量 population/divisor 比值表避免逐Key精确除法；263169个表项与GPU FP32除法逐位一致，另有6项数值/Graph测试通过。初始化在首次forward内，预热后的表只读，不缓存输入信息。', '',
        '## 逐层 attention 诊断','',
        '每次抽样15组、每组4个 Query，覆盖24个 global attention 层。local_projected_nmse 使用同一稀疏轨迹上的 Q/K/V，与该层 Dense attention 比较；counterfactual 使用同一个 mask 只改计权。trajectory 则额外运行原 Dense 诊断前向，对相同 token 比较累计输出。三者不可混用。诊断六项几何指标均为 N/A，见 all_runs.csv。', '',
        table(diagnostics,['run','layers','local_projected_nmse','counterfactual_cluster_nmse','trajectory_output_nmse','last_trajectory_output_nmse']), '',
        '原 audit_original8_v2 已完成24层及模型前向，但最后 layers.json 序列化失败；保留 audit_partial.json，不能称为完整质量实验。后续诊断修复了保存逻辑。', '',
        '## 六项质量：按相同范围比较','',
        'AUC/F1 为百分数，原始 JSON/CSV 为 fraction；距离为米，CD=(ACC+Comp)/2，使用非平方距离。7Scenes：阈值0.05m、下采样0.0078125m；ETH3D：阈值0.25m、下采样0.0390625m。均为 pose + recon_unposed，沿用原对齐、预测相机和融合设置。新方法不是 SAF3R，Top-K/profile 不适用；引用的 Dense 也没有 SAF3R patch。', '',
        '新配置在读取7Scenes评测真值前冻结，依据 ETH3D kicker8 的 Dense attention 保真诊断选择。没有依照7Scenes测试指标调参。Dense/原 Shared Value 为历史本地测量参考，来源路径保留；不把旧 RTX5090D 时间作为本机 RTX4090 时间基线。', '',
        '历史 Dense 的精确帧列表、索引、形状、checkpoint文件名和质量协议与新运行逐项核查，见 report_summary.json/checks；旧记录没有 checkpoint 内容摘要，因此不声称已用摘要证明跨机器权重逐字节一致。原 Shared Value 的六项分场景指标来自已同步 summary，缺少本机同批完整重跑。', '',
        '**Pumpkin 输入范围说明：** 本机源序列只有773个可用RGB/pose配对，存在缺失文件；本次从这773个配对均匀采128帧。导入的11月4日Dense参考从1000帧采样，不能直接比较。已按精确输入/协议匹配规则，改用既有10月21日Dense记录（`outputs/global_attention/reuse_explore_20261021/quality128/pumpkin/dense/result.json`），其128条路径、索引、形状和完整质量协议均与本次一致。选择依据是输入匹配，不是分数；不匹配的11月4日记录未混入宏平均。其他六场景仍使用原导入参考。本地七场景均值是这些固定采样范围的等场景平均，不是对原始数据所有序列/帧的穷尽评估。', '',
        table(macro,['scope','method','scenes',*KEYS]), '',
        'full7 只有七个场景全部完成才输出宏平均；旧 Shared Value 仅有三个7Scenes场景，full7标N/A。ETH3D这里只是三个场景子集，不是11场景全集。', '',
        '### 逐场景','',table(per_scene+references,['dataset','scene','method','frames',*KEYS]), '',
        '## 耗时与显存','',
        '配对计时：同一个模型权重、相同 heads128 输入、同一物理 GPU，原方法/修复版交替执行，分别预热2次、正式3次；完整 forward 包含初始化、聚类、路由、历史、attention和预测头。切换 wrapper 和加载 checkpoint/图像不在计时中。两套 Graph 缓存同时驻留，独立峰值显存应看质量运行。', '',
        table(timing_rows,['method','median_seconds','speedup_vs_original_shared','speedup_vs_dense']), '',
        '优化前的有效配对记录保留在 timing128_retry：原方案17.069334s、修复版17.777671s，约增加4.15%。上表对应最终查表实现的独立新批次，不跨GPU比较两批绝对时间，也不删除此前较慢的结果。', '',
        '相对原始 VGGT Dense 的端到端加速比为 N/A：本轮没有重跑未变更 Dense 的配对计时；Dense 本身按定义1.00x。原/新 Shared Value 的相互加速比不能冒充 Dense 加速比。', '',
        '首次 GPU3 计时被异物理进程占卡，监控中止并作废，见 timing_monitor.json；重试记录单独保存。其他资源失败/中断尝试保留，不纳入有效均值。', '',
        table(per_scene,['dataset','scene','warmed_seconds','peak_gib','speedup_vs_dense']), '',
        '## relaxed 参考同步状态','',
        '已同步并检查 Sparse-Attention-VGGT 远程全部可见分支（23个）。当前可见提交树未包含 `configs/evaluation/reuse_rep_m5c_relaxed.json` 或 `saf3r/patch/reuse_relaxed_budget.py`。旧聊天能确认这些文件曾在服务器上未跟踪，但不能据此恢复完整算法。尚未给出与 rep_m5c_relaxed 的真实 mask 重合率；没有用另一个 reuse 版本冒充它。', '',
        '## 原始记录','',
        '`quality/*/*/result.json`：六项原始指标、逐次耗时、硬件、完整输入和协议；`layers.json`：24层预算/密度；source_snapshot.tar/source_sha256.json：实际实现；`*_monitor.json`：独占GPU核查；`all_runs.csv`：包括诊断/失败在内的全部六指标记录；`diagnostics.csv`：逐层诊断汇总。', '',
        '结果仅支持已完成范围内的改善；仍存在的 pose/重建回退和累计轨迹差异必须保留，不能宣称与 Dense 等价。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(macro=macro,timing=timing_rows),indent=2))


if __name__=='__main__':main()
