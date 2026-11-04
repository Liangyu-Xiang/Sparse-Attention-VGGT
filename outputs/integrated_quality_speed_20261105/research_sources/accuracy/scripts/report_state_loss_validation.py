"""Matched six-metric validation, 24-layer states/similarities and paired timings."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_record,quality_markdown,FIELDS,quality_cells

SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],
        'eth3d':['kicker','playground','courtyard']}
VARIANTS=['A_history_self_residual_rep','B_history_group']


def load(path):return json.loads(path.read_text() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes()))
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def csvsave(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def pool(scores):
    return dict(relative_l2=math.sqrt(sum(s['error_sum_sq'] for s in scores)/max(1e-30,sum(s['dense_sum_sq'] for s in scores))),
        cosine=sum(s['cosine_sum'] for s in scores)/max(1,sum(s['queries'] for s in scores)),queries=sum(s['queries'] for s in scores))
def same_input(a,b):
    return {k:a[k]==b[k] for k in ['image_paths','frame_indices','shape']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;old=ROOT/'outputs/shared_value_accuracy_20261105'
    states=[];similarities=[];diag_summary=[];checks=[]
    runs=[('原方案','kicker',old/'layer_details_original31'),('上一修复版','kicker',old/'layer_details_repaired31'),
        ('A_history_self_residual_rep','kicker',out/'matched_residual_kicker'),
        ('B_history_group','kicker',out/'matched_group_centered_kicker'),
        ('A_history_self_residual_rep','playground',out/'rollout_residual_playground_v2'),
        ('B_history_group','playground',out/'rollout_group_centered_playground_v2')]
    reference=load(old/'layer_details_original31/input.json')
    for method,scene,folder in runs:
        if not (folder/'dense_trajectory.json').exists():continue
        inp=load(folder/'input.json');layers=load(folder/'layers.json');audit=load(folder/'audit.json');trajectory=load(folder/'dense_trajectory.json')
        if scene=='kicker':
            check=dict(run=str(folder),**same_input(inp,reference),sampled_queries=inp['sampled_query_indices']==reference['sampled_query_indices'])
            checks.append(check)
            if not all(v for k,v in check.items() if k!='run'):raise ValueError(f'Unmatched layer inputs: {folder}')
        for row in layers:
            c=row['fine_clusters'];state=dict(method=method,scene=scene,layer=row['layer'],
                **{f'state_{k}':row[f'state_{k}_clusters_mean']/c for k in [0,1,5,10]},
                density=row['actual_unique_density'],history_key_reuse=row.get('actual_history_key_reuse_fraction_mean'))
            states.append(state)
        for layer in range(24):
            for subset in ['all','camera','probe_non_camera']:
                scores=[r['output_similarity'][subset]['projected'] for r in audit if r['layer']==layer and subset in r['output_similarity']]
                if scores:similarities.append(dict(method=method,scene=scene,layer=layer,comparison='local_same_QKV',subset=subset,**pool(scores)))
                tr=next(r for r in trajectory if r['layer']==layer)
                if subset=='all':
                    s=dict(relative_l2=tr['output_relative_l2'],cosine=tr['output_cosine'],queries=len(inp['sampled_query_indices']))
                else:s=pool([tr['output_'+subset]])
                similarities.append(dict(method=method,scene=scene,layer=layer,comparison='independent_Dense_trajectory',subset=subset,**s))
        for comparison in ['local_same_QKV','independent_Dense_trajectory']:
            for subset in ['camera','probe_non_camera']:
                if comparison=='local_same_QKV':scores=[r['output_similarity'][subset]['projected'] for r in audit if 10<=r['layer']<=16 and subset in r['output_similarity']]
                else:scores=[r['output_'+subset] for r in trajectory if 10<=r['layer']<=16]
                diag_summary.append(dict(method=method,scene=scene,comparison=comparison,subset=subset,**pool(scores)))
    quality=[];quality_states=[]
    for dataset,scenes in SCENES.items():
        for scene in scenes:
            dense_path=ROOT/f'outputs/sampled_history_quality_20261104/references/dense/{dataset}/{scene}.json'
            if dataset=='7scenes' and scene=='pumpkin':dense_path=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128/pumpkin/dense/result.json'
            dense=load(dense_path)
            reference_rows=[('历史本地Dense',dense_path),('上一修复版',old/f'quality/{dataset}/{scene}/result.json')]
            for method,path in reference_rows+[(m,out/f'quality/{m}/{dataset}/{scene}/result.json') for m in VARIANTS]:
                result=load(path) if path.exists() else None
                q=result['quality'] if result else quality_record({},'Run not complete or not yet available; inspect progress and monitor logs')
                row=dict(dataset=dataset,scene=scene,method=method,**q['values'],complete=q['complete'],source=str(path),
                    frames=None if result is None else result['frames'],speedup_vs_dense=1. if method=='历史本地Dense' else None,
                    warmed_seconds=None if result is None else result['timing']['median_seconds'],
                    peak_allocated_gib=None if result is None else max(s['peak_allocated_gib'] for s in result['timing']['samples']))
                if result:
                    check=dict(run=str(path),**same_input(result['input'],dense['input']),quality_protocol=result['quality_protocol']==dense['quality_protocol'],
                        checkpoint_filename=Path(result['input']['checkpoint']).name==Path(dense['input']['checkpoint']).name)
                    checks.append(check)
                    if not all(v for k,v in check.items() if k!='run'):raise ValueError(f'Unmatched quality inputs: {path}')
                    if row['complete'] and abs(row['cd']-(row['acc']+row['comp'])/2)>1e-8:raise ValueError('CD definition mismatch')
                    if method in VARIANTS:
                        for layer in load(path.parent/'layers.json'):
                            quality_states.append(dict(dataset=dataset,scene=scene,method=method,layer=layer['layer'],density=layer['actual_unique_density'],
                                **{f'state_{k}':layer[f'state_{k}_clusters_mean']/layer['fine_clusters'] for k in [0,1,5,10]},
                                history_key_reuse=layer.get('actual_history_key_reuse_fraction_mean')))
                quality.append(row)
    macros=[]
    for dataset,scenes in SCENES.items():
        for method in ['历史本地Dense','上一修复版']+VARIANTS:
            rows=[r for r in quality if r['dataset']==dataset and r['method']==method]
            complete=len(rows)==len(scenes) and all(r['complete'] for r in rows)
            macros.append(dict(scope='7Scenes full7' if dataset=='7scenes' else 'ETH3D subset3',method=method,complete=complete,
                scenes=f'{sum(r["complete"] for r in rows)}/{len(scenes)}',**{k:statistics.mean(r[k] for r in rows) if complete else None for k in FIELDS}))
    macro_deltas=[]
    for row in macros:
        if row['method'] not in VARIANTS or not row['complete']:continue
        for reference_name in ['历史本地Dense','上一修复版']:
            ref=next(r for r in macros if r['scope']==row['scope'] and r['method']==reference_name)
            macro_deltas.append(dict(scope=row['scope'],method=row['method'],reference=reference_name,
                **{k:row[k]-ref[k] for k in FIELDS},units='AUC/F1 fraction difference; ACC/Comp/CD metres; positive means higher value, not necessarily better'))
    macro_states=[]
    for dataset,scenes in SCENES.items():
        for method in VARIANTS:
            for layer in range(24):
                rows=[r for r in quality_states if r['dataset']==dataset and r['method']==method and r['layer']==layer]
                if {r['scene'] for r in rows}!=set(scenes):continue
                macro_states.append(dict(dataset=dataset,method=method,layer=layer,scenes=len(scenes),
                    **{k:statistics.mean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
                       for k in ['density','history_key_reuse','state_0','state_1','state_5','state_10']}))
    timing=[]
    for method,tag in [(VARIANTS[0],'paired_residual_timing_v2'),(VARIANTS[1],'paired_group_timing_v2'),
        (VARIANTS[0]+' / sparse sums experiment','paired_residual_fast_timing'),
        (VARIANTS[1]+' / sparse sums experiment','paired_group_fast_timing')]:
        path=out/tag/'result.json'
        if path.exists():
            r=load(path);monitor=load(out/f'{tag}_monitor.json')
            if monitor['aborted'] or monitor['same_gpu_overlap_detected']:raise ValueError('Invalid paired timing')
            timing.append(dict(method=method,original_seconds=r['medians']['original'],candidate_seconds=r['medians']['repaired'],
                increase_fraction=r['runtime_increase_fraction'],speedup_vs_original_shared=r['speedup_vs_original_shared'],speedup_vs_dense=None,source=str(path)))
    # Every externally interrupted attempt and kernel test gets six explicit unavailable fields.
    attempts=[]
    for monitor_path in sorted(out.glob('*_monitor.json')):
        m=load(monitor_path)
        if m['returncode']==0 and not m['aborted']:continue
        reason=f"Externally interrupted/failed diagnostic (returncode={m['returncode']}, foreign CUDA PIDs={m['foreign_gpu_pids']}); no complete pose/recon_unposed evaluation"
        attempts.append(dict(run=monitor_path.stem,**quality_record({},reason),monitor=str(monitor_path)))
    for monitor_path in sorted((out/'quality').rglob('*_monitor.json')):
        m=load(monitor_path)
        if m['returncode']==0 and not m['aborted']:continue
        attempts.append(dict(run=monitor_path.stem,**quality_record({},'GPU exclusivity monitor interrupted this attempt; no complete quality result'),monitor=str(monitor_path)))
    for tag in ['kernel_test','kernel_test_v2','kernel_test_v3','kernel_test_fast']:
        folder=out/tag;folder.mkdir(exist_ok=True)
        q=quality_record({},'Numerical kernel/Graph test only; pose and recon_unposed not evaluated')
        save(folder/'quality.json',q);(folder/'quality.md').write_text(quality_markdown(q))
    joint=[]
    from scripts.report_state_loss_search import summarize
    for tag in ['joint_kicker','joint_playground']:
        path=out/tag/'loss_candidates.json'
        if not path.exists() and not Path(str(path)+'.gz').exists():continue
        rows=load(path)
        if {r['layer'] for r in rows}!=set(range(24)):continue
        for method in sorted({r['method'] for r in rows}):
            middle=[r for r in rows if r['method']==method and 10<=r['layer']<=16]
            s=summarize(middle);before=[r['predicted_objective_before'] for r in middle if r.get('predicted_objective_before') is not None]
            after=[r['predicted_objective_after'] for r in middle if r.get('predicted_objective_after') is not None]
            joint.append(dict(run=tag,method=method,**s,predicted_objective_after_over_before=sum(after)/max(1e-30,sum(before)) if before else None,
                quality=quality_record({},'Offline frozen-feature joint-output diagnostic; no full rollout/pose/recon_unposed/timing evaluation')))
    save(out/'validation.json',dict(quality=quality,macro=macros,macro_deltas=macro_deltas,checks=checks,diagnostic_middle=diag_summary,timing=timing,failed_attempts=attempts,joint_diagnostics=joint,
        limitation='Losses and output statistics remain approximations; no all-layer Dense equivalence or mask comparison to missing rep_m5c_relaxed implementation'))
    csvsave(out/'quality_per_scene.csv',quality);csvsave(out/'quality_macro.csv',macros)
    csvsave(out/'layer_states.csv',states);csvsave(out/'layer_similarities.csv',similarities);csvsave(out/'quality_layer_states.csv',quality_states)
    csvsave(out/'quality_layer_state_macro.csv',macro_states);csvsave(out/'quality_macro_deltas.csv',macro_deltas)
    save(out/'failed_attempts.json',attempts)
    details=['# 24层完整状态与输出相似度','','状态为簇占比；历史复用为实际执行patch Key中曾在上一层支持出现的比例。它与History状态占比不同。',
        '索引从0开始。局部比较使用相同稀疏QKV；累计比较使用独立Dense轨迹。输出均在Wo之后、残差/LayerScale之前。RelL2为百分数，余弦无量纲。',
        '这些是诊断，pose/recon_unposed六项均N/A，详见各运行quality.md；运行时间/显存为带插桩诊断，不当作预热计时。','']
    for method,scene,folder in runs:
        selected=[r for r in states if r['method']==method and r['scene']==scene]
        if not selected:continue
        details += [f'## {scene} / {method}','','| 层 | Drop % | Rep % | History % | Exact % | 历史Key复用 % | 总Key密度 % |',
            '|---:|---:|---:|---:|---:|---:|---:|']
        for r in selected:
            vals=[r[f'state_{s}']*100 for s in [0,1,5,10]]+[None if r['history_key_reuse'] is None else r['history_key_reuse']*100,r['density']*100]
            details.append('| '+str(r['layer'])+' | '+' | '.join('N/A' if v is None else f'{v:.3f}' for v in vals)+' |')
        if any(r['history_key_reuse'] is None for r in selected):details += ['','旧运行未记录实际历史Key复用，不用History状态比例代填。']
        for comparison in ['local_same_QKV','independent_Dense_trajectory']:
            details += ['',f'### {comparison}','','| 层 | 普通查询余弦 | 普通查询RelL2 % | Camera余弦 | Camera RelL2 % |','|---:|---:|---:|---:|---:|']
            for layer in range(24):
                found={r['subset']:r for r in similarities if r['method']==method and r['scene']==scene and r['comparison']==comparison and r['layer']==layer}
                p=found['probe_non_camera'];c=found['camera']
                details.append(f"| {layer} | {p['cosine']:.6f} | {p['relative_l2']*100:.3f} | {c['cosine']:.6f} | {c['relative_l2']*100:.3f} |")
            details += ['']
    (out/'layer_details.md').write_text('\n'.join(details).rstrip()+'\n')
    text=['# 状态损失修复：两种 History 语义的实测对照','',
        '**本轮没有得到可替换原方案的合格修复版。** 状态损失与执行语义已统一，但完整重建质量仍有回退，耗时也有增加。A/B均保留为实验对照，不提升为默认。','',
        '分支 `codex/shared-value-state-loss-search-20261105`；原默认配置保持不变。公式和34候选解释见 `docs/shared_value_state_losses.md`，固定特征筛选见 `loss_report/report.md`。','',
        'A：普通 History 权重1，剩余成员由显式 Rep 承担。B：History 集合共同代表整簇，使用 π/保留估计质量校准。两者均使用中心化 Drop、输出空间8维投影和簇内均匀未知概率。',
        '保持原 Query-256 历史采样、每32 Query共享支持集、20%总Key上限（含Special）、原聚类与16轮预算求解。没有强制History配额；缺少历史信息的当前成员不计算QK用于选mask。','',
        '## 中间层输出诊断（层索引10–16，从0开始）','',
        'Kicker 原/旧修复/A/B使用完全相同的帧、预处理和91个查询（31 camera+60普通探针）；JSON checks核验。Playground为32帧、92个查询。指标均经过Wo、在LayerScale/残差之前；局部对比共用当前稀疏QKV，累计对比分别走完整稀疏/Dense轨迹。',
        '下表 RelL2 为百分数，余弦为无量纲数；累计平方误差与Dense能量先求和再开方。','',
        '| 方法 | 场景 | 比较 | 查询 | RelL2 % | 余弦 |','|---|---|---|---|---:|---:|']
    for r in diag_summary:text.append(f"| {r['method']} | {r['scene']} | {r['comparison']} | {r['subset']} | {r['relative_l2']*100:.2f} | {r['cosine']:.5f} |")
    text += ['','24层完整Markdown表见 [layer_details.md](layer_details.md)，包括各状态比例、实际历史Key复用、普通/camera局部及累计余弦/相对L2。CSV见 `layer_states.csv` 和 `layer_similarities.csv`。',
        '逐层图：[layer_comparison.png](layer_comparison.png)。诊断不等于几何质量评测，六项质量均为N/A，原因记录在各运行quality.md。','',
        '## 六项质量验收','',
        'AUC/F1列为百分数，JSON/CSV为fraction；ACC/Comp/CD单位米，CD=(ACC+Comp)/2、非平方距离。7Scenes阈值0.05m/下采样0.0078125m；ETH3D阈值0.25m/下采样0.0390625m。',
        '均为相同checkpoint文件名、精确输入帧列表和预处理、pose+recon_unposed、相同对齐/融合/阈值。历史Dense无checkpoint内容摘要，不能宣称已核验旧服务器权重逐字节一致。SAF3R Top-K/profile=N/A，这些运行没有安装SAF3R补丁。',
        '全部候选在新几何质量评测之前按Dense内部输出诊断冻结，见quality/frozen_manifest.json；不按7Scenes测试真值再调损失。七场景未齐时不输出7Scenes宏分数，ETH3D仅3场景子集。',
        'Dense和上一修复版为既有本地测量参考；Pumpkin选择精确匹配本机773个有效RGB/pose配对所采128帧的既有Dense记录，未混用从1000帧采样的旧控制。','',
        '| 范围 | 方法 | 完成场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in macros:text.append('| '+' | '.join([r['scope'],r['method'],r['scenes']]+quality_cells(dict(values=r)))+' |')
    text += ['','| 数据集 | 场景 | 方法 | 帧 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in quality:text.append('| '+' | '.join([r['dataset'],r['scene'],r['method'],str(r['frames'])]+quality_cells(dict(values=r)))+' |')
    text += ['','## 耗时与显存','',
        'heads128同一GPU、同一套模型权重，原Shared Value/新方法交替，各预热2次+正式3次，包含完整forward的初始化、聚类、路由、历史和预测头。权重/图像加载及wrapper切换不在forward内。',
        '各配对仅在本配对内计算增幅；两种候选位于不同物理GPU，不能横比绝对秒数。两套Graph缓存共存，独立显存见quality_per_scene.csv。质量运行的单次时间只记录，不用作跨批配对比较。','',
        '| 方法 | 原Shared Value s | 新方法 s | 耗时增幅 % | 对原Shared Value加速 | 对Dense加速 |','|---|---:|---:|---:|---:|---|']
    for r in timing:text.append(f"| {r['method']} | {r['original_seconds']:.4f} | {r['candidate_seconds']:.4f} | {r['increase_fraction']*100:+.2f} | {r['speedup_vs_original_shared']:.3f}x | N/A |")
    text += ['','对原始VGGT Dense的端到端加速比N/A：按项目要求复用未变更Dense，不为计时重跑；没有同输入/同硬件/同协议的配对预热时间。Dense自身按定义1.00x。',
        '因其他作业抢占GPU而被监控中止的尝试单独保留，不进入有效质量或计时汇总；见failed_attempts.json，六项均N/A并列原因。','',
        '额外尝试了“预计算簇Value和/模长和＋稀疏历史修正”的代数等价统计优化，CPU/CUDA公式核验通过，但配对计时未观察到稳定收益，未默认启用。它改变浮点求和次序，仅完成内核/计时诊断，没有用初版质量分数冒充优化版质量评测。','',
        '## 联合输出损失补充实验','',
        '针对跨簇误差及归一化问题，已实现直接优化联合输出误差的离线坐标更新，每次最多32个状态变更；以B的可分中心化解为起点，比较γ=0/0.01/0.1的可分能量正则。它能修正统一概率反例，但真实输出仍需核验。',
        '以下只改变联合目标及其正则，保持本组控制的输入特征/历史/概率/语义相同。预计目标在8维输出投影中计算；Dense误差在完整Wo输出中测量。after/before为相同γ的目标值之比，不能跨γ当作同一种MSE比较。',
        '原型是CPU编排的诊断，尚未完成生产内核、独立轨迹及几何质量评估；六项质量及相对Dense速度均N/A，原因见joint_report各候选quality.md。','',
        '| 场景 | 候选 | 普通查询RelL2 % | Camera RelL2 % | 预计目标after/before |','|---|---|---:|---:|---:|']
    for r in joint:
        ratio=r['predicted_objective_after_over_before'];text.append(f"| {r['run']} | {r['method']} | {r['probe_non_camera']['relative_l2']*100:.3f} | {r['camera']['relative_l2']*100:.3f} | "+('N/A' if ratio is None else f'{ratio:.5f}')+' |')
    text += ['',
        '## 尚未解决的限制','',
        '新增的统一概率反例确认：即使簇内均匀概率完全正确，可分簇损失相加仍会选错mask，因为它遗漏簇间误差交叉项及整张mask的归一化分母。相同2个patch Key预算，Rep+Rep的真实误差为0，Exact+Drop的真实误差为0.8，可分能量和可分矩损失却都偏好后者。详见 `docs/shared_value_state_losses.md` 与 `separable_counterexample.json`。这证明结构缺陷存在，但未测出它对数据集退化的贡献比例。',
        '修正的是状态损失与执行语义之间的可避免不一致；未知QK的簇内均匀、head共享mask、8维输出投影和跨层误差累积仍是近似。不能仅凭History比例或局部误差下降宣称重建质量恢复，更不能宣称每层等同Dense。',
        '`rep_m5c_relaxed`配置及实现仍未出现在已同步的可见远程提交树中，本次没有伪造与它的mask重合率。',
        '所有24层原始数据、各场景六项指标、源快照、输入列表、GPU独占监控都保留于本输出目录及quality子目录。']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(macro=macros,timing=timing),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
