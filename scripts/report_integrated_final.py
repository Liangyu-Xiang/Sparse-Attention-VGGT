"""Report frozen final quality and separately audited 7Scenes speed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import FIELDS,quality_record


def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def csvsave(path,rows):
    if not rows:return
    units={'auc03':'auc03_fraction','auc30':'auc30_fraction','acc':'acc_m','comp':'comp_m','cd':'cd_m','f1':'f1_fraction'}
    rows=[{units.get(k,k):v for k,v in row.items()} for row in rows]
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def fmt(v,percentage=False):return 'N/A' if v is None else f'{v*(100 if percentage else 1):.4f}'
def metrics(row):return ' | '.join(fmt(row[k],k in ('auc03','auc30','f1')) for k in FIELDS)


def schedule_text(config):
    rates=config.get('layer_retentions') or [config['retention']]*24
    parts=[];start=0
    for end in range(1,len(rates)+1):
        if end==len(rates) or rates[end]!=rates[start]:
            parts.append(f'L{start}–{end-1}: {rates[start]*100:g}%')
            start=end
    return '; '.join(parts)


def timing_match(reference,candidate):
    for key in ('image_paths','frame_indices','shape','checkpoint_sha256'):
        if reference['input'].get(key)!=candidate['input'].get(key):return False,'Input/checkpoint mismatch: '+key
    for key in ('gpu_uuid','torch','cuda'):
        if reference['hardware'].get(key)!=candidate['hardware'].get(key):return False,'Hardware/software mismatch: '+key
    for key in ('warmup','repeats','capture'):
        if reference['timing'].get(key)!=candidate['timing'].get(key):return False,'Timing protocol mismatch: '+key
    return True,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=read(a.plan);a.output.mkdir(parents=True,exist_ok=True)
    rows=[];checks=[];scene_inputs={};scene_protocols={}
    for dataset,scenes in plan['datasets'].items():
        for scene in scenes:
            for name,spec in plan['methods'].items():
                folder=a.runs/dataset/scene/name;path=folder/'result.json'
                d=read(path) if path.exists() else (read(folder/'forward.json') if (folder/'forward.json').exists() else None)
                q=d.get('quality') if d else None
                if q is None:q=quality_record({},'Final pose/recon_unposed run has not completed')
                monitor=a.runs/(dataset+'_'+scene+'_'+name+'.monitor.json')
                m=read(monitor) if monitor.exists() else None
                if d and (m is None or m.get('same_gpu_overlap_detected') or m.get('returncode')!=0):
                    q=quality_record({},'Exclusive GPU audit pending or failed; see monitor '+str(monitor))
                row=dict(dataset=dataset,scene=scene,method=name,**q['values'],quality_complete=q['complete'],
                         quality_unavailable=json.dumps(q['unavailable'],ensure_ascii=False),frames=None,
                         full_forward_seconds=None,peak_allocated_gib=None,mean_unique_key_density=None,
                         speedup_vs_dense=None,time_ratio_vs_saf3r=None,
                         speed_unavailable='No current paired Dense/SAF3R timing; historical timings are not speed baselines',
                         source=str(path))
                if d:
                    if spec.get('config'):
                        config_path=Path(spec['config'])
                        if hashlib.sha256(config_path.read_bytes()).hexdigest()!=spec['config_sha256']:
                            raise ValueError('Frozen config file changed: '+str(config_path))
                        if d.get('config')!=read(config_path):
                            raise ValueError('Actual run differs from frozen config: '+str(path))
                    if plan.get('checkpoint_sha256') and d['input'].get('checkpoint_sha256')!=plan['checkpoint_sha256']:
                        raise ValueError('Frozen checkpoint contents changed: '+str(path))
                    key=(dataset,scene)
                    if key in scene_inputs and scene_inputs[key]!=d['input']:
                        raise ValueError('Final methods use different exact inputs: '+str(path))
                    scene_inputs[key]=d['input']
                    if key in scene_protocols and scene_protocols[key]!=d.get('quality_protocol'):
                        raise ValueError('Final methods use different quality protocols: '+str(path))
                    scene_protocols[key]=d.get('quality_protocol')
                    for rel,digest in plan.get('model_source_sha256',{}).items():
                        if rel in d.get('source_sha256',{}) and d['source_sha256'][rel]!=digest:
                            raise ValueError('Frozen source mismatch: '+rel)
                    row.update(frames=d['frames'],full_forward_seconds=d['timing']['median_seconds'],
                               peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),
                               gpu_uuid=d['hardware'].get('gpu_uuid'),warmup=d['timing']['warmup'],repeats=d['timing']['repeats'])
                    if q['complete'] and abs(row['cd']-(row['acc']+row['comp'])/2)>1e-8:raise ValueError('Invalid CD definition')
                    layers=folder/'layers.json'
                    if layers.exists():
                        ls=read(layers)
                        if [r['layer'] for r in ls]!=list(range(24)):raise ValueError('Missing global layers')
                        row['mean_unique_key_density']=statistics.mean(r['actual_unique_density'] for r in ls)
                        row['equivalent_sparsity']=1-row['mean_unique_key_density']
                        n=len(d['input']['image_paths'])*(d['input']['shape'][-2]//14*d['input']['shape'][-1]//14+5)
                        row['mean_executed_main_qk_density']=sum(r['direct_logical_main_qk_per_head'] for r in ls)/(24*n*n)
                    if dataset=='7scenes' and m and not m['same_gpu_overlap_detected']:
                        matched=[]
                        for control in ('dense','saf3r'):
                            cp=a.runs/dataset/scene/control/'timing_result.json'
                            cm=a.runs/(dataset+'_'+scene+'_'+control+'.monitor.json')
                            if not cp.exists() or not cm.exists():continue
                            c=read(cp);audit=read(cm);valid,reason=timing_match(c,d)
                            valid=valid and not audit['same_gpu_overlap_detected'] and audit['returncode']==0
                            checks.append(dict(dataset=dataset,scene=scene,method=name,control=control,valid=valid,reason=reason))
                            if valid:
                                if control=='dense':row['speedup_vs_dense']=c['timing']['median_seconds']/row['full_forward_seconds']
                                else:row['time_ratio_vs_saf3r']=row['full_forward_seconds']/c['timing']['median_seconds']
                                matched.append(control)
                        if len(matched)==2:row['speed_unavailable']=None
                rows.append(row)
    macros=[]
    for dataset,scenes in plan['datasets'].items():
        for name in plan['methods']:
            rs=[r for r in rows if r['dataset']==dataset and r['method']==name]
            complete=len(rs)==len(scenes) and all(r['quality_complete'] for r in rs)
            macro=dict(dataset=dataset,method=name,completed_scenes=sum(r['quality_complete'] for r in rs),
                       expected_scenes=len(scenes),quality_complete=complete,
                       **{k:statistics.mean(r[k] for r in rs) if complete else None for k in FIELDS})
            if dataset=='7scenes':
                timed=all(r['full_forward_seconds'] is not None for r in rs)
                macro['mean_warmed_median_seconds']=statistics.mean(r['full_forward_seconds'] for r in rs) if timed else None
                macro['peak_allocated_gib']=max(r['peak_allocated_gib'] for r in rs) if timed else None
                macro['mean_unique_key_density']=statistics.mean(r['mean_unique_key_density'] for r in rs) if all(r['mean_unique_key_density'] is not None for r in rs) else None
                macro['equivalent_sparsity']=None if macro['mean_unique_key_density'] is None else 1-macro['mean_unique_key_density']
                # Ratio of equal-scene mean seconds, not the mean of per-scene speedups.
                for key in ('speedup_vs_dense','time_ratio_vs_saf3r'):
                    if timed and all(r[key] is not None for r in rs):
                        control=statistics.mean(r['full_forward_seconds']*(r[key] if key=='speedup_vs_dense' else 1/r[key]) for r in rs)
                        macro[key]=control/macro['mean_warmed_median_seconds'] if key=='speedup_vs_dense' else macro['mean_warmed_median_seconds']/control
                    else:macro[key]=None
            macros.append(macro)
    result=dict(complete=all(r['quality_complete'] for r in macros),per_scene=rows,macro=macros,paired_timing_checks=checks,
                units=dict(auc03='fraction',auc30='fraction',f1='fraction',acc='m',comp='m',cd='m'),
                cd_definition='(ACC+Comp)/2 unsquared nearest-neighbour distances',
                selection='Final configs frozen before geometric quality; RGB-only calibration proxy is not a quality guarantee',
                timing_scope_correction='Complete synchronized wall timing includes support/history validation synchronizations already present in the method. The legacy raw scope string no per-layer sync is not an accurate description of those internal checks; no validation cost was removed.')
    save(a.output/'summary.json',result);csvsave(a.output/'per_scene.csv',rows);csvsave(a.output/'macro.csv',macros)
    config_lines=[]
    for name,spec in plan['methods'].items():
        if spec.get('config') and Path(spec['config']).exists():
            c=read(Path(spec['config']))
            config_lines.append('| '+name+' | '+schedule_text(c)+' | '+str(c.get('cluster_fraction'))+' × patch数 | '+str(c['local_iterations'])+' | '+c.get('shared_value_weighting','unit')+' |')
    lines=['最终配置在 pose/recon_unposed 测试前冻结；没有依据测试真值调参。AUC/F1 表中为百分数，原始 JSON/CSV 为 fraction；距离为米，CD=(ACC+Comp)/2，未平方。','',
           '两个配置是已完整评测的实验候选，尚未证明同时保精度并达到SAF3R耗时目标，因此没有升级为通用默认。原工作流source32／BQ64入口已保留；main／fast的冻结配置和原始测量不变，详见 adoption_status.json。','',
           '7Scenes 覆盖七场景、每场景128帧；ETH3D 使用仓库现行11场景（不含 terrace、meadow），最多128帧，实际帧数见逐场景表。数据集均为场景等权宏平均；未齐全时六项宏指标全部标N/A。校准使用 heads/chess 同序列中与最终帧无重叠的RGB，不读取几何真值，不声称独立训练集验证。','',
           'main 为预先选定的主方案，fast 为速度候选，命名不依赖最终真值排名。两者使用 Query group256、连续编号source32无持久锚点、64行Query tile／128列Key tile、4 warps／1 stage、16轮预算搜索、25%残差候选、history interval256、稳定人口分桶Graph缓存（8项／2048MiB），保留全部Special Key。逐层预算通过实际执行路径生效，预算变化时使用连续且宽度准确的双缓存历史。','',
           '| 方法 | 各层总Key预算上限（含Special） | 固定初始簇数 | 聚类更新次数 | 状态执行权重 |','|---|---|---|---:|---|',
           *config_lines,'',
           '每帧Camera Query另做完整Key attention并覆盖该Query输出；主稀疏observer仍生成history，两部分工作及重复QK均计时和记录。History∪Rep中心化损失路径作为可选研究实现保留，本次主方案未默认采用，不能把两种语义混称。实际密度受离散预算未用满影响，不能用配置上限代替；冻结前密度筛选边界修正已明确记录在 freeze_density_eligibility_correction.json。','',
           '聚类/归约含原有非确定浮点操作。质量使用最后一次正常前向输出，只做一次几何评测，不把微小单次差异声明为稳定提升；不将局部校准误差当作几何质量。','',
           '| 数据集 | 方法 | 完成场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in macros:lines.append(f"| {r['dataset']} | {r['method']} | {r['completed_scenes']}/{r['expected_scenes']} | "+metrics(r)+' |')
    lines+=['','速度只比较7Scenes。每场景1次完整预热＋3次完整前向，取中位数后按场景等权平均；包括本次输入所需初始化、聚类、分配、Graph、历史、稀疏/稠密Query输出修正和所有输出头。模型权重加载、图像解码及后续几何评估在计时外。峰值显存为所有热运行中的最大 allocated。','',
            '方法保留支持集/历史计数校验引起的层间同步，其耗时包含在完整前向。原始 scope 字符串中的 no per-layer sync 是旧runner措辞，不作为本批事实；没有删除校验来制造收益。','',
            '| 方法 | 完整前向均值 s | 峰值 allocated GiB | 实际平均Key密度 % | 等效稀疏率 % | vs Dense 加速 | 耗时/SAF3R |','|---|---:|---:|---:|---:|---:|---:|']
    for r in macros:
        if r['dataset']!='7scenes':continue
        lines.append('| '+r['method']+' | '+' | '.join([fmt(r.get('mean_warmed_median_seconds')),fmt(r.get('peak_allocated_gib')),fmt(r.get('mean_unique_key_density'),True),fmt(r.get('equivalent_sparsity'),True),fmt(r.get('speedup_vs_dense')),fmt(r.get('time_ratio_vs_saf3r'))])+' |')
    lines+=['','Dense 加速按 Dense热运行中位秒数/方法中位秒数定义，Dense自身为1.00x，低于1x表示更慢。耗时/SAF3R ≤1.10才达到10%目标。缺少本批同卡/同输入/同协议计时则标N/A，不以历史或H100时间代替。本批若未获得对照重测授权，不声明已达到SAF3R目标。ETH3D只输出精度；单次前向运行记录不是预热速度比较。','',
            '7Scenes F1阈值0.05m、下采样0.0078125m；ETH3D阈值0.25m、下采样0.0390625m。对齐、TSDF融合、有效深度掩码和点采样保持仓库协议；详见每次 result.json 的 quality_protocol 和冻结源快照。','',
            'SAF3R参照配置为 lazy 首个对应头 Top-4，profile为 eth3d-train-fltr-calib_cmpmse.json；不是论文DINO Top-K，不是Top-128。Shared Value变体没有安装该SAF3R补丁，具体预算、聚类及执行配置见冻结配置。','',
            '| 数据集 | 场景 | 方法 | 帧数 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | 完整前向 s | vs Dense |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['dataset']} | {r['scene']} | {r['method']} | {r['frames']} | "+metrics(r)+f" | {fmt(r['full_forward_seconds'])} | {fmt(r['speedup_vs_dense'])} |")
    lines+=['','所有原始 pose/recon_unposed 指标、输入帧列表、checkpoint内容SHA-256、源码哈希与快照、正常/首次运行时间、逐层密度/Graph统计、GPU独占监控保存在 runs 中。统计诊断与正常测速分开，不把局部误差当作六项精度。','']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=result['complete'],macro=macros),ensure_ascii=False))


if __name__=='__main__':main()
