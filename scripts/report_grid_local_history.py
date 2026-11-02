"""Combine complete quality, paired timings and numerical evidence; no score tuning."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
FIELDS=['auc03','auc30','acc','comp','cd','f1']


def read(p):return json.loads(p.read_text())
def write_csv(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def normalize(x):
    if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
    if isinstance(x,list):return [normalize(v) for v in x]
    if isinstance(x,str) and x.startswith('/'):return str(Path(x).resolve())
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);a=p.parse_args();base=a.results_root;out=base/'grid_local_final_20261102';out.mkdir(exist_ok=True)
    manifest=read(base/'headmean_single_qk_quality_20261102/manifest.json')
    new_manifest=read(base/'grid_local_quality_20261102/manifest.json');source=Path(new_manifest['source_root'])
    unchanged=all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in new_manifest['source_sha256'].items())
    rows=[];audits=[];missing=[];macro=[]
    for ds,scenes in manifest['datasets'].items():
        for scene in scenes:
            oldpath=base/'headmean_single_qk_quality_20261102'/ds/scene/'headmean_single_qk/result.json';old=read(oldpath)
            candidates=[base/n/ds/scene/'grid_local/result.json' for n in ['grid_local_quality_retry_20261102','grid_local_quality_20261102']]
            current=next((f for f in candidates if f.exists() and read(f)['quality']['complete']),None)
            refs=manifest['references'][ds][scene]
            data={'reference':(oldpath,old),'dense':(Path(refs['dense']),read(Path(refs['dense']))),'saf3r':(Path(refs['saf3r']),read(Path(refs['saf3r'])))}
            if current:
                d=read(current);data['grid_local']=(current,d);monitor=read(current.parent.parent/'lease_monitor.json');layers=read(current.parent/'layers.json')
                checks=dict(source_unchanged=unchanged,input_identical=normalize(d['input'])==normalize(old['input']),protocol_identical=d['quality_protocol']==old['quality_protocol'],quality_complete=d['quality']['complete'],gpu_exclusive=not monitor['same_gpu_overlap_detected'] and monitor.get('strict_contexts') and monitor['returncode']==0,all24=len(layers)==24,budget_20=all(l['actual_unique_density']<=.2+1e-7 for l in layers),no_extra_qk=all(l['evidence_extra_qk_per_head']==0 for l in layers),no_cpu_cache=all(l['history_cpu_bytes']==0 for l in layers),config_frozen=d['config']==new_manifest['config'])
                checks['model_identical']=normalize(d['model_config'])==normalize(old['model_config'])
                checks['frame_count_identical']=d['frames']==old['frames'] and (ds!='7scenes' or d['frames']==128)
                audits.append(dict(dataset=ds,scene=scene,checks=checks,passed=all(checks.values())))
            else:missing.append(dict(dataset=ds,scene=scene))
            for method,(path,d) in data.items():
                v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-6
                samples=d['timing']['samples']
                rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],**{k:v[k] for k in FIELDS},seconds=d['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in samples),reused=method!='grid_local',source=str(path),dense_speedup=1 if method=='dense' else None,dense_speedup_reason='Dense reference' if method=='dense' else 'No contemporaneous paired Dense; unchanged control not rerun'))
        for method in ['dense','saf3r','reference','grid_local']:
            rr=[r for r in rows if r['dataset']==ds and r['method']==method]
            if len(rr)==len(scenes):macro.append(dict(dataset=ds,method=method,scenes=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in FIELDS},dense_speedup=1 if method=='dense' else None))
    write_csv(out/'per_scene.csv',rows);write_csv(out/'macro.csv',macro)
    paired=read(base/'grid_local_paired_heads128_20261102/summary.json');monitor=read(base/'grid_local_paired_heads128_20261102_monitor.json')
    assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected'] and monitor['strict_contexts']
    times=[]
    for method,v in paired['profiles'].items():
        groups=v['groups'];inc=lambda n:groups[n]['inclusive_seconds'];exc=lambda n:groups[n]['exclusive_seconds']
        attn=inc('execution_and_evidence');producer=inc('m5c_evidence_and_host_copy')
        core=inc('attention_core_fused_grid')+inc('attention_grid_normalize_reduce')
        qkv=inc('qkv_rope_prepare');proj=inc('output_projection')+inc('output_dropout')
        times.append(dict(method=method,full_forward_warmed_median=paired['median_seconds'][method],profile_full_forward=v['wall_seconds'],global_attention=inc('global_attention'),step9_attention_and_layout=attn,step10_producer=producer,step9_plus_10=attn+producer,actual_attention_with_grid=core,mask_and_history=inc('global_attention')-core-qkv-proj,peak_allocated_gib=max(s['peak_allocated_gib'] for s in paired['samples'][method]),global_selected_logsumexp=inc('kernel_step1_selected_logsumexp') if method=='reference' else 0.,grid_calibration_or_local_lse=inc('kernel_step2_grid_exp_segment_sum') if method=='reference' else inc('kernel_local_normalizers'),probability_headmean_edge_encoding=inc('kernel_step34_headmean_fp16_u14_store') if method=='reference' else inc('kernel_local_store_local'),uniform_probability_headmean=0. if method=='reference' else inc('kernel_local_uniform'),legacy_eps_small_grid_reduction=0. if method=='reference' else inc('kernel_local_compat')))
    write_csv(out/'paired_timing.csv',times)
    numerical=read(base/'grid_local_validation_final_20261102/validation.json');real=read(base/'grid_local_real_window_v2_20261102/result.json');kernel=read(base/'grid_local_kernel_audit_20261102/result.json')
    stability=read(base/'grid_local_rollout_stability_20261102/rollout_comparison.json')
    stability_summary={m:dict(changed_groups=sum(r['support_fingerprint_changed_groups'] for r in rr),total_groups=sum(r['groups'] for r in rr),max_sample_attention_relative_l2=max(r['sample_attention_relative_l2'] for r in rr)) for m,rr in stability['comparisons'].items()}
    deltas=[]
    for ds in manifest['datasets']:
        x=next((r for r in macro if r['dataset']==ds and r['method']=='reference'),None)
        y=next((r for r in macro if r['dataset']==ds and r['method']=='grid_local'),None)
        if x and y:deltas.append(dict(dataset=ds,**{k:(y[k]/x[k]-1)*100 for k in FIELDS}))
    complete=not missing and len(audits)==18 and all(r['passed'] for r in audits)
    summary=dict(complete=complete,macro=macro,relative_quality_change_percent=deltas,rollout_stability=stability_summary,audits=audits,missing=missing,source_unchanged=unchanged,paired_timing=times,numerical_complete=numerical['complete'],quality_controls_executed=0,timing_reference_rerun_explicitly_requested=True,dense_speedup=None,dense_reason='No new Dense timing; not inferred from history-method reference',optimization_status='Speed improvement verified, but 7Scenes has >1% relative degradation in several metrics; do not claim preserved-quality success; default remains reference')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Grid局部归一化历史证据：技术与实验报告','','远程基点：908ae1fc43f1b54874e7ae5ae2c4cd3514655bb6。独立分支codex/grid-local-history-20261102。保留默认reference；新配置configs/evaluation/reuse_grid_local_history.json。', '',
           '## 实际调用链与数学定义','','ReuseResidualExploration.forward → 原observe（逐头Coarse/Special归一化不变）→ reuse_single_qk_grid.weighted_attention（正式AV及原始Probe QK导出不变）→ reuse_grid_local_history.save_evidence → Grid局部LogSumExp → 条件概率/Head-mean/FP16-u14缓存 → 原build_prior。', '',
           'c=exp(raw-logD)，u=exp(grid_logit-logD)。Head在局部归一化后才消去，不平均Q/K或Logit。Special始终参与正式Attention。第10步没有QK，也没有调用selected_probe_logits。复用Attention已经计算的稳定排序与offsets，支持任意合法逻辑Key顺序，没有另付排序成本。', '',
           'EPS兼容：原corrected_grid.clamp_min(1e-30)在极小Grid质量时阻止全局分母完全消去。为保持旧定义，额外用小型Grid分母和已有Special Logit计算logZ，再乘min(1,exp(logD-logZ-logEPS))。不构造逐边Dense概率，也不扫描RAW做全局LogSumExp。未删除observe或build_prior中的Special依赖。', '',
           '## Tensor与内核审计','','典型RAW：[64,16,4,26009] FP32，426131456字节（406.39MiB）；Grid：[64,16,4,512] FP32；Special：[64,16,4,640] FP32。原observe的Grid/Special为非连续head-major view；新路径显式连续化，这些复制包含在计时中。RAW保持逻辑顺序，order/offset只用于局部分母的Gather。详细shape/stride见kernel audit原始JSON。', '',
           '|实现|GPU Kernel事件数/窗口|临时峰值增量MiB|边缓存MiB/64组|','|---|---:|---:|---:|']
    for m,k in kernel['results'].items():lines.append(f"|{m}|{k['gpu_kernel_count']}|{k['temporary_peak_delta_bytes']/2**20:.3f}|{k['cache_bytes']/2**20:.3f}|")
    lines+=['','Kernel审计使用实际heads128尺寸的合成合法支持集；内核活跃时间与CUDA事件区间分开保存。事件数包含所有CUDA kernel（含零填充），CPU launch API计数可能因Memset/图层实现不同而不同。不是端到端计时。', '',
            '## 正确性','','20组校验覆盖1/3/16 heads，FP32、packed与普通FP16缓存，极端Logit、无已选Key、全选、仅一个成员的Grid、Padding、末尾Probe重复权重及非连续观测Tensor。另有真实BF16 Attention执行测试，确认新增布局元数据不改变Attention输出，Special和无补偿原始Probe Logit正确，覆盖末尾15行Query。逐边条件概率、未观测概率、每Grid质量守恒、下一层prior及pi均通过阈值。FP32算法误差与缓存量化误差单独保存。极端合成数据中全选Grid的未观测概率无实际成员对应，数值可能很大，此时必须同时查看相对误差，不能只看最大绝对差。', '',
            '真实64-group窗口固定相同RAW、Mask、QKV和Grid：']
    for r in real['rows']:lines.append(f"- L{r['layer']}：旧生产{r['producer_median_ms']['reference']:.4f}ms，新生产{r['producer_median_ms']['local']:.4f}ms；缓存relative L2={r['cache_rel_l2']:.3g}，uniform最大绝对差={r['uniform_max_abs']:.3g}。")
    lines+=['','完整24层支持集指纹、边数变化及抽样Attention输出差异见rollout_comparison.json。不声称Mask逐位一致：微小量化差异和原聚类的浮点归约可能通过离散决策传播；原实现自身重复稳定性单独诊断。没有把输出变化隐藏为零误差。', '',
            '## 独占同卡配对时间','','heads128、相同checkpoint/输入/环境；严格context监控，其他进程出现则终止。一个方法预热一次，三个方法交替三次取中位数；分段另跑一次插桩前向。插桩有Host发射间隙，不能把GPU Event区间说成纯内核活跃时间。', '',
            '|实现|完整Forward中位数s|Global Attention s|第9步s|第10步s|第9+10步s|构建及维护s|峰值allocated GiB|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in times:lines.append('|'+t['method']+'|'+'|'.join(f'{t[k]:.6f}' for k in ['full_forward_warmed_median','global_attention','step9_attention_and_layout','step10_producer','step9_plus_10','mask_and_history','peak_allocated_gib'])+'|')
    lines+=['','内核细分见paired_timing.csv：条件概率、Head-mean和边缓存编码融合，不能分别独立计时或重复累加。旧全局RAW LogSumExp已删除；新局部normalizer与小型EPS兼容归约分别列出。', '',
            '## 两数据集质量（等场景宏平均）','','仅运行grid_local packed的质量；原reference、Dense和SAF3R复用完整历史结果。质量运行1预热+1次输出，不以该次耗时进行速度验收；上表配对计时独立采用3次中位数。未基于测试质量调参。', '',
            '|数据集|方法|场景数|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Dense配对加速|','|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in macro:lines.append('|'+ '|'.join([r['dataset'],r['method'],str(r['scenes'])]+[f"{r[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS]+['1.00x' if r['method']=='dense' else 'N/A'])+'|')
    lines+=['','AUC/F1为百分比，距离为米，CD=(ACC+Comp)/2（未平方）。7Scenes阈值0.05m、降采样0.0078125m；ETH3D阈值0.25m、降采样0.0390625m。SAF3R为本地lazy first-correspondence Top-4，非论文DINO Top-K；完整profile见引用result.json。7Scenes全7场景各128帧，ETH3D全11场景实际14–76帧。Dense相对加速N/A：未获要求重跑Dense，没有本轮配对分母。', '',
            f'完整审计通过：{complete}；缺失场景：{missing}。', '',
            '## 验收结论与数值稳定性','','速度收益成立，但不能宣布保精度优化成功。7Scenes的ACC、Comp、CD和F1均有超过1%的相对退化，AUC@3也下降超过1%；ETH3D六项均改善。保留本轮结果，不据此调整参数或挑选重复质量结果。默认reference不变。', '',
            '|数据集|AUC@3 相对变化%|AUC@30 相对变化%|ACC 相对变化%|Comp 相对变化%|CD 相对变化%|F1 相对变化%|','|---|---:|---:|---:|---:|---:|---:|']
    for r in deltas:lines.append('|'+r['dataset']+'|'+'|'.join(f'{r[k]:+.6f}' for k in FIELDS)+'|')
    lines+=['','原方法重复稳定性诊断（固定输入，未重跑GT质量）：']
    for m,s in stability_summary.items():lines.append(f"- {m}：支持集指纹变化{s['changed_groups']}/{s['total_groups']}组，24层首32个Query抽样Attention输出最大relative L2={s['max_sample_attention_relative_l2']:.6f}。")
    lines+=['','原实现自身重复推理也存在离散Mask漂移，因此不能把历史批次质量差异全部归因于局部归一化；本轮也不据此豁免质量风险。新旧固定输入概率对照与完整推理传播误差是两个不同层次的验证。普通FP16的L0差异也不能归因于尚未消费的历史缓存。', '',
            '## 保留的限制与失败记录','','最初真实窗口测试因忽略Grid/Special非连续stride失败，结果保留在grid_local_real_window_20261102，已由v2回归覆盖；该失败版不计入性能结论。首批质量作业的部分归档阶段因远程分支缺少baseline_source_hashes引用的旧配置而失败；按原SHA补齐依赖后，仅重跑失败场景，原失败记录保留。初次合成数值测试不足以替代真实布局验证。', '',
            '本次保留406MiB RAW导出与两遍读取，尚未把局部分母直接融合进Attention。没有为形式上的单遍融合改变Attention的Key顺序、BF16累加或算法含义。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
