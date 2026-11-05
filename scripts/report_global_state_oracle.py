"""Report global-budget oracle key sets, state reachability and the layout fix."""
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'outputs/global_state_loss_oracle_20261105'
SCENES=('pipes','playground','courtyard')
FIELDS=('auc03','auc30','acc','comp','cd','f1')


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aggregate={};example_rows=[]
    methods=['native','legacy_true_group_probability','joint_same_features_estimated_probability',
             'joint_full_wo_estimated_probability','state_space_oracle','free_key_oracle']
    for scene in SCENES:
        run=BASE/(scene+'_oracle');cases=json.loads((run/'cases.json').read_text())
        assert len(cases)==8
        by_case={(c['layer'],c['group']):c for c in cases}
        assert all(c['native_replay_support_difference']==0 for c in cases)
        for c in cases:
            for name in ('state_space_oracle','free_key_oracle'):
                assert c['variants'][name]['relative_l2']<=c['native_relative_l2']+1e-6
                assert c['variants'][name]['keys']<=c['budget']
            assert c['variants']['free_key_oracle']['relative_l2']<=c['variants']['state_space_oracle']['relative_l2']+1e-6
        groups={key:[] for key in by_case}
        with gzip.open(run/'clusters.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);groups[(row['layer'],row['group'])].append(row)
        counts=Counter();reasons=Counter();cover_ratios=[];examples=[]
        for key,rows in groups.items():
            c=by_case[key];assert len(rows)==c['clusters']
            errors=np.array([[math.inf if x is None else x for x in row['legacy_losses']] for row in rows])
            costs=np.array([row['costs'] for row in rows])
            # Reconstruct the production 16-step global budget multiplier, with
            # the same per-cluster constant subtraction used by its objective.
            objective=errors-errors[:,1:2]
            lo=np.float32(0);hi=np.float32(errors[:,0].max()+1e-12)
            for _ in range(16):
                mid=np.float32((lo+hi)*.5)
                chosen=(objective.astype(np.float32)+mid*costs.astype(np.float32)).argmin(1)
                total=costs[np.arange(len(rows)),chosen].sum()
                if total>c['budget']:lo=mid
                else:hi=mid
            counts['cluster_cases']+=len(rows);counts['oracle_keys']+=c['variants']['free_key_oracle']['keys']
            counts['unreachable_keys']+=c['oracle_keys_unreachable_by_any_existing_state']
            counts['keys_in_unexpressible_supports']+=c['oracle_keys_in_unexpressible_cluster_supports']
            relaxed_cover=0
            for row,e,cost in zip(rows,errors,costs):
                desired=set(row['oracle_keys']);native=set(row['native_keys']);hist=set(row['eligible_history_keys'])
                counts['native_oracle_key_intersection']+=len(desired&native)
                if desired:
                    counts['oracle_active_clusters']+=1
                    counts['unexpressible_active_clusters']+=not row['oracle_support_expressible']
                    cheapest=row['population']
                    if desired<=hist:cheapest=min(cheapest,len(hist))
                    if desired=={row['rep']}:cheapest=1
                    relaxed_cover+=cheapest
                current=row['native_state'];target=row['state_oracle_state']
                if current==target:reason='same_state'
                elif e[target]>e[current]+1e-8:reason='proxy_prefers_native_even_before_budget_price'
                elif cost[target]>cost[current] and e[target]+hi*cost[target]>=e[current]+hi*cost[current]-1e-8:
                    reason='lower_proxy_error_but_priced_out_by_global_budget'
                else:reason='ties_or_joint_budget_reallocation'
                reasons[reason]+=1
                if desired and not row['oracle_support_expressible'] and len(examples)<3 and row['population']>=4:
                    examples.append(row|dict(scene=scene,reason='free oracle key set cannot be expressed by one existing state',global_budget=c['budget'],case_native_l2=c['native_relative_l2'],case_oracle_l2=c['variants']['free_key_oracle']['relative_l2']))
            cover_ratios.append(relaxed_cover/c['budget'])
        means={'native':sum(c['native_relative_l2'] for c in cases)/8}
        means.update({name:sum(c['variants'][name]['relative_l2'] for c in cases)/8 for name in methods[1:]})
        aggregate[scene]=dict(cases=8,counts=dict(counts),state_differences=dict(reasons),mean_relative_l2=means,
            oracle_keys_unreachable_fraction=counts['unreachable_keys']/counts['oracle_keys'],
            native_oracle_key_recall=counts['native_oracle_key_intersection']/counts['oracle_keys'],
            mean_minimum_cover_budget_ratio_if_exact_allowed_everywhere=sum(cover_ratios)/8)
        example_rows+=examples
    (BASE/'oracle_summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    (BASE/'cluster_examples.json').write_text(json.dumps(example_rows,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    labels=['Native','True P, old loss','Joint old features','Joint full Wo','State oracle','Free-key oracle']
    for ax,scene in zip(axes,SCENES):
        means=aggregate[scene]['mean_relative_l2']
        ax.bar(range(len(methods)),[means[name]*100 for name in methods])
        ax.set_xticks(range(len(methods)),labels,rotation=65,ha='right',fontsize=7)
        ax.set_title(scene);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean frozen-input post-Wo relative L2 (%)')
    fig.tight_layout();fig.savefig(BASE/'global_oracle_output_errors.png',dpi=170);plt.close(fig)
    lines=['# 全局Key预算下的Oracle与状态损失诊断','',
      '每个case联合优化整个Query组的mask：所有cluster共同竞争同一个全局patch Key预算。',
      '不预设每簇保留几个Key。保护Key始终保留；使用同一cluster_population执行算子。',
      '3个ETH3D场景，每场景Global层6/11/13/19、2个Query组，共8个case。组内全部稀疏执行Query、全部head和完整Wo参与真实输出误差；排除本来就由Dense覆盖的Camera Query。',
      'Oracle是真实Dense QK上的近似组合搜索：40步straight-through Adam，状态空间2个起点，自由Key空间3个起点，报告所有硬mask中的最优值；不保证组合全局最优，不进入推理。', '',
      '## 确认并修复的实现错误','',
      '`residual=sparse.mm(...).T`生成列连续张量，`zeros_like(residual)`保留该布局；原state-loss Triton按`group*F+cluster`读取allowed，误读了Exact候选。',
      '旧pipes第6层group12：129个逻辑上禁止的整簇支持实际被执行；冻结输入的eager和Graph完全一致，排除Graph专有缓存错误。386个被禁止的Exact状态被赋有限损失。',
      '修复：候选掩码直接按行连续分配（没有额外copy），同时状态损失kernel读取真实stride；新的规则测试覆盖普通/bounded两种kernel和转置mask。',
      '此前改变簇数等参数也会改变错误读取的映射，故那些收益不能再单独当作簇数压缩是普适根因的证据。', '',
      '## 状态损失定义本身也有不一致','',
      '令History恰为{Rep}。当前cluster_population执行中，两状态都保留同一Key、权重population、成本1，实际输出完全相同，但损失分别为`sum p_j ||V_j-V_rep||²`和`sum_{j不在History} p_j ||V_j||²`，通常不相等。',
      '真实冻结特征中，三个场景共有29,833个History={Rep}的cluster-case，其中29,504个的两项损失相差超过1e-8。这是定义与执行算子不一致的证据，不是这些簇各自造成多少几何退化的测量。',
      '精确概率反例：一个簇的两个Value都是1，一个Special Value是0，概率为(1/11,9/11,1/11)。Rep保留第一个成员且权重2，真实输出为2/3，Dense为10/11，误差非零；但Rep失真损失为0。即使给损失真实概率，也不能从Value相同推出softmax输出无误差，因为代表点放大后的质量2*p_rep不等于簇质量。',
      '均匀概率反例还显示，同支持同权重且都精确恢复Dense时，Rep损失为0而History损失为1/3。原始反例及真实特征统计见`loss_definition_contracts/result.json`。', '',
      '## 同预算全局输出误差','',
      '下表为8个冻结case的等权均值，单位为relative L2百分数，不是几何精度或完整独立Dense轨迹误差。', '',
      '| Scene | 当前损失 | 真实组平均概率+原损失 | 同特征联合损失+估计概率 | 完整Wo联合损失+估计概率 | 四状态oracle | 自由Key oracle |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for scene,item in aggregate.items():lines.append('| '+scene+' | '+' | '.join(f"{item['mean_relative_l2'][m]*100:.4f}" for m in methods)+' |')
    lines+=['','![Global oracle output errors](global_oracle_output_errors.png)','',
      '真实概率替换控制冻结了原有cluster、历史资格、候选限制、损失特征、执行权重和预算。联合同特征控制还保持head-mean特征，但改变目标为带全局softmax分母的联合输出误差。',
      '后两种oracle使用真实当前QK信息；改进幅度只能说明现有估计/状态还存在空间，不能作为可部署方法的精度。', '',
      '## 各cluster整体需要的Key与状态为什么不同','',
      '| Scene | Oracle Key中任何现有状态都无法选到 | Native对oracle Key的召回 | 即使放开全部Exact，覆盖oracle集合的最小Key成本/预算 |',
      '|---|---:|---:|---:|']
    for scene,item in aggregate.items():lines.append(f"| {scene} | {item['oracle_keys_unreachable_fraction']*100:.2f}% | {item['native_oracle_key_recall']*100:.2f}% | {item['mean_minimum_cover_budget_ratio_if_exact_allowed_everywhere']:.3f}x |")
    lines+=['',
      '这里“无法选到”指该Key既不是当前Rep，也不在合格History内，且所在簇未开放Exact。即使把Exact对所有簇开放，要覆盖整个oracle集合也常需超过预算：四状态不能表达任意多个成员的组合。',
      '各簇完整native/oracle Key列表、数量、History资格、Rep、四状态损失与成本保存于每场景oracle目录的`clusters.jsonl.gz`。编号是patch域内索引；映射到全局token使用capture中的patches。具体例子见`cluster_examples.json`。',
      '状态oracle与native的逐簇差异分类见`oracle_summary.json`：区分代理损失本身更偏好native、较低代理损失被预算价格压制、以及联合重分配/平局。不能把联合更优mask的每个单簇选择都说成独立正确状态。', '',
      '### 一个同预算联合搜索的具体例子','',
      'pipes层11、Query组12的簇0：population=11，当前Rep=153；native保留{153}，全局oracle为该簇分配{0,85,224}，合格History={0,3,153,224}，Exact未开放。',
      'Key85无法由任何现有可用状态选入。Rep损失0.000741338、成本1；History损失0.001096850、成本4。损失和成本同时偏向Rep，但真实全局输出目标会在联合重分配中改选History；自由Key搜索进一步摆脱了History多余成员的限制。',
      '同一全局oracle又把簇116从native的12个Exact成员变为0个Key：各簇配额是整体输出误差与全局预算共同决定的，不是每簇保留一个Key。', '',
      '## 布局修复后的六指标（3/11场景）','',
      'AUC/F1为百分数，距离为米，CD=(ACC+Comp)/2非平方；固定同权重、帧表、预处理和评测协议。', '',
      '| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    quality_rows=[]
    for scene in SCENES:
        for name,path in [('before',ROOT/'outputs/reuse_layer_comparison_20261105'/scene/'main_independent/saved_quality/result.json'),
                          ('layout_fixed',BASE/(scene+'_fixed_capture')/'saved_quality/result.json')]:
            q=json.loads(path.read_text())['quality']['values']
            quality_rows.append(dict(scene=scene,method=name,values=q))
            cells=[f"{q[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS]
            lines.append('| '+scene+' | '+name+' | '+' | '.join(cells)+' |')
    for name in ('before','layout_fixed'):
        rows=[r for r in quality_rows if r['method']==name]
        values={key:sum(r['values'][key] for r in rows)/3 for key in FIELDS}
        cells=[f"{values[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS]
        lines.append('| 3/11场景等权均值（非全ETH3D） | '+name+' | '+' | '.join(cells)+' |')
    lines+=['','所有oracle/capture/kernel诊断的六指标单独标N/A并附原因；完整几何结果来自saved_quality。',
      'ETH3D没有配对Dense暖机计时，加速为N/A。7Scenes heads128配对计时另存speed_heads128；不能称为全7场景速度。']
    speed_root=BASE/'speed_heads128'
    speed_names=('dense','saf3r','before_layout_fix_v2','layout_fixed')
    if all((speed_root/name/'timing_result.json').exists() for name in speed_names):
        from scripts.quality_metrics import quality_markdown
        data={name:json.loads((speed_root/name/'timing_result.json').read_text()) for name in speed_names}
        dense=data['dense'];speed=[]
        lines+=['','## heads128同批完整前向速度','',
          'GPU6独占；1次暖机+3次计时中位数；BF16，TF32关闭；包括reset/聚类/路由/历史/重排，不含模型加载和图像加载。',
          'Dense和原SAF3R只补计时，几何六指标N/A；SAF3R为Top4、lazy first-correspondence-head Top-K，非论文DINO Top-K。', '',
          '| Method | Full-forward s | vs Dense | peak allocated GiB |','|---|---:|---:|---:|']
        for name,item in data.items():
            ref=dict(dense['input']);inp=dict(item['input'])
            assert Path(ref.pop('checkpoint')).resolve()==Path(inp.pop('checkpoint')).resolve()
            assert ref==inp and item['hardware']==dense['hardware'] and item['quality_protocol']==dense['quality_protocol']
            assert all(item['timing'][k]==dense['timing'][k] for k in ('warmup','repeats','capture'))
            monitor=json.loads((speed_root/(name+'_gpu.json')).read_text())
            assert monitor['gpu']==6 and not monitor['same_gpu_overlap_detected']
            t=item['timing']['median_seconds'];ratio=dense['timing']['median_seconds']/t
            item['quality'].update(speedup_vs_vggt=ratio,speedup_unavailable=None)
            item['paired_dense_result']=str(speed_root/'dense/timing_result.json')
            (speed_root/name/'timing_result.json').write_text(json.dumps(item,indent=2)+'\n')
            (speed_root/name/'quality.json').write_text(json.dumps(item['quality'],indent=2)+'\n')
            (speed_root/name/'quality.md').write_text(quality_markdown(item['quality']))
            memory=max(x['peak_allocated_gib'] for x in item['timing']['samples'])
            speed.append(dict(method=name,median_seconds=t,speedup_vs_dense=ratio,peak_allocated_gib=memory))
            lines.append(f'| {name} | {t:.6f} | {ratio:.4f}x | {memory:.4f} |')
        change=data['layout_fixed']['timing']['median_seconds']/data['before_layout_fix_v2']['timing']['median_seconds']-1
        lines+=['',f'布局修复相对修复前耗时变化 {change*100:+.3f}%。小样本未观察到降速，不能推断全7场景或统计显著提速。']
        (speed_root/'paired_summary.json').write_text(json.dumps(dict(rows=speed,relative_change_vs_before=change,scope='7Scenes heads128 only'),indent=2)+'\n')
    (BASE/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
