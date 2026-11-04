"""Layer states and explicitly defined local/trajectory output similarities."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'outputs/shared_value_accuracy_20261105'
OUT=BASE/'layer_analysis'
STATES=[('drop',0),('rep',1),('history',5),('exact',10)]


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def csvsave(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def state_row(layer,method,scope):
    count=layer['fine_clusters']
    row=dict(scope=scope,method=method,layer=layer['layer'],clusters=count,patch_budget=layer['patch_budget'],
        selected_patch_keys_mean=layer['counts_mean'],actual_key_density=layer['actual_unique_density'],
        forbidden_rep_pct=100*layer.get('removed_rep_clusters_mean',0)/count)
    for name,index in STATES:
        row[name+'_clusters_mean']=layer[f'state_{index}_clusters_mean']
        row[name+'_pct']=100*row[name+'_clusters_mean']/count
    assert abs(sum(row[name+'_clusters_mean'] for name,_ in STATES)-count)<.02
    for state,key in [('rep','representatives_mean'),('history','reuse_edges_mean'),('exact','exact_edges_mean')]:
        row[state+'_keys_mean']=layer.get(key,0.)
        row[state+'_key_budget_pct']=100*row[state+'_keys_mean']/layer['counts_mean']
    assert abs(sum(row[state+'_keys_mean'] for state in ('rep','history','exact'))-layer['counts_mean'])<.02
    return row


def pool(rows,subset,point):
    values=[r['output_similarity'][subset][point] for r in rows if subset in r['output_similarity']]
    n=sum(v['queries'] for v in values)
    assert n and all(v['zero_dense_vectors']==0 for v in values)
    e=sum(v['error_sum_sq'] for v in values);d=sum(v['dense_sum_sq'] for v in values)
    return dict(queries=n,cosine=sum(v['cosine_sum'] for v in values)/n,relative_l2=math.sqrt(e/d),error_sum_sq=e,dense_sum_sq=d)


def markdown_table(rows,keys):
    def val(key,value):
        if value is None:return 'N/A'
        if key.endswith('relative_l2'):return f'{100*value:.2f}%'
        if isinstance(value,float):return f'{value:.4f}'
        return str(value)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
        ['| '+' | '.join(val(k,row.get(k)) for k in keys)+' |' for row in rows])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    states=[];similarities=[];causal=[];inputs=[]
    for method,folder in [('original','layer_details_original31'),('repaired','layer_details_repaired31')]:
        root=BASE/folder
        data=json.loads((root/'audit.json').read_text())
        layers=json.loads((root/'layers.json').read_text())
        trajectory=json.loads((root/'dense_trajectory.json').read_text())
        inputs.append(json.loads((root/'input.json').read_text()))
        assert len(layers)==len(trajectory)==24
        for layer in layers:states.append(state_row(layer,method,'ETH3D kicker31'))
        for index in range(24):
            rows=[row for row in data if row['layer']==index]
            for subset in ['all','probe_non_camera','camera']:
                for point in ['av','projected']:
                    values=pool(rows,subset,point)
                    assert values['queries']=={'all':91,'probe_non_camera':60,'camera':31}[subset]
                    similarities.append(dict(method=method,layer=index,comparison='same_input_local',subset=subset,point=point,**values))
                if subset=='all':
                    t=trajectory[index]
                    values=dict(queries=sum(r['output_similarity']['all']['projected']['queries'] for r in rows),
                        cosine=t['output_cosine'],relative_l2=t['output_relative_l2'])
                else:
                    v=trajectory[index]['output_'+subset]
                    values=dict(queries=v['queries'],cosine=v['cosine_mean'],relative_l2=v['relative_l2'],
                        error_sum_sq=v['error_sum_sq'],dense_sum_sq=v['dense_sum_sq'])
                similarities.append(dict(method=method,layer=index,comparison='dense_trajectory',subset=subset,point='projected',**values))
            causal.append(dict(method=method,layer=index,
                mean_dropped_cluster_dense_probability=statistics.mean(r['dropped_cluster_dense_mass'] for r in rows),
                mean_dropped_patch_population_fraction=statistics.mean(r['dropped_patch_population_fraction'] for r in rows),
                mean_dense_special_probability=statistics.mean(r['dense_special_mass'] for r in rows),
                mean_sparse_special_probability=statistics.mean(r['errors']['executed']['special_mass'] for r in rows),
                prior_tv_mean=statistics.mean(r['prior_total_variation'] for r in rows) if index else None,
                counterfactual_same_mask_unit_projected_nmse_mean=statistics.mean(r['errors']['unit']['projected_nmse'] for r in rows),
                counterfactual_same_mask_cluster_projected_nmse_mean=statistics.mean(r['errors']['cluster_count']['projected_nmse'] for r in rows)))
    for key in ['image_paths','frame_indices','shape','checkpoint','sampled_query_indices','camera_query_indices']:
        assert inputs[0][key]==inputs[1][key],key
    for row in similarities:
        assert -1.000001<=row['cosine']<=1.000001 and row['relative_l2']>=0
    quality_states=[]
    for p in sorted((BASE/'quality').glob('*/*/layers.json')):
        dataset,scene=p.parent.parts[-2:]
        for layer in json.loads(p.read_text()):quality_states.append(state_row(layer,'repaired',dataset+'/'+scene))
    csvsave('states_kicker31.csv',states);csvsave('output_similarities_kicker31.csv',similarities)
    csvsave('causal_diagnostics_kicker31.csv',causal);csvsave('states_quality_10_scenes.csv',quality_states)
    macro_states=[]
    for dataset,count in [('7scenes',7),('eth3d',3)]:
        for layer in range(24):
            rows=[r for r in quality_states if r['scope'].startswith(dataset+'/') and r['layer']==layer]
            assert len(rows)==count
            keys=['actual_key_density','forbidden_rep_pct']+[s+suffix for s,_ in STATES for suffix in ['_pct']]+[s+'_key_budget_pct' for s in ['rep','history','exact']]
            macro_states.append(dict(dataset=dataset,scenes=count,layer=layer,aggregation='equal_scene_mean',
                **{key:statistics.mean(r[key] for r in rows) for key in keys}))
    csvsave('states_quality_equal_scene_macro.csv',macro_states)

    # Exact counterexample: identical executed support and weights, different
    # Rep/History proxy losses. This is algebra, not a dataset quality result.
    import numpy as np
    z=np.array([[1.,0.],[1.,.1],[1.,-.1]],dtype=np.float64)
    z=z/np.sqrt(np.mean(np.sum(z*z,axis=1)))
    p=np.array([.4,.3,.3]);rep_loss=float(np.sum(p*np.sum((z-z[0])**2,axis=1)))
    history_loss=float(np.sum(p[1:]*np.sum(z[1:]**2,axis=1)))
    def toy_forward(support,weight):
        value=np.vstack([z[support],np.array([[.2,.4]])])
        logits=np.array([.2]*len(support)+[-.1])+np.log(np.array([weight]*len(support)+[1.]))
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        return probability@value
    forward_checks={}
    for method,weight in [('original',1.),('repaired',3.)]:
        rep_output=toy_forward([0],weight);history_output=toy_forward([0],weight)
        assert np.array_equal(rep_output,history_output)
        forward_checks[method]=dict(rep_output=rep_output.tolist(),history_output=history_output.tolist(),
            output_l2_difference=float(np.linalg.norm(rep_output-history_output)))
    toy=dict(prior=p.tolist(),values=z.tolist(),rep_support=[0],history_support=[0],rep_loss=rep_loss,
        history_loss=history_loss,history_to_rep_proxy_ratio=history_loss/rep_loss,
        support_cost_each=1,same_forward_output_for_equal_support_and_weights=True,
        forward_checks=forward_checks,
        note='Original unit weights: both1. Repaired cluster_population weights: both3. Within either method the two states execute the same operator.')
    save('same_support_proxy_counterexample.json',toy)
    # Identical Values give zero Rep distortion, but heterogeneous Keys make
    # population compensation misestimate the softmax cluster mass.
    logits=np.array([0.,2.,0.])
    probability=np.exp(logits-logits.max());probability/=probability.sum()
    dense_output=float(probability@np.array([1.,1.,0.]))
    sparse_logits=np.array([np.log(2.),0.])
    sparse_probability=np.exp(sparse_logits-sparse_logits.max());sparse_probability/=sparse_probability.sum()
    sparse_output=float(sparse_probability@np.array([1.,0.]))
    key_mass_toy=dict(values=[1.,1.,0.],dense_logits=logits.tolist(),representative=0,rep_value_distortion=0.,
        representative_weight=2.,dense_output=dense_output,sparse_output=sparse_output,
        relative_l2=abs(sparse_output-dense_output)/abs(dense_output),cosine=1.,
        reason='First two Keys have identical Values but different QK logits; third Key is a retained special with Value0')
    assert key_mass_toy['rep_value_distortion']==0 and key_mass_toy['relative_l2']>.2
    save('zero_value_loss_key_mass_counterexample.json',key_mass_toy)

    def get(method,layer,comparison,subset='probe_non_camera'):
        return next(r for r in similarities if r['method']==method and r['layer']==layer and r['comparison']==comparison and r['subset']==subset and r['point']=='projected')
    state_display=[];local_display=[];trajectory_display=[];camera_display=[]
    for layer in range(24):
        sr=dict(layer=layer)
        for method in ['original','repaired']:
            row=next(r for r in states if r['method']==method and r['layer']==layer)
            sr[method+'_D/R/H/E %']='/'.join(f'{row[s+"_pct"]:.1f}' for s,_ in STATES)
        state_display.append(sr)
        for table,comparison,subset in [(local_display,'same_input_local','probe_non_camera'),
            (trajectory_display,'dense_trajectory','probe_non_camera'),(camera_display,'dense_trajectory','camera')]:
            row=dict(layer=layer)
            for method in ['original','repaired']:
                v=get(method,layer,comparison,subset)
                row[method+'_cosine']=v['cosine'];row[method+'_relative_l2']=v['relative_l2']
            table.append(row)
    save('summary.json',dict(scope=inputs[0],state_rows=states,similarity_rows=similarities,causal_rows=causal,
        state_table=state_display,local_table=local_display,trajectory_table=trajectory_display,camera_trajectory_table=camera_display,
        toy=toy,key_mass_toy=key_mass_toy,quality_state_rows=len(quality_states),quality_state_equal_scene_macro=macro_states,complete=True))

    from scripts.quality_metrics import quality_record,quality_markdown
    q=quality_record({},'Layer-attention diagnostic and postprocessing only; no new pose/recon_unposed evaluation')
    q['speedup_unavailable']='Diagnostic instrumentation plus Dense diagnostic forward; not a warmed complete-forward speed test'
    save('quality.json',q);(OUT/'quality.md').write_text(quality_markdown(q))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'original':'#d05a38','repaired':'#2577ad'}
    fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True,constrained_layout=True)
    for col,comparison in enumerate(['same_input_local','dense_trajectory']):
        for method in ['original','repaired']:
            for subset,style in [('probe_non_camera','-'),('camera','--')]:
                a=[get(method,l,comparison,subset) for l in range(24)]
                label=method+(' camera' if subset=='camera' else ' regular probes')
                axes[0,col].plot(range(24),[r['cosine'] for r in a],style,color=colors[method],label=label)
                axes[1,col].plot(range(24),[100*r['relative_l2'] for r in a],style,color=colors[method],label=label)
        axes[0,col].set_title('Same-input local attention' if col==0 else 'Independent Dense trajectory')
        axes[0,col].set_ylabel('Mean query cosine similarity');axes[0,col].set_ylim(-.05,1.025)
        axes[1,col].set_ylabel('Relative L2 (%)');axes[1,col].set_yscale('log');axes[1,col].set_xlabel('Global layer (0-based)')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xticks(range(0,24,2))
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.savefig(OUT/'output_similarity.png',dpi=180);fig.savefig(OUT/'output_similarity.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True,constrained_layout=True)
    for ax,method in zip(axes,['original','repaired']):
        base=np.zeros(24)
        for state,color in zip(['drop','rep','history','exact'],['#9c9fa5','#4c83ad','#65a083','#dd9e4c']):
            y=[next(r for r in states if r['method']==method and r['layer']==l)[state+'_pct'] for l in range(24)]
            ax.bar(range(24),y,bottom=base,color=color,label=state);base+=y
        ax.set_title(method);ax.set_xlabel('Global layer (0-based)');ax.set_xticks(range(0,24,2));ax.set_ylim(0,100)
    axes[0].set_ylabel('Cluster state selection (%)');axes[1].legend(ncol=4,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01))
    fig.savefig(OUT/'state_selection.png',dpi=180);fig.savefig(OUT/'state_selection.svg');plt.close(fig)
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

    metric_keys=['layer','original_cosine','original_relative_l2','repaired_cosine','repaired_relative_l2']
    lines=['# Shared Value Loss Query-256：逐层状态、输出相似度与退化原因','',
        '## 范围与定义','',
        '本表来自新补充的 ETH3D kicker 全31帧诊断：原方案和修复版使用相同 checkpoint、精确图像列表、预处理与20% Key预算。每层观察固定60个常规探针，并额外覆盖所有31个 camera 查询，共91个查询。共24层，层编号从0开始。状态统计覆盖全部901个Query组，而输出相似度只覆盖这91个查询，不是所有token的穷尽比较，也不能直接推广到所有数据集。', '',
        '输出取 global attention 的 Wo 投影及 bias 之后、LayerScale/残差之前。局部比较从实际 sparse AV kernel 输出出发，投影用FP32计算；全轨迹比较使用模型实际执行的投影输出（BF16）。相对L2均以Dense为分母；不是绝对L2，也不是NMSE。', '',
        r'$$\operatorname{RelL2}=\frac{\|O_s-O_d\|_F}{\|O_d\|_F},\qquad \operatorname{Cos}=\frac1M\sum_{i=1}^M\frac{O_{s,i}^{\top}O_{d,i}}{\|O_{s,i}\|_2\|O_{d,i}\|_2}.$$','',
        'RelL2通过各组误差平方和、Dense平方和先求和再开方；Cos按查询等权平均。此前的组均值NMSE不能直接当成相对L2百分比，也不能直接开方当成本表的全样本聚合值。', '',
        '## 原因：已证实的结构问题','',
        '1. **状态损失没有一致描述实际执行算子。** 如果历史合格成员只有当前Rep，Rep和History发出的都是同一个Key，且同一方法内权重也相同；实际attention输出完全相同，但Rep使用簇内替代失真，History把其余成员按Drop计算，得到不同的状态损失。受控反例的Rep loss为'+f'{rep_loss:.6f}'+', History loss为'+f'{history_loss:.6f}'+', 相差'+f'{history_loss/rep_loss:.1f}倍'+'。这说明目标函数的状态标签能改变预算决策，即使真实mask/输出没有变化。反例属于代数证据，不是几何质量测量。','',
        '2. **历史Rep禁用放大上述不一致。** 原方案只要历史中出现当前Rep就禁止Rep，即使那个低成本支持集仍有效。其替代的History/Exact可能更贵，预算求解会通过Drop其他簇释放预算。表中保留各层被禁用Rep比例、Drop比例、每状态实际Key花费；因此可以区分“簇覆盖损失”和“Key预算是否耗尽”。这是机制解释，不能把全部Drop或全部精度损失都单独归因于这一条件。','',
        '3. **单位权重破坏了压缩前后的相对概率质量。** 保留单个Rep却不给它承接簇成员质量，会使未压缩的special Keys相对过重。原始audit同时保存同mask只改计权的反事实结果，支持这是独立于选mask的误差来源。补偿只能修正近似质量，不能恢复被Drop的簇或簇内不均匀QK。','',
        '4. **Value失真为零，也不能保证attention输出正确。** 一个簇的两个Value均为1、QK logits分别为0和2；另有一个保留的special，其Value为0、logit为0。用第一个成员作Rep时，Rep Value loss严格为0。即使给Rep乘簇大小2，Dense输出仍为'+f'{dense_output:.6f}'+', Sparse输出为'+f'{sparse_output:.6f}'+', 相对L2为'+f'{100*key_mass_toy["relative_l2"]:.2f}%'+'，余弦却为1。原因是真实簇质量为exp(0)+exp(2)，近似质量只有2exp(0)。这是Value-only损失遗漏QK/softmax质量误差的可复现反例；它证明方法没有输出保真保证，但不能直接量化这个机制在真实数据中占多少精度损失。','',
        '## 修复后为什么仍有误差','',
        '修复后大量簇选择Rep，History很少被选。这并不证明历史没有用：当前History loss仍按未保留成员Drop计费，而修复后的History执行已经承接簇质量，损失语义与执行仍不完全一致。另一个原因是簇数等于Key预算：全部簇各保留一个Rep已用满预算，要增加History/Exact就必须从其他簇释放预算。', '',
        '实测原方案L8被Drop的簇在抽样查询中承载约19.68%的真实Dense attention概率质量；修复版同层约0.16%。这里先在每个被抽样Query组内平均，再按组等权平均；这是不同轨迹各自的诊断值，不是固定Q/K/V的单变量消融。另一方面，两者L1–L23的先验总变差均值仍约0.29：比较的是组共享先验和本组抽样查询的真实分布，既含概率预测误差，也含组内查询差异，不能当成独立历史模块的误差归因。', '',
        'Value聚类相似也不保证Key的QK相似；n/m补偿隐含的簇内代表性只是近似。32个Query及多个heads共享支持集，而概率依据粗Grid和Query-256历史估计，这些都是剩余近似来源。当前诊断能测量其合成误差，不能独立量化每个近似对最终AUC/F1的因果贡献。', '',
        '最关键的区分是：同输入的单层局部误差很小，不意味着跨24层累计误差很小。camera head直接读取每帧token0，因此下面另外列出camera查询；它们的偏差是姿态风险证据，但不是最终pose误差的直接等价量。', '',
        '## 各层四状态分布','',
        'D/R/H/E依次是Drop、Rep、History、Exact；数字为占所有簇的百分比，四项加起来约100%。每组共有5611个簇；CSV同时给出每Query组平均簇数（可为小数）、各状态实际Key数/花费比例和Rep禁用比例。', '',
        markdown_table(state_display,['layer','original_D/R/H/E %','repaired_D/R/H/E %']), '',
        f'![state selection]({OUT}/state_selection.png)', '',
        '## 同一层输入下：局部稀疏近似误差','',
        '本表使用60个常规探针：Sparse和Dense使用相同的该层Q/K/V，回答“这一层本身的mask/计权造成多大误差”。两个方法各自沿自己的稀疏轨迹运行。', '',
        markdown_table(local_display,metric_keys), '',
        '## 独立Dense整网轨迹：累计误差','',
        '本表仍为同样60个探针，但Dense沿自身完整稠密轨迹运行，Sparse沿自身稀疏轨迹运行。该层输入已经不同，反映此前所有层的累计影响，不能称为单层近似误差。', '',
        markdown_table(trajectory_display,metric_keys), '',
        '## 全31个camera查询：累计输出误差','',
        markdown_table(camera_display,metric_keys), '',
        f'![output similarity]({OUT}/output_similarity.png)', '',
        '## 文件与实验属性','',
        '`states_kicker31.csv`：本次原/修复方案48条逐层状态记录；`output_similarities_kicker31.csv`：AV/投影、局部/累计、常规/camera/all的完整指标；`causal_diagnostics_kicker31.csv`：被Drop簇的真实Dense概率质量、special质量偏差、先验TV；`states_quality_10_scenes.csv`：此前7Scenes七场景和ETH3D三场景质量运行的240条逐层状态记录；`states_quality_equal_scene_macro.csv`：分别按7/3个场景等权平均的48条状态比例记录。后两者没有完整Dense输出观测，不能据此填造其相似度。', '',
        '新增诊断没有计算pose或recon_unposed，六项指标全部N/A，见quality.md；它们不替代此前的10场景完整质量评估。诊断包含额外Dense QK和Dense前向，只用于测量，不能当作方法耗时或Dense加速比。生产推理代码、配置和预算本轮均未修改。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',OUT,'state rows',len(states),'similarity rows',len(similarities),'quality scene-layer rows',len(quality_states))


if __name__=='__main__':main()
