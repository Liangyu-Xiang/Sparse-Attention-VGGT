"""Exact, mutually exclusive runtime ledger from one profiled forward per scene.

All display numbers are integer microseconds; explicit parent remainders preserve
both timing coverage and rounding closure. Never scale stages to a normal run.
"""
from pathlib import Path
from dataclasses import dataclass,field
import csv
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/deep_saf3r_query256_20261105_final'
SCENES=('heads','chess','stairs')
METHODS=('saf3r','shared_value_q256')

@dataclass
class Node:
    name:str
    seconds:float
    children:list=field(default_factory=list)
    kind:str='measured'
    us:int=0

def leaf(name,value,kind='measured'):return Node(name,float(value),kind=kind)

def parent(name,value,children):
    value=float(value);remain=value-sum(c.seconds for c in children)
    # A meaningful negative remainder means an attribution/overlap error.
    if remain < -0.00005:raise AssertionError((name,'negative remainder',remain,value,[(c.name,c.seconds) for c in children]))
    if abs(remain)<1e-12:remain=0.
    label='未细分、分配、提交与阶段边界' if remain>=0 else '时钟及计量差额（带符号）'
    return Node(name,value,children+[leaf(label,remain,'remainder')])

def quantize(n):
    n.us=round(n.seconds*1e6)
    for c in n.children:quantize(c)
    if n.children:
        assert n.children[-1].kind=='remainder'
        n.children[-1].us=n.us-sum(c.us for c in n.children[:-1])
        assert sum(c.us for c in n.children)==n.us
    return n

def combine(nodes):
    first=nodes[0]
    assert all(n.name==first.name and len(n.children)==len(first.children) for n in nodes)
    return Node(first.name,sum(n.seconds for n in nodes)/len(nodes),
        [combine([n.children[i] for n in nodes]) for i in range(len(first.children))],first.kind)

def leaves(n,path=()):
    path=path+(n.name,)
    if not n.children:return [('/'.join(path),n.us,n.kind)]
    return sum((leaves(c,path) for c in n.children),[])

def build_tree(d,p):
    ev={name:v['cuda_interval_seconds'] for name,v in p['intervals'].items()}
    get=lambda name:ev.get(name,0.)
    P=lambda name,key,children:parent(name,get(key),children)
    L=lambda name,key:leaf(name,get(key))
    projection=L('QKV与输出投影','projection/qkv');projection.seconds+=get('projection/output')
    if d['method']=='saf3r':
        support=Node('支持集构建',sum(v for n,v in ev.items() if n.startswith('sf_select/')))
        support=parent(support.name,support.seconds,[
            P('首次对应头Top-K','sf_select/lazy_topk',[
                L('融合QK与逐帧Top-K kernel','sf_detail/fused_correspondence_qk_topk')]),
            L('Query-probe均值Q、打分、Top-K','sf_select/q_probe'),
            L('Stride索引、并集去重','sf_select/stride_indices')])
        av=sum(v for n,v in ev.items() if n.startswith('sf_av/'))
        dispatch=sum(v for n,v in ev.items() if n.startswith('sf_dispatch/'))
        execution=parent('支持集上的attention执行',dispatch-get('sf_select/q_probe')-get('sf_select/stride_indices'),[
            L('SDPA：首帧头','sf_av/sdpa/all_to_first'),L('SDPA：Stride头','sf_av/sdpa/stride'),
            L('SDPA：Query-probe头','sf_av/sdpa/q_probe_topk'),L('SDPA：帧内头','sf_av/sdpa/global_to_frame'),
            L('SDPA：对应头特殊Query','sf_av/sdpa/dino_topk'),L('融合稀疏AV：对应头patch Query','sf_av/correspondence_fused'),
            L('Query-probe K/V gather','sf_pack/q_probe_kv'),L('Stride K/V gather','sf_pack/stride_kv')])
        history=leaf('历史概率统计与补全',0.)
    else:
        cluster=P('Value聚类','shared_stage/value_update',[
            L('各头V均值与patch特征','cluster/head_mean_value_features'),
            L('簇中心与人口重算','cluster/means_population'),
            P('L0全局成员分配','cluster/L0_global_assignment',[
                L('Token×全部中心距离','cluster_sub/global_distances'),
                L('最小距离、保留旧标签与写回','cluster_sub/global_argmin_assignment')]),
            P('中心邻接图','cluster/center_neighbor_graph',[
                L('中心×全部中心距离','cluster_sub/center_graph_distances'),
                L('Top-32邻居与边界距离','cluster_sub/center_graph_topk')]),
            P('后续层局部成员分配','cluster/local_assignment',[
                L('Gather32候选、距离与分配','cluster_sub/local_gather_distance_assignment'),
                L('认证边界计算','cluster_sub/local_certificate_math')]),
            L('空簇检测与修复','cluster/empty_repair'),L('非空检查与同步','cluster/nonempty_check_sync'),
            L('认证统计与nonzero同步','cluster/local_diagnostics_sync'),L('成员变化计数与同步','cluster/change_count_sync'),
            L('最终真实代表点选择','cluster/final_real_representatives')])
        geometry=P('池化与共享几何','shared_stage/pooling',[
            L('簇人口、Grid池化K、Cluster–Grid交叠','geometry/population_grid_pool_overlap'),
            L('Value范数与代表误差','geometry/value_errors'),L('按簇人口建立kernel分桶','geometry/population_bins'),
            L('Cluster×Grid共享损失统计','geometry/cluster_grid_statistics')])
        coarse=P('粗感知','shared_stage/m5c_coarse_observation',[
            L('四个真实Query取样与布局','coarse/probe_gather'),L('Grid与特殊Key的QK','coarse/grid_special_qk'),
            L('含特殊Key的粗归一化','coarse/normalization'),leaf('Patch-only粗概率（未单列，实际计时包含在本节剩余）',get('coarse/patch_only_probability'),'included_in_remainder')])
        candidates=P('残差候选筛选','shared_stage/residual_fine_probe',[
            L('Grid残差→Cluster稀疏矩阵乘','candidate/grid_to_cluster_mm'),L('Top-25%簇筛选','candidate/topk_clusters'),
            L('构建Exact-allowed标志','candidate/allow_flags')])
        prior=P('历史prior','selection/prior_total',[
            L('当前粗分布、beta和均匀补全','prior_detail/grid_beta_and_uniform'),
            L('展开到全部patch的prior','prior_detail/expand_all_patch_prior'),
            L('采样历史概率预测','prior_detail/history_predict'),L('Scatter汇总簇概率','prior_detail/cluster_scatter'),
            L('概率归一化校验','prior_detail/normalization_check')])
        gd=p['graph_detail'];gs=gd['stages_seconds']
        G=lambda name,key:leaf(name,gs.get(key,0.),'graph_device_timestamp')
        pack=parent('Graph内支持集打包',gs['pack'],[
            G('逐patch选中flags','pack_flags'),G('逐patch前缀和','pack_prefix'),G('预算及状态校验','pack_validate'),
            G('输出紧凑Key ID','pack_emit'),G('数量统计与输出标志','pack_output_statistics')])
        graph=parent('Graph内四状态选边计算',gd['total_seconds'],[
            G('历史门控、计数、代表覆盖','gate'),G('历史概率修正共享损失','history_loss_corrections'),
            G('四状态误差、成本和目标','four_state_errors'),G('16轮预算二分与最终选择','budget16'),
            G('状态编码转换','decode'),pack,G('诊断统计与延迟校验','diagnostics')])
        replay=P('CUDA Graph重放','graph_host/replay',[graph])
        selector=P('Selector与Graph管理','selection/selector_total',[
            P('Graph缓存miss时创建与捕获','graph_host/create_capture',[
                L('创建过程中的几何拷贝','graph_host/geometry_copies_during_create')]),
            L('跨层几何常量拷贝','graph_host/geometry_copies'),L('每窗口prior等输入拷贝','graph_host/input_copies'),
            replay,L('Graph诊断输出拷贝','graph_host/record_copies')])
        budget=P('prior、损失、预算与打包','shared_stage/reuse_and_budget',[prior,selector])
        support=parent('支持集构建',sum(get('shared_stage/'+n) for n in
            ('value_update','pooling','m5c_coarse_observation','coarse_observation','residual_fine_probe','reuse_and_budget','cluster_seed_prepare')),
            [cluster,geometry,coarse,L('本层粗稳定性与残差','shared_stage/coarse_observation'),candidates,budget,
             L('L0初始支持准备','shared_stage/cluster_seed_prepare')])
        execution=P('支持集上的attention执行','shared_stage/execution_and_evidence',[
            L('索引式融合QK、softmax、AV及inline采样','shared_execution/sampled_direct_av'),
            L('独立历史概率导出','shared_execution/sampled_probability_export')])
        history=parent('历史概率统计与补全',sum(get('shared_stage/'+n) for n in
            ('sampled_probe_grid_summary','m5c_evidence_and_host_copy','history_summary')),[
            L('选中Key概率→Grid统计','shared_stage/sampled_probe_grid_summary'),L('缺失概率补全与FP16历史缓存','shared_stage/m5c_evidence_and_host_copy'),
            L('下一层历史ID、Grid摘要','shared_stage/history_summary')])
    attention=P('Global attention','global/attention_wrapper',[projection,support,execution,history])
    global_other=get('model/global_blocks')-get('global/attention_wrapper')-get('global/mlp')
    heads=[L('相机头','model/camera_head'),L('深度头','model/depth_head'),L('点云头','model/point_head')]
    model_known=get('model/image_encoder')+get('model/frame_blocks')+get('model/global_blocks')+sum(h.seconds for h in heads)
    root=parent('完整前向wall时间',p['wall_seconds'],[
        L('图像编码','model/image_encoder'),L('Frame blocks','model/frame_blocks'),attention,
        L('Global MLP','global/mlp'),leaf('Global归一化、残差等',global_other),*heads,
        leaf('模型其余：初始化、布局、未细分区间',p['cuda_interval_seconds']-model_known),
        leaf('Wall计时边界、返回与末尾同步差额',p['wall_seconds']-p['cuda_interval_seconds'])])
    return root


def find(n,path):
    if not path:return n
    return find(next(c for c in n.children if c.name==path[0]),path[1:])


def main():
    trees={};runs={};monitors={};selected={};mean_trees={}
    for scene in SCENES:
        trees[scene]={};runs[scene]={};selected[scene]={}
        for method in METHODS:
            path=OUT/scene/method/'result.json'
            if not path.exists():raise RuntimeError('Missing complete run:'+str(path))
            d=json.loads(path.read_text());monitor=json.loads((OUT/f'{scene}_{method}_monitor.json').read_text())
            assert not monitor['same_gpu_overlap_detected'] and not monitor['foreign_gpu_pids'] and monitor['returncode']==0
            assert len(d['event_profiles'])==3 and len(d['normal_samples'])==4
            # A table uses the WHOLE-forward median pass, never per-row medians.
            idx=sorted(range(3),key=lambda i:d['event_profiles'][i]['wall_seconds'])[1]
            tree=build_tree(d,d['event_profiles'][idx]);quantize(tree)
            assert sum(us for _,us,_ in leaves(tree))==tree.us
            trees[scene][method]=tree;runs[scene][method]=d;selected[scene][method]=idx;monitors[f'{scene}/{method}']=monitor
        a,b=(runs[scene][m] for m in METHODS)
        assert a['input']==b['input'] and a['hardware']==b['hardware']
        assert monitors[f'{scene}/saf3r']['gpu_uuid']==monitors[f'{scene}/shared_value_q256']['gpu_uuid']
        ha=json.loads((OUT/scene/'saf3r/source_sha256.json').read_text());hb=json.loads((OUT/scene/'shared_value_q256/source_sha256.json').read_text());assert ha==hb
    for method in METHODS:mean_trees[method]=quantize(combine([trees[s][method] for s in SCENES]))
    def serialize(n):return dict(name=n.name,seconds_raw=n.seconds,integer_microseconds=n.us,kind=n.kind,children=[serialize(c) for c in n.children])
    summary=dict(query_group_size=256,frames=128,scenes=SCENES,selected_profile_index=selected,
        per_scene={s:{m:serialize(trees[s][m]) for m in METHODS} for s in SCENES},
        mean_tree={m:serialize(mean_trees[m]) for m in METHODS},monitors=monitors,
        normal_warmed_median_seconds={s:{m:runs[s][m]['normal_warmed_median_seconds'] for m in METHODS} for s in SCENES},
        closure={m:dict(complete_wall_us=mean_trees[m].us,leaf_sum_us=sum(v for _,v,_ in leaves(mean_trees[m])),difference_us=0) for m in METHODS},
        complete=True,quality_evaluated=False,quality={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},
        quality_unavailable='Explicit operation-timing diagnostic; pose/recon_unposed not evaluated',
        speedup_vs_dense=None,speedup_unavailable='No new paired Dense timing; unchanged control not rerun')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_reports(trees,mean_trees,runs,selected)
    print(json.dumps(summary['closure'],indent=2))
    print('COMPLETE WALL', {m:mean_trees[m].us/1e6 for m in METHODS})


def write_reports(trees,means,runs,selected):
    def f(us):return f'{us/1e6:.6f}'
    def table_for_nodes(nodes):
        lines=['| 操作 | SAF3R 秒 | Query-256 秒 | Query-256额外秒 |','|---|---:|---:|---:|']
        for label,a,b in nodes:lines.append(f'| {label} | {f(a)} | {f(b)} | {(b-a)/1e6:+.6f} |')
        return lines
    notes=['所有数值来自同一次带统计的完整前向：每序列选择3次统计运行中wall总时间居中的那一次，并取其全部明细。',
        '总表使用三个序列的等权平均。没有逐行取不同运行的中位数，也没有用阶段缩放去匹配另一次正常测速。',
        '显示精度为整数微秒（秒保留6位）。每个父节点=子节点+显式剩余；微秒取整差额保留在剩余项。个别剩余为负1–2微秒时仅表示显示取整，不表示实际负耗时。',
        'included_in_remainder表示未独立计时，时间包含在父项剩余中，不把显示的0解释成该操作无开销。',
        'Graph时间戳会增加标记kernel，细分探针也会扰动运行；因此本报告总账对应的是同一次带统计完整前向，正常测速另存normal_timings.md，不能混合。',
        'CUDA Event区间包含CPU提交/队列间隙；Graph内用每次重放独立保存的GPU时间戳。两者之间的差额单列，不是纯kernel self time。',
        'Query-256：真实每组256 Query，20% Key预算，原始约20%簇数，历史采样间隔256；SAF3R lazy首个对应头Top4，每轮重新选边，非论文DINO Top-K。',
        '同序列两方法在同一独占RTX4090；只测heads/chess/stairs各128帧，不是完整7Scenes宏平均。']
    sf,sv=(means[m] for m in METHODS)
    def top_rows(a,b):
        ga=find(a,['Global attention']);gb=find(b,['Global attention'])
        rows=[]
        for name in ('支持集构建','支持集上的attention执行','历史概率统计与补全','QKV与输出投影'):
            rows.append((name,find(ga,[name]).us,find(gb,[name]).us))
        rem_a=ga.us-sum(c.us for c in ga.children if c.name in ('支持集构建','支持集上的attention执行','历史概率统计与补全','QKV与输出投影'))
        rem_b=gb.us-sum(c.us for c in gb.children if c.name in ('支持集构建','支持集上的attention执行','历史概率统计与补全','QKV与输出投影'))
        rows.append(('Global attention其余',rem_a,rem_b))
        for ca,cb in zip(a.children,b.children):
            if ca.name=='Global attention':continue
            assert ca.name==cb.name;rows.append((ca.name,ca.us,cb.us))
        assert sum(x[1] for x in rows)==a.us and sum(x[2] for x in rows)==b.us
        return rows
    lines=['# Query-256 对 SAF3R：完整运行时间的闭合耗时账本','',*notes,'','## 总账（以下各行可以直接相加）','']
    rows=top_rows(sf,sv);lines+=table_for_nodes(rows)
    lines+=[f'| **完整运行总时间＝以上各行之和** | **{f(sf.us)}** | **{f(sv.us)}** | **{(sv.us-sf.us)/1e6:+.6f}** |','',
        f'自动校验：SAF3R合计={f(sum(x[1] for x in rows))}秒；Query-256合计={f(sum(x[2] for x in rows))}秒；两者与完整wall时间之差均为0微秒。','',
        '## 支持集构建展开（子项已包含在总账父项中，不要重复相加）','']
    supa=find(sf,['Global attention','支持集构建']);supb=find(sv,['Global attention','支持集构建'])
    lines+=['| Query-256操作 | 秒 |','|---|---:|']
    for child in supb.children:lines.append(f'| {child.name} | {f(child.us)} |')
    lines.append(f'| **支持集构建合计** | **{f(supb.us)}** |')
    lines+=['','| SAF3R对应操作 | 秒 |','|---|---:|']
    for child in supa.children:lines.append(f'| {child.name} | {f(child.us)} |')
    lines.append(f'| **支持集构建合计** | **{f(supa.us)}** |')
    # Expand the expensive shared-only branches, keeping parent sums visible.
    def expand(node,depth=0):
        out=[]
        for child in node.children:
            out.append(f'| {"　"*depth}{child.name} | {f(child.us)} | {"展开项，已含子项" if child.children else child.kind} |')
            if child.children:out+=expand(child,depth+1)
        return out
    for name in ('Value聚类','池化与共享几何','粗感知','残差候选筛选','prior、损失、预算与打包'):
        n=find(supb,[name]);lines+=['',f'## {name}：{f(n.us)}秒','',
            '树中父行和子行有包含关系；仅叶子项相加等于本节父项。','',
            '| 操作 | 秒 | 计量关系 |','|---|---:|---|',*expand(n)]
    lines+=['','## 每序列闭合总账','']
    for scene in SCENES:
        a,b=(trees[scene][m] for m in METHODS)
        lines += [f'### {scene}（统计pass索引：{selected[scene]}）','',*table_for_nodes(top_rows(a,b)),
            f'| **完整wall与明细之和** | **{f(a.us)}** | **{f(b.us)}** | **{(b.us-a.us)/1e6:+.6f}** |','']
    lines+=['## 六项质量字段与Dense速度比','',
        '本次明确是细粒度性能诊断，不评估pose/recon_unposed；每个运行六指标均N/A，原因记录在quality.json/quality.md。没有重跑Dense，因此本批配对Dense加速比为N/A。',
        '重建若评估：距离单位米，F1阈值0.05m，下采样0.0078125m，CD=(ACC+Comp)/2且使用未平方距离。','',
        '| Scene | Method | AUC@3 | AUC@30 | ACC | Comp | CD | F1 | vs Dense |','|---|---|---|---|---|---|---|---|---|']
    for scene in SCENES:
        for method in METHODS:lines.append(f'| {scene} | {method} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    # Flat leaf ledgers are the unambiguous additive form, never include parent rows.
    for label,pair in [('mean',means)]+[(s,trees[s]) for s in SCENES]:
        maps={m:{path:(us,kind) for path,us,kind in leaves(pair[m])} for m in METHODS}
        paths=sorted(set(maps['saf3r'])|set(maps['shared_value_q256']))
        flat=['# 完整前向叶子账本：'+label,'',*notes,'','以下只列互斥叶子项，全部相加必须等于完整wall时间。','',
            '| 操作路径 | SAF3R秒 | Query-256秒 |','|---|---:|---:|']
        csv_rows=[]
        for path in paths:
            a=maps['saf3r'].get(path,(0,'absent'))[0];b=maps['shared_value_q256'].get(path,(0,'absent'))[0]
            flat.append(f'| {path} | {f(a)} | {f(b)} |');csv_rows.append([path,a,b])
        assert sum(x[1] for x in csv_rows)==pair['saf3r'].us
        assert sum(x[2] for x in csv_rows)==pair['shared_value_q256'].us
        flat.append(f"| **全部叶子项合计＝完整wall时间** | **{f(pair['saf3r'].us)}** | **{f(pair['shared_value_q256'].us)}** |")
        (OUT/f'leaf_ledger_{label}.md').write_text('\n'.join(flat)+'\n')
        with (OUT/f'leaf_ledger_{label}.csv').open('w') as handle:
            writer=csv.writer(handle);writer.writerow(['operation','saf3r_integer_us','query256_integer_us']);writer.writerows(csv_rows)
    normal=['# 正常测速（独立运行，不与细分账本拼接）','',
        '同一方法1warmup+3不带细分统计的完整前向，中位数如下；统计账本使用另外的完整前向，因此不能拿此表总时间与账本阶段相加。','',
        '| Scene | SAF3R秒 | Query-256秒 |','|---|---:|---:|']
    for scene in SCENES:normal.append(f"| {scene} | {runs[scene]['saf3r']['normal_warmed_median_seconds']:.6f} | {runs[scene]['shared_value_q256']['normal_warmed_median_seconds']:.6f} |")
    (OUT/'normal_timings.md').write_text('\n'.join(normal)+'\n')

if __name__=='__main__':main()
