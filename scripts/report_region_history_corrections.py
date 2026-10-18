"""Evidence-backed comparison of fixes and controlled Dense-information oracles."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LABELS={
 'legacy':'原版B','hard_history':'仅硬保留history','size_bias_only':'仅区域大小偏置',
 'hard_size_rep':'硬保留+大小偏置，仍取中心','exact_mass':'精确group-mass oracle，仍取中心',
 'hard_exact_rep':'硬保留+精确mass，仍取中心','corrected_B':'修正版B：硬保留+大小偏置+局部选Key',
 'corrected_A':'修正版A：硬保留+大小偏置+局部选Key',
 'full_oracle':'硬保留+精确mass+Dense最佳成员oracle',
 'best_member_frozen':'原版区域/历史固定，仅Dense最佳成员oracle',
 'exact_mass_best_frozen':'精确mass区域/历史固定，仅Dense最佳成员oracle',
 'uniform':'均匀真实Key','dense_topk':'Dense attention TopK oracle',
 'influence_topk':'单边删除影响力TopK diagnostic'}


def run(root):
    all_rows=[];summary=[]
    for budget in [10,20]:
        r=json.loads((root/f'chess_two_frames/budget{budget}/result.json').read_text())
        assert r['dense_capture_bitwise_equal'] and r['legacy_masks_bitwise_equal'] and r['legacy_output_error_reproduced']
        all_rows.extend(r['rows'])
        for method in LABELS:
            rows=[row for row in r['rows'] if row['method']==method]
            assert len(rows)==7
            summary.append(dict(budget=budget,method=method,**{key:mean(row[key] for row in rows)
                for key in rows[0] if key not in ['method','layer','uses_dense_oracle','retention']},
                uses_dense_oracle=rows[0]['uses_dense_oracle']))
    native=json.loads((root/'native_two_frames/result.json').read_text())
    for flag in ['all_incoming_history_preserved','all_specials_preserved','all_masks_unique_exact_budget',
                 'dense_prefix_bitwise_equal','repeated_pose_bitwise_equal']:assert native[flag]
    out=root/'report';out.mkdir(exist_ok=True)
    for name,rows in [('all_layers',all_rows),('summary',summary),('native_layers',native['rows'])]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    get=lambda b,m:next(r for r in summary if r['budget']==b and r['method']==m)
    lines=['# 区域Attention修正与四组受控oracle','',
        'Chess seq-01第0/10帧；0–9及17–23层Dense，10–16为测试层。'
        '总边预算10%/20%，均含10个special Key；history transfer固定50%。'
        '受控表所有方法使用同一组原始Dense输入，各自递推history；冻结最佳成员对照除外。'
        '表内误差为7层分别计算的输出相对Frobenius误差的平均，不是NMSE，也不是数据集精度。','',
        '| 方法 |10%输出误差 |20%输出误差 |10%Top10精确召回 |20%Top10精确召回 |使用Dense oracle |',
        '|---|---:|---:|---:|---:|---|']
    for method,label in LABELS.items():
        a,b=get(10,method),get(20,method)
        lines.append(f"| {label} | {100*a['relative_output_error']:.2f}% | {100*b['relative_output_error']:.2f}% | "
            f"{100*a['recall_top10']:.2f}% | {100*b['recall_top10']:.2f}% | {'是' if a['uses_dense_oracle'] else '否'} |")
    lines+=['','## 诊断结论与限制','',
        '- 旧版Mask逐bit复现，逐层输出误差复现到1e-5；不是更换输入/归一化后重新定义baseline。',
        '- 硬保留history不再用coarse gate删除任何传入边，但这组样本上的单独收益很小。'
        'history仍是明确选出的50%mandatory子集，不是把所有旧边永久累计。',
        '- 大小偏置使用未覆盖fine cells包含的真实token数量，仍不包含组内logit方差。'
        '精确mass按真实eligible成员的logsumexp排序；本例单独替换为精确mass没有降低输出误差，'
        '不能据聊天分析就认定coarse scoring是当前主因。',
        '- 冻结区域/历史后改为Dense最佳成员，隔离了region→Key这一步；'
        '生产修正版只在已选fine小块内部做局部QK和逐头softmax后平均，不读取Dense分布。',
        '- 单边影响力公式描述删除一条边的精确变化；按它选多个Key只是诊断heuristic，'
        '并不构成最优子集oracle。DenseTopK、influence均不可以当作快速生产实现。',
        '- 这组同协议结果中DenseTopK好于uniform，不能直接套用另一个实验的NMSE均值比较。'
        'TopK仍未达到最小输出误差；影响力排序在本例更好。',
        '- 四种改动存在交互作用，不能把误差简单拆成四项相加。半径2/4邻域召回不作为主要成功指标。',
        '- 所有模式保留旧A/B结构便于对照；生产修正版默认A，B为可选。'
        '20%下A/B几乎等效；10%下B仍有小幅误差收益。','',
        '## 真正稀疏递推的GPU检查','',
        '以下为非oracle修正版在真实Dense/sparse/Dense主干中运行的结果。'
        '局部输出误差仍以当前层相同输入的Dense输出为参照，召回目标则为原始Dense轨迹。'
        '已逐Query验证所有history都保留、specials保留、预算精确且无重复、首个稀疏层输入与Dense逐bit一致。'
        '最后一组pose重复运行逐bit一致。','',
        '| 预算 |结构 |局部输出误差 |原始DenseTop10召回 |局部选Key额外QK/Query |',
        '|---|---|---:|---:|---:|']
    native_summary=[]
    for budget in [10,20]:
        for variant in ['A','B']:
            setting=f'budget{budget}_{variant}';rows=[r for r in native['rows'] if r['setting']==setting]
            records=[r for r in native['records'][setting] if r['mode']=='sparse']
            row=dict(budget=budget,variant=variant,
                local_output_error=mean(r['local_output_error'] for r in rows),
                original_dense_top10_recall=mean(r['original_dense_top10_recall'] for r in rows),
                terminal_member_qk=mean(r['terminal_member_qk'] for r in records))
            native_summary.append(row)
            lines.append(f"| {budget}% | {variant} | {100*row['local_output_error']:.2f}% | "
                f"{100*row['original_dense_top10_recall']:.2f}% | {row['terminal_member_qk']:.1f} |")
    lines+=['','局部成员QK包含最终胜出Key，当前最终执行会再算一次，额外成本必须计入；'
        '还包含coarse/intermediate搜索和池化成本。这里没有速度提升、128帧或7Scenes精度结论。','',
        '## 逐层受控修正版B','',
        '| 层 |10%旧误差 |10%新误差 |20%旧误差 |20%新误差 |','|---|---:|---:|---:|---:|']
    for layer in range(10,17):
        values=[next(r['relative_output_error'] for r in all_rows if r['layer']==layer
            and r['retention']==budget/100 and r['method']==method)
            for budget in [10,20] for method in ['legacy','corrected_B']]
        lines.append(f'| {layer} | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
    new_masks=np.load(root/'native_two_frames/masks.npz')
    for budget in [10,20]:
        old_masks=np.load(ROOT/f'outputs/region_history_attention/chess_two_frames/budget{budget}_B/masks.npz')
        fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
        for row,layer in enumerate([10,13,16]):
            ids=[old_masks[f'native_h50_L{layer}_indices'],new_masks[f'budget{budget}_B_L{layer}']]
            masks=[]
            for x in ids:
                mask=np.zeros((2082,2082),bool);mask[np.arange(2082)[:,None],x]=True;masks.append(mask)
            difference=np.zeros((2082,2082),np.uint8)
            difference[masks[0]&masks[1]]=1;difference[masks[0]&~masks[1]]=2;difference[~masks[0]&masks[1]]=3
            for col,mask in enumerate(masks):
                axes[row,col].imshow(mask,cmap=ListedColormap(['#eeeeee','#144f83']),interpolation='nearest')
                axes[row,col].set_title(f'L{layer}: '+('legacy B' if col==0 else 'corrected B'),fontsize=11)
            axes[row,2].imshow(difference,cmap=ListedColormap(['#eeeeee','#39a65a','#df4545','#357ad7']),vmin=0,vmax=3,interpolation='nearest')
            axes[row,2].set_title('Green=both; red=old only; blue=new only',fontsize=9)
            for ax in axes[row]:
                ax.axhline(1040.5,color='cyan',lw=.5);ax.axvline(1040.5,color='cyan',lw=.5)
                ax.set_xticks([0,1041,2082]);ax.set_yticks([0,1041,2082]);ax.tick_params(labelsize=7)
        fig.suptitle(f'{budget}%budget; actual native trajectories; original Query/Key coordinates\nHard history + population bias + query-dependent local terminal Key',fontsize=12)
        filename=f'budget{budget}_old_new.png';fig.savefig(out/filename,dpi=170);plt.close(fig)
        lines += ['',f'![{budget}%old/new masks]({filename})']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(controlled=summary,native=native_summary),indent=2)+'\n')
    print(json.dumps(native_summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    run(p.parse_args().directory)
