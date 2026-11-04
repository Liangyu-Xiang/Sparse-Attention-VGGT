"""Aggregate frozen-input state-loss diagnostics, with six unavailable quality fields."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_record,quality_markdown


def pooled(rows,subset):
    scores=[r['similarity'][subset] for r in rows if subset in r['similarity']]
    return dict(relative_l2=math.sqrt(sum(s['error_sum_sq'] for s in scores)/max(1e-30,sum(s['dense_sum_sq'] for s in scores))),
        cosine=sum(s['cosine_sum'] for s in scores)/max(1,sum(s['queries'] for s in scores)),
        queries=sum(s['queries'] for s in scores))


def summarize(rows):
    return dict(**{s:pooled(rows,s) for s in ['all','camera','probe_non_camera']},
        state_fractions={s:sum(r['state_fractions'][s] for r in rows)/len(rows) for s in ['0','1','5','10']},
        history_key_reuse_fraction=sum(r['history_key_reuse_fraction'] for r in rows)/len(rows),
        density=sum(r['density'] for r in rows)/len(rows),group_layers=len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    all_results={};csv_rows=[]
    q=quality_record({},'Frozen-input attention diagnostic only; pose and recon_unposed were not evaluated')
    for folder in args.runs:
        path=folder/'loss_candidates.json'
        rows=json.loads(path.read_text() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes()))
        scene=folder.name;all_results[scene]={}
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['method']==method]
            if set(r['layer'] for r in selected)!=set(range(24)):raise ValueError(f'Incomplete {scene}/{method}')
            result=dict(all_layers=summarize(selected),middle_10_16=summarize([r for r in selected if 10<=r['layer']<=16]),
                layers={str(l):summarize([r for r in selected if r['layer']==l]) for l in range(24)},quality=q)
            all_results[scene][method]=result
            path=args.output/scene/method;path.mkdir(parents=True,exist_ok=True)
            (path/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            (path/'quality.md').write_text(quality_markdown(q))
            for layer,s in result['layers'].items():
                csv_rows.append(dict(run=scene,method=method,layer=int(layer),density=s['density'],history_key_reuse=s['history_key_reuse_fraction'],
                    **{f'state_{k}':v for k,v in s['state_fractions'].items()},
                    **{f'{subset}_{metric}':s[subset][metric] for subset in ['all','camera','probe_non_camera'] for metric in ['cosine','relative_l2']}))
    (args.output/'results.json').write_text(json.dumps(all_results,indent=2)+'\n')
    with (args.output/'layers.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
    joint_mode=all(m.startswith('joint/') for methods in all_results.values() for m in methods)
    scope=('本组固定B语义、概率、门控和特征，仅比较可分控制与三种联合输出目标；每次最多32次单簇坐标更新，是离线原型。' if joint_mode else
        '34 个候选同时改变了损失/权重、History 语义、门控和未知概率假设；跨系列结果不能归因于单一损失变化。')
    lines=['# 状态损失筛选：固定输入特征诊断','','本报告只比较同一基准稀疏轨迹上的候选 mask，不是独立推理或完整质量评测。',
        '中间层索引为 10–16（从 0 开始）；相对 L2 以累计平方误差/累计 Dense 能量开方计算，余弦按查询数加权。',
        '输出经过实际 Wo 投影；mask 选择不使用当前成员 QK，Dense QK 仅在 mask 确定后用于测量。',
        scope,
        '20% 是包含始终保留的 Special Key 在内的总 Key 预算上限；patch 预算为 floor(0.2 × 总 token 数) − Special 数量。表内密度为总 Key 使用量/总 token 数。',
        'SAF3R Top-K：N/A，本诊断为 VGGT 全局注意力补丁；输入/阈值/硬件/采样清单见各原始运行目录。',
        '各候选共享基准特征和历史输入，无法分别测量完整运行时间/峰值显存；原始 runtime.json 为整次带插桩诊断。',
        '相对 VGGT Dense 端到端加速比：N/A，未做配对预热全前向计时。','','每个候选的六项质量字段均为 N/A：未执行 pose/recon_unposed；详见独立 quality.md。','']
    for scene,methods in all_results.items():
        lines += [f'## {scene}：中间层 10–16','','| 候选 | 普通查询 RelL2 % | 余弦 | Camera RelL2 % | 余弦 | Drop % | Rep % | History % | Exact % | 历史 Key 复用 % | 实际密度 % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method,result in sorted(methods.items(),key=lambda item:item[1]['middle_10_16']['probe_non_camera']['relative_l2']):
            s=result['middle_10_16'];p=s['probe_non_camera'];c=s['camera'];st=s['state_fractions']
            lines.append(f"| {method} | {p['relative_l2']*100:.2f} | {p['cosine']:.5f} | {c['relative_l2']*100:.2f} | {c['cosine']:.5f} | "+' | '.join(f'{st[k]*100:.2f}' for k in ['0','1','5','10'])+f" | {s['history_key_reuse_fraction']*100:.2f} | {s['density']*100:.3f} |")
        lines += ['']
    (args.output/'report.md').write_text('\n'.join(lines).rstrip()+'\n')


if __name__=='__main__':main()
