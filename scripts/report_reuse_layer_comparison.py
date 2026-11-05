"""Write full-token cosine/L2 tables, scientific plots and matched quality evidence."""
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/reuse_layer_comparison_20261105'
sys.path.insert(0,str(ROOT))
METHODS=['main_independent','main_teacher_forced','fast_repair_independent','fast_repair_teacher_forced']
SCENES=['pipes','playground','courtyard']


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    flattened=[];summary={};quality=[]
    for scene in SCENES:
        summary[scene]={}
        for folder in sorted((BASE/scene).iterdir()):
            if not folder.is_dir():continue
            if (folder/'saved_quality/result.json').exists():
                q=json.loads((folder/'saved_quality/result.json').read_text())
                quality.append(q|dict(scene=scene,method=folder.name))
            path=folder/'layers.json'
            if not path.exists():continue
            rows=json.loads(path.read_text())
            for r in rows:
                for subset,m in r['subsets'].items():
                    # Independent consistency check on the accumulated scalar metrics.
                    identity=m['norm_ratio']**2+1-2*m['norm_ratio']*m['cosine']
                    assert abs(identity-m['relative_l2']**2)<2e-5,(folder,r['layer'],subset)
                    flattened.append(dict(scene=scene,method=folder.name,kind=r['kind'],layer=r['layer'],subset=subset,
                                          cosine=m['cosine'],relative_l2=m['relative_l2'],norm_ratio=m['norm_ratio'],tokens=m['tokens']))
            global_rows=[r for r in rows if r['kind']=='global_attention']
            if not global_rows:continue
            worst=max(global_rows,key=lambda r:r['subsets']['all']['relative_l2'])
            summary[scene][folder.name]=dict(layers=len(global_rows),worst_layer=worst['layer'],
                    max_relative_l2=worst['subsets']['all']['relative_l2'],
                    mean_layer_relative_l2=sum(r['subsets']['all']['relative_l2'] for r in global_rows)/len(global_rows),
                    min_cosine=min(r['subsets']['all']['cosine'] for r in global_rows))
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
        for name in METHODS:
            path=BASE/scene/name/'layers.json'
            if not path.exists():continue
            rows=[r for r in json.loads(path.read_text()) if r['kind']=='global_attention']
            for col,subset in enumerate(('all','camera')):
                axes[0,col].plot([r['layer'] for r in rows],[r['subsets'][subset]['cosine'] for r in rows],label=name)
                axes[1,col].plot([r['layer'] for r in rows],[r['subsets'][subset]['relative_l2'] for r in rows],label=name)
                axes[0,col].set_title(f'{scene}: {subset} tokens')
        for axis in axes.flat:axis.grid(alpha=.25)
        axes[0,0].set_ylabel('Cosine similarity vs original reuse')
        axes[1,0].set_ylabel('Relative L2 vs original reuse')
        for axis in axes[1]:axis.set_xlabel('Global layer (0-based)')
        axes[0,0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(BASE/f'{scene}_global_attention.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        for name in ('main_independent','fast_repair_independent'):
            path=BASE/scene/name/'layers.json'
            if not path.exists():continue
            rows=[r for r in json.loads(path.read_text()) if r['kind']=='frame_attention']
            for axis,metric in zip(axes,('cosine','relative_l2')):
                axis.plot([r['layer'] for r in rows],[r['subsets']['all'][metric] for r in rows],label=name)
                axis.set_ylabel(metric);axis.grid(alpha=.25)
        axes[0].legend();axes[0].set_title(f'{scene}: unchanged Frame attention under independent trajectories')
        axes[1].set_xlabel('Frame layer (0-based)');fig.tight_layout()
        fig.savefig(BASE/f'{scene}_frame_attention.png',dpi=160);plt.close(fig)
    with (BASE/'all_layers.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flattened[0]),lineterminator='\n');writer.writeheader();writer.writerows(flattened)
    subset_macro={}
    for name in ('original','main_independent','fast_repair_independent','original_cluster_count'):
        items=[q for q in quality if q['method']==name and q['scene'] in SCENES]
        if len(items)==3 and all(q['quality']['complete'] for q in items):
            subset_macro[name]={key:sum(q['quality']['values'][key] for q in items)/3 for key in items[0]['quality']['values']}
    (BASE/'summary.json').write_text(json.dumps(dict(layer_summary=summary,quality=quality,
         three_scene_subset_macro=subset_macro,subset_scope='3/11 ETH3D scenes, equal-scene mean; not full ETH3D'),indent=2)+'\n')
    lines=['# 原版 reuse 与当前方法：逐层对比', '',
           '固定同一 checkpoint、RGB 帧表、预处理张量；BF16、TF32 关闭、seed=0。只使用 GPU4/5/6。',
           '这是 ETH3D 三个诊断场景，不能称为全11场景数据集结果。全部 token 参与，不是采样点近似。',
           'Cosine = <current,reference>/(||current|| ||reference||)；相对 L2 = ||current-reference||/||reference||。',
           'Global attention 输出为 Wo 之后；另保存 Wo 之前、Global block 输入和 Frame attention 的对比。',
           'teacher_forced 每层替换 Global block 输入，但保留当前方法自己的历史链；它没有单独隔离历史误差。',
           '原版输出的接近程度是诊断指标，不是几何精度的替代品。Frame 的同输入反事实行也不代表 Frame 输入相同。', '',
           '## 同批保存预测的六指标（无额外模型前向）', '',
           'AUC/F1 为百分数，ACC/Comp/CD 为米；CD=(ACC+Comp)/2，非平方距离。', '',
           '| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for item in quality:
        v=item['quality']['values']
        cells=['N/A' if v[k] is None else f"{v[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in ('auc03','auc30','acc','comp','cd','f1')]
        lines.append('| '+item['scene']+' | '+item['method']+' | '+' | '.join(cells)+' |')
    lines+=['','### 3/11场景子集的等场景宏平均（不是全ETH3D）','',
            '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in subset_macro.items():
        cells=[f"{v[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in ('auc03','auc30','acc','comp','cd','f1')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['', '上述ETH3D诊断的计时及原Dense加速：N/A。逐层保存/比较不能与暖机完整前向速度比较。',
            '逐层诊断本身的六个几何指标为N/A并附原因；完整几何评测另存各运行的 saved_quality/。', '',
            '## Global 逐层 cosine / 相对L2', '', '每个单元格为 cosine / relative L2；层编号从0开始。']
    for scene in SCENES:
        lines+=['',f'### {scene}', '', '| Layer | '+' | '.join(METHODS)+' |', '|---:|'+'---:|'*len(METHODS)]
        values={}
        for name in METHODS:
            path=BASE/scene/name/'layers.json'
            values[name]={r['layer']:r['subsets']['all'] for r in json.loads(path.read_text()) if r['kind']=='global_attention'} if path.exists() else {}
        for layer in range(24):
            cells=[f"{values[name][layer]['cosine']:.6f} / {values[name][layer]['relative_l2']:.6f}" if layer in values[name] else 'N/A: incomplete' for name in METHODS]
            lines.append('| '+str(layer)+' | '+' | '.join(cells)+' |')
        lines+=['',f'![Global attention]({scene}_global_attention.png)', '',f'![Frame attention]({scene}_frame_attention.png)']
    speed_root=BASE/'speed_heads128'
    names=('dense','saf3r','main','original_cluster_count')
    if all((speed_root/name/'timing_result.json').exists() for name in names):
        from scripts.quality_metrics import quality_markdown
        results={name:json.loads((speed_root/name/'timing_result.json').read_text()) for name in names}
        dense=results['dense'];speed_rows=[]
        lines+=['','## 7Scenes heads128 配对完整前向计时','',
                '仅heads场景128帧；不是全7场景平均。GPU6独占，BF16，1次暖机+3次计时，中位数。',
                'SAF3R为原版Top4，lazy first-correspondence-head Top-K；不是论文DINO feature Top-K。',
                '包含初始化、聚类、路由、历史及重排；不包括图像加载和模型权重加载。', '',
                '| Method | Median full-forward s | Speedup vs Dense | Peak allocated GiB |',
                '|---|---:|---:|---:|']
        for name,item in results.items():
            for key in ('input','hardware','quality_protocol'):assert item[key]==dense[key],(name,key)
            for key in ('warmup','repeats','capture'):assert item['timing'][key]==dense['timing'][key],(name,key)
            monitor=json.loads((speed_root/(name+'_gpu.json')).read_text())
            assert monitor['gpu']==6 and not monitor['same_gpu_overlap_detected']
            seconds=item['timing']['median_seconds']
            ratio=dense['timing']['median_seconds']/seconds
            item['quality'].update(speedup_vs_vggt=ratio,speedup_unavailable=None)
            item['paired_dense_result']=str(speed_root/'dense/timing_result.json')
            (speed_root/name/'timing_result.json').write_text(json.dumps(item,indent=2)+'\n')
            (speed_root/name/'quality.json').write_text(json.dumps(item['quality'],indent=2)+'\n')
            (speed_root/name/'quality.md').write_text(quality_markdown(item['quality']))
            memory=max(s['peak_allocated_gib'] for s in item['timing']['samples'])
            lines.append(f'| {name} | {seconds:.6f} | {ratio:.4f}x | {memory:.4f} |')
            speed_rows.append(dict(method=name,median_seconds=seconds,speedup_vs_dense=ratio,peak_allocated_gib=memory))
        (speed_root/'paired_summary.json').write_text(json.dumps(dict(rows=speed_rows,scope='7Scenes heads128 only',
          gpu=6,relative_change_vs_main=results['original_cluster_count']['timing']['median_seconds']/results['main']['timing']['median_seconds']-1,
          relative_change_vs_saf3r=results['original_cluster_count']['timing']['median_seconds']/results['saf3r']['timing']['median_seconds']-1),indent=2)+'\n')
    failed=[]
    for path in BASE.glob('*/**/status.json'):
        status=json.loads(path.read_text())
        if status.get('stage')=='failed':failed.append((str(path.relative_to(BASE)),status.get('error')))
    if failed:
        lines+=['','## 失败诊断保留','']+[f'- {path}: {reason}；六指标N/A，未执行有效前向。' for path,reason in failed]
    (BASE/'report.md').write_text('\n'.join(lines)+'\n')
    print('Verified norm/cosine/L2 identities for',len(flattened),'full-token records')


if __name__=='__main__':main()
