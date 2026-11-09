import json
import sys
from pathlib import Path

from evaluate_scannet500_beta import save

base, candidate = (Path(arg) for arg in sys.argv[1:])
def load(path):
    return {row['scene']: row for row in json.loads((path / 'summary.json').read_text())['per_scene']}
reference, optimized = load(base), load(candidate)
assert len(reference) == 20 and reference.keys() == optimized.keys()
directions = dict(auc03=1, auc30=1, acc=-1, comp=-1, cd=-1, f1=1)
rows = []
for scene, old in reference.items():
    new = optimized[scene]
    assert all(old.get(k) == new.get(k) and old.get(k) is not None for k in ('input', 'protocol')), scene
    assert old['quality']['complete'] and new['quality']['complete'], scene
    delta = {key: new['quality']['values'][key] - old['quality']['values'][key] for key in directions}
    rows.append(dict(scene=scene, actual_frames=new['actual_frames'], delta=delta,
                     baseline_seconds=old['timing']['median_seconds'],
                     candidate_seconds=new['timing']['median_seconds']))
macro = {key: sum(r['delta'][key] for r in rows) / len(rows) for key in directions}
full = [r for r in rows if r['actual_frames'] == 500]
seconds = sum(r['candidate_seconds'] for r in full) / len(full)
# Historical first-20 campaign, restricted to the 19 actual 500-frame scenes.
# The four-scene speed diagnostic has a separate historical Dense reference.
historical_dense = 117.29282471802281
speed_target = 4.5
quality_pass = all(macro[key] * direction >= -1e-8 for key, direction in directions.items())
save(candidate / 'acceptance.json', dict(
    macro_delta=macro, quality_pass=quality_pass,
    tolerance=1e-8, full_500_scenes=len(full), mean_seconds=seconds,
    historical_dense_seconds=historical_dense, historical_dense_speedup=historical_dense / seconds,
    historical_speed_target=speed_target,
    historical_speed_target_pass=historical_dense / seconds >= speed_target,
    paired_dense_speedup=None, paired_dense_unavailable='No contemporaneous Dense run',
    per_scene=rows))

labels = {
    'auc03': 'AUC@3 (%)', 'auc30': 'AUC@30 (%)',
    'acc': 'ACC (m)', 'comp': 'Comp (m)', 'cd': 'CD (m)', 'f1': 'F1 (%)',
}
lines = [
    '# ScanNet50 前 20 序列质量验收', '',
    '质量宏平均对 20 个场景等权。AUC 和 F1 的差值以百分点计，距离以米计。',
    '速度只汇总实际输入 500 帧的场景。Dense 倍率为历史参考，本批配对倍率为 N/A。', '',
    '| 指标 | 冻结稀疏参考 | CUDA 候选 | 候选 − 参考 |',
    '|---|---:|---:|---:|',
]
for key, label in labels.items():
    unit = 100 if key in ('auc03', 'auc30', 'f1') else 1
    old = sum(r['quality']['values'][key] for r in reference.values()) / len(rows)
    new = sum(r['quality']['values'][key] for r in optimized.values()) / len(rows)
    lines.append(f'| {label} | {old*unit:.6f} | {new*unit:.6f} | {macro[key]*unit:+.6f} |')
lines.extend([
    '', f"六项质量不下降检查 {'通过' if quality_pass else '未通过'}。",
    f'{len(full)} 个实际 500 帧场景平均前向 {seconds:.4f} s。'
    f'历史 Dense 参考 {historical_dense:.4f} s，估算 {historical_dense/seconds:.4f}×。',
    '', '| 场景 | 帧数 | ' + ' | '.join(labels.values()) + ' |',
    '|---|---:|' + '---:|' * len(labels),
])
for row in rows:
    cells = [f'{row["delta"][key] * (100 if key in ("auc03","auc30","f1") else 1):+.6f}' for key in labels]
    lines.append('| ' + ' | '.join([row['scene'], str(row['actual_frames']), *cells]) + ' |')
(candidate / 'acceptance.md').write_text('\n'.join(lines) + '\n')
