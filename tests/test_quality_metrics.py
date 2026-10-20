import json
from pathlib import Path

import pytest

from scripts.quality_metrics import quality_record, quality_cells, quality_markdown
from scripts.compare_three_method_quality import SCENES, report


def metrics():
    return dict(pose=dict(auc03=.2, auc30=.8),
                recon_unposed=dict(acc=.1, comp=.2, overall=.15, fscore=.6))


def test_units_and_required_fields():
    record = quality_record(metrics())
    assert record['complete']
    assert quality_cells(record) == ['20.0000', '80.0000', '0.1000', '0.2000', '0.1500', '60.0000']
    assert record['units']['cd'] == 'm'
    assert 'F1 %' in quality_markdown(record)


def test_missing_or_failed_tasks_are_not_successful_quality():
    record = quality_record({'pose': metrics()['pose']})
    assert not record['complete'] and record['values']['f1'] is None
    assert len(record['unavailable']) == 4
    failed = quality_record({}, 'OOM before inference completed')
    assert not failed['complete'] and len(failed['unavailable']) == 6
    assert 'OOM' in quality_markdown(failed)
    invalid = metrics()
    invalid['recon_unposed']['acc'] = float('inf')
    assert quality_record(invalid)['values']['acc'] is None


def write_result(path, scene):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(input=dict(image_paths=[scene], frame_indices=[0], shape=[128, 3, 392, 518], checkpoint='same'),
                frames=128, dataset='7scenes', metrics=metrics(), protocol=dict(saf3r_topk=128))
    path.write_text(json.dumps(data))


def test_complete_macros_and_input_audit(tmp_path):
    root, current = tmp_path / 'comparison', tmp_path / 'current'
    root.mkdir()
    for scene in SCENES:
        for directory in (root / 'dense', root / 'saf3r', current / 'region'):
            write_result(directory / scene / 'result.json', scene)
    assert report(root, current, 128)
    summary = json.loads((root / 'comparison.json').read_text())
    assert len(summary['macro']) == 3
    assert summary['macro'][0]['quality']['values']['cd'] == pytest.approx(.15)
    path = root / 'dense' / 'chess' / 'result.json'
    data = json.loads(path.read_text())
    data['input']['checkpoint'] = 'different'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Input/checkpoint mismatch'):
        report(root, current, 128)


def test_incomplete_method_never_gets_full_dataset_macro(tmp_path):
    root, current = tmp_path / 'comparison', tmp_path / 'current'
    root.mkdir()
    for scene in SCENES:
        write_result(current / 'region' / scene / 'result.json', scene)
    write_result(root / 'dense' / 'chess' / 'result.json', 'chess')
    assert not report(root, current, 128)
    summary = json.loads((root / 'comparison.json').read_text())
    assert [r['method'] for r in summary['macro']] == ['Ours History80']


def test_speedup_is_ratio_of_scene_mean_full_forward_times(tmp_path):
    root, current = tmp_path / 'comparison', tmp_path / 'current'
    root.mkdir()
    for i,scene in enumerate(SCENES):
        for mode,base in [('dense',10),('saf3r',7),('region',5)]:
            path=root/mode/scene/'result.json'
            write_result(path,scene)
            data=json.loads(path.read_text())
            data.update(median_seconds=base+i,median_peak_gib=12)
            path.write_text(json.dumps(data))
        write_result(current/'region'/scene/'result.json',scene)
    assert report(root,current,128)
    summary=json.loads((root/'comparison.json').read_text())
    macro={r['method']:r for r in summary['macro']}
    assert macro['VGGT Dense']['speedup_vs_vggt']==1
    assert macro['Ours History80']['speedup_vs_vggt']==pytest.approx(13/8)
    ours=next(r for r in summary['per_scene'] if r['scene']=='chess' and r['method']=='Ours History80')
    assert ours['speedup_vs_vggt']==2
