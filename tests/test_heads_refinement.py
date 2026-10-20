from argparse import Namespace
from pathlib import Path

from scripts.refine_heads_ablation import PARTS,intervention,child


def test_substages_partition_only_previously_failing_middle_and_deep():
    flattened=[layer for part in PARTS for layer in intervention(part)]
    assert sorted(flattened)==list(range(10,24))
    assert len(set(flattened))==len(flattened)


def test_child_passes_accuracy_only_settings_without_modifying_source(monkeypatch,tmp_path):
    import scripts.evaluate_attention_first_residual_value as evaluator
    import saf3r.patch.av_stage_ablation as ablation
    captured={}
    monkeypatch.setattr(ablation,'STAGES',dict(ablation.STAGES))
    monkeypatch.setattr(evaluator,'run',lambda args:captured.update(vars(args)))
    child(Namespace(part='deep_21_23',output_dir=tmp_path))
    assert ablation.STAGES['deep_21_23']==(21,22,23)
    assert captured['active_stage']=='deep_21_23'
    assert captured['warmup']==0 and captured['repeats']==1
    assert captured['frames']==128 and captured['mode']=='old'
    assert captured['tasks']==['pose','depth','recon_unposed']
    assert captured['baseline_result'] is None
