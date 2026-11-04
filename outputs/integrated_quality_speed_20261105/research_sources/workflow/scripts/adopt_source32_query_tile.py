"""Adopt the fastest complete predeclared candidate; no test-truth parameter selection."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source32_query_tiles_20261105'
s=json.loads((OUT/'summary.json').read_text());assert s['complete'] and s['valid_quality_runs']==15
for label in ('operators','resource_probe','resource_tails'):
 assert json.loads((OUT/label/'result.json').read_text())['passed']
 monitor=json.loads((OUT/f'{label}.monitor.json').read_text());assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
winner=s['selected_by_full_forward_time'];plan=json.loads((OUT/'plan.json').read_text());path=Path(next(v['config'] for v in plan['variants'] if v['name']==winner));cfg=json.loads(path.read_text())
baseline=json.loads((OUT/'initial_default.json').read_text());allowed={'attention_block_queries','attention_stages','attention_warps'}
assert all(cfg[k]==baseline[k] for k in baseline if k not in allowed)
assert cfg['query_group_size']==256 and cfg['source_width']==32 and not cfg.get('history_split_export',False)
canonical=ROOT/'configs/evaluation/reuse_shared_value_loss_q256.json'
reference=canonical.with_name('reuse_shared_value_loss_q256_source32_bq32_reference.json')
if not reference.exists():reference.write_text(json.dumps(baseline,indent=2)+'\n')
canonical.write_text(json.dumps(cfg,indent=2)+'\n')
means={r['variant']:r for r in s['three_scene_means']};chosen=means[winner];base=means['bq32']
adoption=dict(selected_variant=winner,config=str(canonical),named_config=str(path),previous_config=str(reference),query_group_size=256,source_width=32,
 block_queries=cfg['attention_block_queries'],num_warps=cfg['attention_warps'],actual_num_stages=2 if cfg['attention_block_queries']==32 else cfg['attention_stages'],
 normal_full_forward_seconds=chosen['full_forward_seconds'],reference_bq32_seconds=base['full_forward_seconds'],speedup_vs_current_bq32=base['full_forward_seconds']/chosen['full_forward_seconds'],
 attention_main_seconds=chosen['attention_main_seconds'],reference_attention_main_seconds=base['attention_main_seconds'],
 selection_uses_test_truth=False,quality_evaluated=True,quality_caution=s['quality_caution'],dense_speedup=None,dense_speedup_reason=s['dense_speedup_reason'])
(OUT/'adoption.json').write_text(json.dumps(adoption,indent=2)+'\n')
print(json.dumps(adoption,indent=2))
