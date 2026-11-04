"""Separate full-forward A/B, isolated-window timing, and Graph equivalence checks."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source_local_quality_20261105'
scenes=('heads','chess','stairs');rows=[]
for scene in scenes:
 pair=[]
 for mode in ('graph','eager'):
  d=json.loads((OUT/scene/f'adaptive_{mode}/result.json').read_text())
  pair.append(d['timing']['median_seconds'])
  rows.append(dict(scene=scene,execution=mode,seconds=pair[-1],quality=d['quality'],warmed_samples=d['timing']['samples']))
mean={mode:statistics.mean(r['seconds'] for r in rows if r['execution']==mode) for mode in ('graph','eager')}
window=json.loads((OUT/'graph_window_diagnostic/selector.json').read_text())
capture=json.loads((OUT/'graph_window_capture_breakdown/selector.json').read_text())
equivalence={}
for scene in scenes:
 d=json.loads((OUT/f'graph_equivalence_{scene}/equivalence.json').read_text())
 equivalence[scene]=dict(windows=d['windows'],support_difference_windows=sum(bool(r['checks']['ids']) for r in d['rows']),eager_repeat_difference_windows=d['eager_nondeterministic_windows'],diagnostic_difference_windows=d['diagnostic_difference_windows'],semantic_differences=[dict(layer=r['layer'],**r.get('semantic_support_difference',{})) for r in d['rows'] if r['checks']['ids']])
lines=['# CUDA Graph selector audit','', '## Same-GPU full-forward A/B','', '| Scene | Graph s | Eager s | Graph / eager − 1 |','|---|---|---|---|']
for scene in scenes:
 g=next(r['seconds'] for r in rows if r['scene']==scene and r['execution']=='graph');e=next(r['seconds'] for r in rows if r['scene']==scene and r['execution']=='eager')
 lines.append(f'| {scene} | {g:.6f} | {e:.6f} | {(g/e-1)*100:+.2f}% |')
lines += [f"| Three-scene mean | {mean['graph']:.6f} | {mean['eager']:.6f} | {(mean['graph']/mean['eager']-1)*100:+.2f}% |",'',
 'Both execute the same `reuse_window_graph.eager()` selector mathematics; eager bypasses Window capture and copies. Full-forward includes every layer and output head. All six quality fields are measured in the corresponding run and included in report.md and the raw summary below. Separate forwards are not bitwise paired outputs: clustering and loss aggregation contain floating-point atomic reductions. No Dense rerun; Dense speedup N/A with the reason recorded in each run.','',
 '## Isolated real layer-1 window','',f"heads, 128 frames; {window['groups']} Query groups, {window['clusters']} clusters; geometry width {window['width']}. Warmed median of seven repetitions. These are isolated selector timings, not full-forward results and not a per-forward ledger.",'', '| Operation | ms |','|---|---|']
for name,label in [('eager','Eager selector'),('pure_graph_replay','Pure replay'),('graph_with_input_and_record_copies','Replay + current inputs and record copies'),('geometry_update','Layer geometry update (separate)')]:lines.append(f"| {label} | {window[name]['median_seconds']*1000:.6f} |")
lines += ['',f"Per-window wrapper speedup {window['full_wrapper_speedup']:.3f}x; this is not speedup vs Dense. Geometry update is once per graph/layer when needed, not once per window. Capture overhead is measured separately below.",'',
 '## One real Window creation/capture: closed wall-time ledger','', '| Operation | Seconds |','|---|---|']
total=round(capture['creation_and_capture_seconds']*1e6);used=0
for name in ('gc_collect','synchronize','empty_cache'):
 value=round(capture['creation_stages_seconds'][name]*1e6);used+=value;lines.append(f'| {name} | {value/1e6:.6f} |')
lines += [f'| Allocation, copies, warm execution, capture and remaining boundaries | {(total-used)/1e6:.6f} |',f'| **Total: exact sum of rows** | **{total/1e6:.6f}** |','',
 'The installed PyTorch `torch.cuda.graph.__enter__()` explicitly runs synchronize(), gc.collect(), empty_cache() before capture_begin(). This measured constructor is already preceded by eager kernel warmups. It includes the Window warm run, static-buffer setup and graph capture. The garbage-collection call is host work, not part of the selection algorithm. Tiny wrapper instrumentation overhead remains in the remainder.','',
 '## Why warmed full forwards still create graphs','',
 'The active configuration uses static_shared_graph=False. Reusable cache keys include Query-window shape, geometry width and power-of-two capacities for population bins. Current clustering changes these structures; the 24-entry cache can also evict old entries. Misses therefore persist after warmup. A new Window clones large static tensors, runs a warm selector and enters PyTorch capture, including Python garbage collection. Each replay also copies current per-window inputs and exported records; each layer updates geometry. Replaying the kernels is faster in the isolated test, but these costs can outweigh launch savings.','',
 '| Scene | Warmed Graph creations | Warmed evictions |','|---|---|---|']
for scene in scenes:
 ss=next(r['warmed_samples'] for r in rows if r['scene']==scene and r['execution']=='graph')
 lines.append(f"| {scene} | {[s['graph_counters']['graph_cache_creations'] for s in ss]} | {[s['graph_counters']['graph_cache_evictions'] for s in ss]} |")
lines += ['', 'Creation counters above refer to the reusable-cache branch. Do not use them to count fallback per-layer graphs when geometry width exceeds 64. Do not add isolated-window measurements to full-forward times or subtract old instrumented ledger numbers from these normal timings.','',
 '## Equivalence limits','', '| Scene | Windows | Graph/eager differing support windows | Eager/eager differing support windows |','|---|---|---|---|']
for scene,d in equivalence.items():lines.append(f"| {scene} | {d['windows']} | {d['support_difference_windows']} | {d['eager_repeat_difference_windows']} |")
lines += ['', 'These checks feed Graph and eager the same current tensors at every window of a real forward. Packed positional differences can be large when only a few Keys are inserted/deleted and later entries shift. Chess/stairs semantic set comparisons show 1–10 added/removed Keys in affected windows; heads in the first diagnostic did not save set-level counts.','',
 'Current code is not strictly bitwise equivalent. `reuse_shared_value_states._corrections` uses relaxed float32 atomic_add for DELTA_D, DELTA_R and KEPT_D; summation order can vary even across two eager calls, and small error changes can cross discrete budget boundaries. The observed eager/eager differences demonstrate nondeterminism; they do not prove every Graph/eager difference has that sole cause. Most windows also differ in state diagnostics; history and exact can produce identical supports when all members are historical eligible. Equal supports alone do not establish identical internal states.','',
 'Graph replay does not intentionally change priors, history or the objective. Strict reproduction requires a separate investigation of deterministic reductions and tied-state/budget decisions, without silently introducing a tolerance or changing the algorithm. Full quality differences in the A/B cannot be assigned causally to Graph alone because the preceding clustering differs across separate forwards.','',
 'Implementation optimization should reduce capture misses using stable shapes/runtime counts and reuse buffers, then compare unchanged selector math. Avoid changing loss, retention, Query group or history to make a speed result look better. The concurrent efficient-Graph branch was not enabled in this clustering experiment.','',
 'All microbenchmarks and equivalence runs are diagnostics with six quality fields unavailable for the explicit reason; the normal A/B runs have all six measured fields. See per-run quality.json and report.md.']
(OUT/'graph_report.md').write_text('\n'.join(lines)+'\n')
(OUT/'graph_summary.json').write_text(json.dumps(dict(full_forward=rows,three_scene_mean_seconds=mean,window=window,capture=capture,equivalence=equivalence),indent=2)+'\n')
print('Graph report saved')
