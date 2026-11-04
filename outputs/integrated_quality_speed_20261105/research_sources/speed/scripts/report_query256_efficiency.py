"""Report measured pairs and a closed ledger from each SAME statistical forward."""
from pathlib import Path
import csv,hashlib,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import FIELDS,LABELS,quality_cells
from scripts.report_deep_saf3r_query256 import build_tree,quantize,leaves,combine,find
OUT=ROOT/'outputs/query256_efficiency_20261105';SCENES=('heads','chess','stairs');METHODS=('baseline','optimized')
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def check_inputs(a,b):
    for k in ('image_paths','frame_indices','shape'):assert a['input'][k]==b['input'][k],k
    assert Path(a['input']['checkpoint']).resolve()==Path(b['input']['checkpoint']).resolve()
    assert a['hardware']==b['hardware']
    if 'quality_protocol' in a:assert a['quality_protocol']==b['quality_protocol']
def main():
    normal={s:{m:read(OUT/s/m/'result.json') for m in METHODS} for s in SCENES}
    profiles={s:{m:read(OUT/s/(m+'_profile')/'result.json') for m in METHODS} for s in SCENES}
    lines=['# Query-256 implementation efficiency: measured pairs',
        '', 'Only heads/chess/stairs, 128 uniform frames each; three-scene equal-weight average, not a complete 7Scenes macro. All normal pairs use the same physical GPU per scene. Dense and SAF3R controls were not rerun; speedup relative to Dense is N/A (no paired Dense timings).',
        '', 'Algorithm: group=256, 20% total Key retention including all special Keys, adaptive_local, two clustering updates, original cluster count, 25% candidate fraction, four-state Value loss, 16 budget iterations, actual history Query interval=256. Checkpoint/preprocessing/heads/pose/recon_unposed protocols are unchanged. This Shared Value variant does not use the SAF3R Top-K selector; the loaded model configuration has that patch removed. The historical SAF3R profile uses lazy first-correspondence-head Top-4, not paper DINO Top-K.',
        '', 'Optional efficient config: `configs/evaluation/reuse_shared_value_loss_q256_efficient.json`. Frozen actual baseline, including previously uncommitted files: `frozen_baseline/` and `frozen_baseline.tar.gz`; hashes in `frozen_sha256.json`, original diff/status retained. No checkout of d7f535ca was substituted.',
        '', 'Actual frozen direct_sampled execution uses 32-row tiles, 128 Keys and 2 pipeline stages: this call path did not forward the config fields attention_block_keys=64 or attention_stages=1. The efficient path uses 64-row indexed attention tiles, 128 Keys per tile, 4 warps, 1 pipeline stage; head-specific weights and outputs remain independent. No per-group K/V copies are materialized. Final head-normalized log-mean history exports use contiguous special/patch partitions of one allocation; evidence and next-layer consumption remain enabled. Shared Value geometry has at least 64 cells per cluster; a single masked fixed cluster list replaces data-dependent population launch bins. Width expands above 64 when necessary; the original non-reusable fallback captures changed structures. Full structural signatures include shapes, strides, dtype, device and captured scalars; values are copied before replay. LRU limits: 4 entries and 2048 MiB of retained live Torch allocation deltas; private-pool reserved memory is reported separately via process peak reserved memory. A capture can transiently exceed the cache limit and is then not retained.',
        '', 'Execution parameters were selected from fixed synthetic Q/K/V/support timing, without 7Scenes truth. Larger tiles change floating-point execution/reduction order. Local operator errors and full quality effects are reported; this is not a claim of bit-identical model trajectories.',
        '', '## Normal speed and memory (no statistical probes)', '', '| Scene | Baseline s | Optimized s | Baseline / optimized | Reduction | Baseline peak alloc GiB | Optimized peak alloc GiB | Baseline peak reserved GiB | Optimized peak reserved GiB | vs Dense |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    times={};quality_rows=[];raw=[]
    for s in SCENES:
        a,b=(normal[s][m] for m in METHODS);check_inputs(a,b)
        mons=[read(OUT/f'{s}_{m}_monitor.json') for m in METHODS]
        assert mons[0]['gpu_uuid']==mons[1]['gpu_uuid']
        assert all(not x['same_gpu_overlap_detected'] and x['returncode']==0 for x in mons)
        aa,bb=[x['timing']['median_seconds'] for x in (a,b)];times[s]=(aa,bb)
        memory=[max(row[k] for row in x['timing']['samples']) for k in ('peak_allocated_gib','peak_reserved_gib') for x in (a,b)]
        lines.append(f'| {s} | {aa:.6f} | {bb:.6f} | {aa/bb:.3f}x | {(1-bb/aa)*100:.2f}% | '+' | '.join(f'{v:.3f}' for v in memory)+' | N/A |')
        for m,x in normal[s].items():
            assert x['quality']['complete'];v=x['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
            quality_rows.append((s,m,x['quality']))
            raw.append(dict(scene=s,method=m,timing=x['timing'],quality=x['quality'],metrics=x['metrics'],hardware=x['hardware'],quality_protocol=x['quality_protocol'],gpu_uuid=mons[METHODS.index(m)]['gpu_uuid']))
    at,bt=[sum(times[s][i] for s in SCENES)/3 for i in (0,1)]
    lines+= [f'| Three-scene mean | {at:.6f} | {bt:.6f} | {at/bt:.3f}x | {(1-bt/at)*100:.2f}% | — | — | — | — | N/A |', '', 'One warmup + three synchronized full forwards per method/scene. Medians use warmed wall times and include reset, initialization, clustering, Graph creation if needed, input copies, routing, history and all model heads. Peaks are maxima over the three timed samples. Cold samples and raw timing repeats are retained in each result.json. Quality evaluates the final timed prediction.', '', '## Quality', '', '| Scene | Method | '+' | '.join(LABELS)+' | vs Dense |','|---|---|'+'---:|'*6+'---|']
    for s,m,q in quality_rows:lines.append('| '+' | '.join([s,m,*quality_cells(q),'N/A'])+' |')
    for m in METHODS:
        q=dict(values={k:sum(normal[s][m]['quality']['values'][k] for s in SCENES)/3 for k in FIELDS})
        lines.append('| '+' | '.join(['Three-scene equal mean',m,*quality_cells(q),'N/A'])+' |')
    lines+=['', 'AUC and F1 are percentages in the table (raw JSON fractions). ACC/Comp/CD are metres, unsquared; CD=(ACC+Comp)/2. F1 threshold=0.05 m; reconstruction downsampling=0.0078125 m. Raw pose/recon_unposed metrics and alignment/evaluation metadata are in per-run result.json and source snapshots. These are local measured rows; no published scores or historical timings are used as matched baselines.', '', '| Optimized minus baseline | AUC@3 pp | AUC@30 pp | ACC m | Comp m | CD m | F1 pp |','|---|---:|---:|---:|---:|---:|---:|']
    deltas={}
    for s in SCENES:
        delta={k:normal[s]['optimized']['quality']['values'][k]-normal[s]['baseline']['quality']['values'][k] for k in FIELDS};deltas[s]=delta
        lines.append('| '+s+' | '+' | '.join(f'{delta[k]*(100 if k in ("auc03","auc30","f1") else 1):+.6f}' for k in FIELDS)+' |')
    lines+=['', 'Quality differences include improvements and regressions and must be assessed per scene; this optimization has not established unchanged end-to-end quality. No parameter was changed after inspecting test quality.', '', '## Separate statistical forwards', '', 'For each scene/method choose the median-wall pass among three instrumented passes, then take every interval, Graph count, memory peak and leaf from that one pass. Do not add these parents to their children or mix them with normal speed measurements. CUDA Event intervals include submission gaps; Graph device timestamp instrumentation perturbs execution. Displayed integer-microsecond rounding is balanced in the explicit remainder; a signed 1–2 microsecond remainder denotes rounding, not negative work.', '', '| Scene | Method | Pass | Wall s | Leaf sum s | Peak alloc GiB | Peak reserved GiB | Graph hit | Created | Evicted | Constants updated | Retained live MiB |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    trees={m:[] for m in METHODS};stat=[]
    for s in SCENES:
        check_inputs(*[profiles[s][m] for m in METHODS])
        for m,d in profiles[s].items():
            check_inputs(d,normal[s][m])
            monitor=read(OUT/f'{s}_{m}_profile_monitor.json')
            assert monitor.get('returncode')==0 and not monitor['same_gpu_overlap_detected']
            assert monitor['gpu_uuid']==read(OUT/f'{s}_{m}_monitor.json')['gpu_uuid']
            p=sorted(enumerate(d['event_profiles']),key=lambda x:x[1]['wall_seconds'])[1];index,p=p
            tree=quantize(build_tree(d,p));flat=leaves(tree);assert sum(x[1] for x in flat)==tree.us;trees[m].append(tree)
            with (OUT/s/(m+'_profile')/'selected_leaf_ledger.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['path','microseconds','kind']);writer.writerows(flat)
            def serialize(n):return dict(name=n.name,seconds_raw=n.seconds,integer_microseconds=n.us,kind=n.kind,children=[serialize(c) for c in n.children])
            save(OUT/s/(m+'_profile')/'selected_tree.json',dict(pass_index=index,tree=serialize(tree),leaf_sum_us=sum(x[1] for x in flat)))
            gc=p['graph_cache'];live=gc.get('live_tensor_bytes');live='N/A' if live is None else f'{live/2**20:.2f}'
            lines.append(f'| {s} | {m} | {index} | {tree.us/1e6:.6f} | {sum(x[1] for x in flat)/1e6:.6f} | {p["peak_allocated_gib"]:.3f} | {p["peak_reserved_gib"]:.3f} | {gc["hits"]} | {gc.get("creations",gc["misses"])} | {gc["evictions"]} | {gc["constant_updates"]} | {live} |')
            stat.append(dict(scene=s,method=m,pass_index=index,wall_seconds=p['wall_seconds'],graph_cache=gc,leaf_sum_us=sum(x[1] for x in flat),wall_us=tree.us))
    means={m:quantize(combine(trees[m])) for m in METHODS}
    lines+=['', '### Statistical stage means (parents, not additive with expanded leaves)', '', '| Stage | Baseline s | Optimized s |','|---|---:|---:|']
    a,b=[means[m] for m in METHODS]
    assert [x.name for x in a.children]==[x.name for x in b.children]
    for x,y in zip(a.children,b.children):lines.append(f'| {x.name} | {x.us/1e6:.6f} | {y.us/1e6:.6f} |')
    lines.append(f'| Full wall | {a.us/1e6:.6f} | {b.us/1e6:.6f} |')
    for m in METHODS:
        flat=leaves(means[m]);assert sum(x[1] for x in flat)==means[m].us
        with (OUT/f'leaf_ledger_three_scene_mean_{m}.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['path','microseconds','kind']);w.writerows(flat)
    lines+=['', '### Attention / history / Graph detail (overlapping parent/child rows)', '', '| Stage | Baseline s | Optimized s |','|---|---:|---:|']
    attention=['Global attention']
    support=attention+['支持集构建']
    selector=support+['prior、损失、预算与打包','Selector与Graph管理']
    paths=[attention+['支持集上的attention执行'],attention+['支持集上的attention执行','索引式融合QK、softmax、AV及inline采样'],attention+['支持集上的attention执行','独立历史概率导出'],attention+['历史概率统计与补全'],support,support+['Value聚类'],support+['池化与共享几何'],selector,selector+['Graph缓存miss时创建与捕获'],selector+['跨层几何常量拷贝'],selector+['每窗口prior等输入拷贝'],selector+['CUDA Graph重放'],selector+['CUDA Graph重放','Graph内四状态选边计算','四状态误差、成本和目标']]
    for path in paths:
        x,y=[find(means[m],path) for m in METHODS]
        lines.append(f'| {path[-1]} | {x.us/1e6:.6f} | {y.us/1e6:.6f} |')
    growth=read(OUT/'validation/graph_growth.json');assert growth['passed']
    operators=read(OUT/'validation/operators.json')
    assert operators['passed']
    maxima={k:max(c[k] for c in operators['cases']) for k in operators['cases'][0] if k.endswith('_max_abs')}
    lines+=['', '## Validation and reproducibility', '', 'Fixed-tensor maximum absolute errors versus the original 32-row execution: '+', '.join(f'{k}={v:.9g}' for k,v in maxima.items())+'.', '', 'Measured Graph replay can become slower with fixed-width masked work even while capture removal lowers total selection time. History export layout optimization is included, but its independent timing benefit is not established. Per-stage values are in the closed ledgers.', '', 'Baseline operator/reset tests: 6 passed. Final tile/tail/special/invalid-slot, actual per-head probability reference, duplicate/out-of-range diagnostic and 24-layer reset suite: 17 passed. Synthetic Graph tests update priors, historical IDs, eligibility, Values, labels, representatives, and geometry; compare exact selected IDs/counts/weights against eager; vary actual cluster populations and their original launch bins; verify stable buffer addresses, new-input serial updates, structural recapture and LRU eviction. The separate graph_growth.json test grows population to 105 and width from 64 to 128, confirming exact supports/statistics against eager and a new fallback capture without truncation. See validation/operators.json for numeric errors and counters, validation/tiles.json for all synthetic tile candidates. Every diagnostic has all six unavailable quality fields with a reason and no Dense speed claim.', '', 'The unrelated installed `benchmarks` package shadows this repository namespace. `launch_local_evaluation.py` selects the repository namespace for both variants without changing evaluation code. The failed import-only attempt is preserved in failed_import_attempt; no model forward ran in that attempt. GPU lease monitor polls every 0.5 s and aborts on foreign contexts during timing; polling cannot rule out a context shorter than the polling interval.', '', 'Reproduce from repository root using the archived sparse-vggt environment (Torch/CUDA versions in results):', '', '```bash', '/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python -m scripts.validate_query256_efficiency', '/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_efficiency.py --variant baseline', '/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_efficiency.py --variant optimized', '/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_efficiency.py --variant baseline --profile', '/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_efficiency.py --variant optimized --profile', '/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/report_query256_efficiency.py', '```', '', 'The driver resumes completed results and preserves incomplete outputs; use a fresh experiment directory for a new measurement. Explicit per-run commands appear in process monitor/log provenance. Heads/stairs use GPU 0 and chess GPU 1; only idle devices are leased. Source hashes/snapshots, checkpoint identity, resolved symlink manifest, exact input frame indices and image paths, configs, raw quality, raw repeats and separate statistical profiles are archived under this experiment directory.']
    save(OUT/'summary.json',dict(complete=True,normal=raw,deltas_optimized_minus_baseline=deltas,statistical=stat,three_scene_time_mean=dict(baseline=at,optimized=bt,ratio=at/bt),scope='three named scenes; not full 7Scenes macro',dense_speedup=None,dense_speedup_reason='Unchanged Dense control not authorized for rerun; no paired Dense batch'))
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
