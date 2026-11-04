"""Repeat stage instrumentation after warming newly marked graphs; reuse separate trace."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from scripts.direct_detail_profile import Detail,save,csvout
from scripts.direct_detail_microbench import run
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,s=build(a.config);images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();rec=Detail(s,a.output_dir);rec.install(model);rec.enabled=True;begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();cold=time.perf_counter()-begin;rec.reset();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();wall=time.perf_counter()-begin;rows,operations=rec.finish()
old=json.loads((a.output_dir/'complete_forward.json').read_text()) if (a.output_dir/'complete_forward.json').exists() else {};base=json.loads((a.output_dir/'forward.json').read_text())['timing']['median_seconds'];totals={c:sum(r['exclusive_seconds'] for r in rows if r['category']==c) for c in 'ABC'};totals['A']+=max(0.,wall-sum(totals.values()));top=[dict(category=c,gpu_phase_seconds=totals[c],includes_children=False,scope='A includes wall residual; stage times include dispatch gaps and markers') for c in 'ABC'];csvout(a.output_dir/'top_level.csv',top);csvout(a.output_dir/'graph_overhead.csv',[r for r in operations if 'graph_' in r['operation']]);old.update(complete=True,baseline_warmed_median_seconds=base,instrumented_warmed_wall_seconds=wall,overhead_seconds=wall-base,overhead_percent=100*(wall/base-1),instrumented_initial_graph_compile_warmup_seconds=cold,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,top_level=top,graph_instrumentation='Graph cache rebuilt before diagnostic warmup; all newly captured selection graphs carry markers; independent earlier operator trace retained');save(a.output_dir/'complete_forward.json',old);run(rec,a.output_dir);print('STAGES_AND_MICROS_COMPLETE',wall,flush=True)
