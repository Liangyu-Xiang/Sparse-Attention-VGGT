import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/dense_sparse_comparison_20261104';OUT.mkdir(exist_ok=False,parents=True)
base=Path('/tmp/sparse-vggt-beta-gridmass-20261104');inp=ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json'
for method in ['dense','v8','p0','beta']:
 source=base if method=='beta' else ROOT
 cmd=[sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu','7','--monitor',str(OUT/f'{method}_monitor.json'),'--log',str(OUT/f'{method}.log'),'--',sys.executable,str(ROOT/'scripts/compare_dense_sparse_forward.py'),'--method',method,'--source',str(source),'--input',str(inp),'--output-dir',str(OUT/method)]
 code=subprocess.call(cmd,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1'))
 print(method,code,flush=True)
 if code:raise RuntimeError(method)
results={m:json.loads((OUT/m/'result.json').read_text()) for m in ['dense','v8','p0','beta']}
for m,d in results.items():d['dense_speedup']=results['dense']['median_seconds']/d['median_seconds'];d['dense_reason']=None;(OUT/m/'result.json').write_text(json.dumps(d,indent=2)+'\n')
(OUT/'completion.json').write_text(json.dumps(dict(complete=True,methods=list(results)),indent=2)+'\n')
