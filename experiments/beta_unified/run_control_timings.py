"""Same physicalGPU per scene, fresh processes, rotated method order."""
import concurrent.futures,hashlib,json,os,queue,subprocess,sys,tarfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FROZEN=Path('/tmp/sparse-vggt-quality-search-query128-20261106')
SCENES=['chess','fire','heads','office','pumpkin','redkitchen','stairs']
OUT=ROOT/'outputs/paired_fullforward_7scenes128_20261109'

def main():
    OUT.mkdir(exist_ok=True)
    history=ROOT/'outputs/saf3r_dino_top128_7scenes128_20261108/historical_fast'
    candidate=ROOT/'outputs/beta_unified_20261108/final7_quality/fast_subgroup/runs'
    expected=ROOT/'outputs/beta_unified_20261108/final7_quality/expected_inputs'
    ref=json.loads((history/'chess.json').read_text());audit={}
    for f,h in ref['source_sha256'].items():
        if f.startswith(('saf3r/patch/','saf3r/triton/','saf3r/models/official_vggt/','saf3r/utils/')):
            path=FROZEN/f;assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==h,f
            audit[f]=h
    (OUT/'quick_frozen_source_audit.json').write_text(json.dumps(audit,indent=2))
    common={}
    for p in (FROZEN/'saf3r/models/official_vggt').rglob('*.py'):
        rel=p.relative_to(FROZEN);assert p.read_bytes()==(ROOT/rel).read_bytes(),str(rel)
        common[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'common_official_model_audit.json').write_text(json.dumps(common,indent=2))
    plan=dict(scope='User authorized quick and originalDense timing-only; frozen changed candidate repeated to obtain same-batch comparisons. No quality reruns or parameter changes.',
        scenes=SCENES,frames=128,methods=['dense','quick','candidate'],GPUs=[4,5,6],
        protocol='1cold+3warm synchronized fullforward;fresh process each method;all predictionheads retained;eachscene samephysicalGPU;rotateorder byscene;sharedCuDNNsearchpolicy;no truth read',
        reference_quality='Reused separately;timing-only allsixfields unavailable',
        dense_definition='Original unpatched officialVGGT,defaultheadchunk8;no sparse installer',
        candidate_definition='Frozenfast_subgroup alpha0.5/tau0.5,gamma0.5,headchunk32;no modeling changes')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['saf3r/patch','saf3r/triton','experiments/beta_unified'] for p in (ROOT/folder).glob('*.py')}
    (OUT/'candidate_source_sha256.json').write_text(json.dumps(source,indent=2))
    with tarfile.open(OUT/'candidate_source.tar.gz','w:gz') as tar:
        for rel in source:tar.add(ROOT/rel,arcname=rel)
    lock=threading.Lock();done=[];running={};tasks=queue.Queue()
    for index,scene in enumerate(SCENES):tasks.put((index,scene))
    def status():
        p=OUT/'progress.tmp';p.write_text(json.dumps(dict(completed=done,running=running),indent=2));p.replace(OUT/'progress.json')
    env=dict(os.environ,LD_LIBRARY_PATH=str(Path(sys.prefix)/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
    def worker(gpu):
        while True:
            try:index,scene=tasks.get_nowait()
            except queue.Empty:return
            methods=['dense','quick','candidate'];order=methods[index%3:]+methods[:index%3]
            for method in order:
                folder=OUT/'runs'/scene/method
                if (folder/'result.json').exists():
                    result=json.loads((folder/'result.json').read_text())
                    with lock:done.append(dict(scene=scene,method=method,gpu=gpu,returncode=0,reused_completed_same_batch=True));status()
                    assert result['hardware']['physical_GPU']==str(gpu)
                    continue
                if folder.exists():
                    # Preserve failed input-audit attempts,never erase them.
                    attempt=OUT/'failed_attempts'/scene/method
                    attempt.parent.mkdir(parents=True,exist_ok=True);folder.rename(attempt)
                    for sub,suffix in [('logs','log'),('monitors','json')]:
                        old=OUT/sub/f'{scene}_{method}.{suffix}'
                        if old.exists():old.rename(attempt/f'{sub}.{suffix}')
                with lock:running[str(gpu)]=dict(scene=scene,method=method,order=order);status()
                repo=FROZEN if method=='quick' else ROOT
                reference=history/f'{scene}.json' if method!='candidate' else candidate/scene/'result.json'
                cmd=[sys.executable,str(ROOT/'experiments/logit_eth3d/run_gpu_exclusive.py'),'--gpu',str(gpu),
                    '--monitor',str(OUT/'monitors'/f'{scene}_{method}.json'),'--log',str(OUT/'logs'/f'{scene}_{method}.log'),'--',
                    sys.executable,'-u',str(ROOT/'experiments/beta_unified/time_controls.py'),'--method',method,'--repo',str(repo),
                    '--expected',str(expected/f'{scene}.json'),'--reference',str(reference),'--scene',scene,'--output',str(OUT/'runs'/scene/method)]
                cmd+=['--pixel-reference',str(candidate/scene/'processed_image.json')]
                start=time.monotonic();code=subprocess.call(cmd,cwd=repo,env=env)
                with lock:done.append(dict(scene=scene,method=method,gpu=gpu,returncode=code,wall_seconds=time.monotonic()-start));running.pop(str(gpu));status()
                print('DONE',scene,method,code,flush=True)
                if code:raise RuntimeError(f'Failed {scene}/{method}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,[4,5,6]))
    assert len(done)==21 and all(d['returncode']==0 for d in done)
    (OUT/'complete.json').write_text(json.dumps(dict(complete_timing=True,quality_rerun=False,runs=done),indent=2))

if __name__=='__main__':main()
