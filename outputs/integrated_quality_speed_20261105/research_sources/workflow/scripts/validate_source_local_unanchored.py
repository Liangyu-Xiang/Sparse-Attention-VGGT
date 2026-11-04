"""CPU invariant checks, not a camera/reconstruction quality evaluation."""
import importlib.util
import json
from pathlib import Path
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_local',ROOT/'saf3r/patch/source_local_unanchored.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Cls=module.SourceLocalUnanchoredValue

def check(state,x,layer):
    state.update(x,layer)
    assert bool((torch.bincount(state.labels,minlength=state.base_count)>0).all())
    assert torch.equal(state.cluster_source[state.labels],state.source)
    assert len(state.reps.unique())==state.base_count
    assert torch.equal(state.labels[state.reps],torch.arange(state.base_count))
    centers=state._centers(x.float())
    d=(x-centers[state.labels]).square().sum(-1)
    nearest=torch.full((state.base_count,),torch.inf).scatter_reduce_(0,state.labels,d,reduce='amin')
    assert torch.allclose(d[state.reps],nearest,atol=1e-4,rtol=1e-4)


def main():
    torch.set_num_threads(2);torch.manual_seed(20261105)
    begin=time.perf_counter();results=[]
    for layout in ('sequential','spatiotemporal'):
        for width in (16,32,64):
            for span in (1,2,4):
                state=Cls(5,9,11,97,'cpu',width,2,layout,span,batch_groups=3)
                first_source=state.source.clone()
                for layer in range(3):
                    x=torch.randn(5*9*11,8)
                    if layer==1:x.zero_()  # Coincident-center ties.
                    check(state,x,layer)
                    assert torch.equal(state.source,first_source)
                assert state.candidates.shape[1]<=width
                results.append(dict(layout=layout,source_width=width,temporal_span=span,status='passed',layers=3))
    state=Cls(1,1,4,2,'cpu',2,1)
    old=state.reps.clone();check(state,torch.tensor([[9.],[0.],[10.],[11.]]),0)
    assert state.labels[old[0]]==1, 'Former representative must be able to escape'
    results.append(dict(test='former_representative_migration',status='passed'))
    state=Cls(1,1,6,3,'cpu',3,1)
    check(state,torch.tensor([[0.],[0.],[0.],[100.],[51.],[51.]]),0)
    results.append(dict(test='empty_repair_nonempty_donors',status='passed'))
    # Real 128-frame geometry layout only; no feature or GPU speed claims.
    for span in (1,2,4):
        labels,sources=module._initial_layout(128,28,37,26009,32,'spatiotemporal',span)
        assert len(labels)==132608 and len(labels.unique())==26009
        sizes=torch.bincount(sources)
        cluster_source=torch.empty(26009,dtype=torch.long);cluster_source[labels]=sources
        assert int(torch.bincount(cluster_source).max())<=32
        results.append(dict(test='128_frame_layout',temporal_span=span,source_groups=len(sizes),max_tokens_per_source=int(sizes.max()),status='passed'))
    elapsed=time.perf_counter()-begin
    quality={k:None for k in ('auc3','auc30','acc','comp','cd','f1')}
    data=dict(scope='CPU clustering invariants and static layout only',status='passed',tests=results,
              elapsed_seconds=elapsed,quality=quality,quality_unavailable_reason='Synthetic clustering diagnostics; no model, camera poses, or recon_unposed evaluated',
              dense_speedup=None,dense_speedup_unavailable_reason='No full-forward GPU timing or paired Dense run',peak_gpu_memory_bytes=None,
              sampling='Synthetic 5-frame tensors plus static 128-frame geometry; no dataset sequences',
              reconstruction_threshold=None,downsampling=None,saf3r_topk=None)
    out=ROOT/'outputs/source_local_unanchored_20261105';out.mkdir(exist_ok=True,parents=True)
    (out/'validation.json').write_text(json.dumps(data,indent=2)+'\n')
    (out/'validation.md').write_text(f'CPU diagnostic checks: {len(results)} passed ({elapsed:.3f}s). No full-forward or quality evaluation.\n\n| Run | AUC@3 (fraction) | AUC@30 (fraction) | ACC (m) | Comp (m) | CD (m) | F1 (fraction) | Dense speedup |\n|---|---|---|---|---|---|---|---|\n| Synthetic invariants | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n\nQuality unavailable: synthetic clustering only; no camera/reconstruction inference. Speedup unavailable: no full-forward GPU timing or paired Dense. CPU elapsed is diagnostic script time, not inference latency.\n')
    print(json.dumps(dict(status='passed',checks=len(results),elapsed_seconds=elapsed)))

if __name__=='__main__': main()
