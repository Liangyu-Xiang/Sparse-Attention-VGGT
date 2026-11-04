"""Diagnostic only: split captured selection without changing method formulas."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation'),str(ROOT/'scripts')]
import torch
import triton
import triton.language as tl
import yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
import saf3r.patch.reuse_residual_exploration as impl
import saf3r.patch.reuse_window_graph as window
from grid_local_scopes import Recorder


@triton.jit
def marker(TIMES,COUNT,LAYER,PHASE:tl.constexpr,CAPACITY:tl.constexpr):
    slot=tl.load(COUNT)
    now=tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)
    tl.store(TIMES+slot*8+PHASE,now,slot<CAPACITY)
    if PHASE==0:
        layer=tl.load(LAYER)
        tl.store(TIMES+slot*8+7,layer.to(tl.uint64),slot<CAPACITY)
    if PHASE==6:
        tl.store(COUNT,slot+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    capacity=20000
    times=torch.zeros((capacity,8),device='cuda',dtype=torch.uint64)
    count=torch.zeros(1,device='cuda',dtype=torch.int32)
    layer_id=torch.zeros_like(count)
    def mark(phase):marker[(1,)](times,count,layer_id,phase,capacity,num_warps=1)
    # Compile each marker before capture, then validate its clock against CUDA Events.
    for i in range(7):mark(i)
    torch.cuda.synchronize();count.zero_()
    first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
    first.record();mark(0);torch.cuda._sleep(10000000);mark(6);last.record()
    torch.cuda.synchronize()
    trial=times[0].cpu().tolist();clock_seconds=(trial[6]-trial[0])/1e9
    event_seconds=first.elapsed_time(last)/1000
    assert abs(clock_seconds-event_seconds)<.0003,(clock_seconds,event_seconds)
    save=lambda name,value:(out/name).write_text(json.dumps(value,indent=2)+'\n')
    save('clock_validation.json',dict(globaltimer_seconds=clock_seconds,cuda_event_seconds=event_seconds,
         absolute_difference_seconds=abs(clock_seconds-event_seconds),passed=True))
    text=inspect.getsource(window.eager)
    text=text.replace('    context=rule.gate','    _profile_mark(0)\n    context=rule.gate',1)
    text=text.replace('    if geometry.get','    _profile_mark(1)\n    if geometry.get',1)
    text=text.replace('    selected,count=budget','    _profile_mark(2)\n    selected,count=budget',1)
    text=text.replace('    ids,real,record=rule.pack','    _profile_mark(3)\n    ids,real,record=rule.pack',1)
    text=text.replace('    weights=rule.transfer','    _profile_mark(4)\n    weights=rule.transfer',1)
    text=text.replace('    record.update(rule.diagnostics','    _profile_mark(5)\n    record.update(rule.diagnostics',1)
    text=text.replace('    return ids,real,weights,record','    _profile_mark(6)\n    return ids,real,weights,record',1)
    (out/'instrumented_eager.py').write_text(text)
    window.__dict__['_profile_mark']=mark
    exec(compile(text,'<diagnostic graph markers>','exec'),window.__dict__)
    original_forward=impl.ReuseResidualExploration.forward
    def layer_forward(att,*a,**kw):
        layer_id.fill_(att.layer)
        return original_forward(att,*a,**kw)
    impl.ReuseResidualExploration.forward=layer_forward
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module');cfg.pop('sparse_config_path')
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda')
    canonical=json.loads(args.input.read_text())
    images=load_images(model,'vggt',canonical['image_paths']).cuda()
    config=json.loads((ROOT/'configs/evaluation/reuse_efficiency_v8.json').read_text())
    state=impl.install_reuse_residual_exploration(model,impl.ReuseExploreConfig(**config))
    count.zero_()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();del raw
    rec=Recorder(state);rec.install()
    # Actual consumer is an imported alias, not the reference module's attribute.
    import saf3r.patch.reuse_grid_fused_history as history
    rec.wrap(history,'build_prior','history_prior_estimation')
    import saf3r.triton.reuse_serial_tuned as attention
    rec.wrap(attention,'weighted_attention','attention_wrapper')
    class Launch:
        def __init__(self,kernel,label):self.kernel=kernel;self.label=label
        def __getitem__(self,grid):
            call=self.kernel[grid]
            def run(*a,**kw):
                with rec.scope(self.label):return call(*a,**kw)
            return run
    for name,label in [('_attention','attention_core_fused_grid'),('_reduce','attention_grid_normalize_reduce')]:
        setattr(attention,name,Launch(getattr(attention,name),label))
    for block in model.aggregator.global_blocks:
        rec.wrap(block.attn.original.proj,'forward','output_projection')
        rec.wrap(block.attn.original.proj_drop,'forward','output_dropout')
    # Same operations/order as State.prepare; expose three distinct responsibilities.
    def prepare(k,features):
        f=state.fine.base_count;r=state.regions
        with rec.scope('cluster_population'):
            population=torch.bincount(state.fine.labels,minlength=f).float()
        means=None
        with rec.scope('history_key_pooling'):
            pooled=torch.zeros((len(k),r,k.shape[-1]),device=k.device)
            pooled.index_add_(1,state.grid,k[:,state.patches].float())
            pooled=pooled/state.grid_sizes[None,:,None]
            if state.config.history_mode=='dense_m5c':state.history_pooled_fp32=pooled.contiguous()
            pooled=pooled.to(k.dtype).contiguous()
        overlap=None
        if state.previous_layer>=0:
            with rec.scope('selection_cluster_grid_overlap'):
                pair=state.fine.labels*r+state.grid
                unique,amount=torch.unique(pair,sorted=True,return_counts=True)
                overlap=torch.sparse_coo_tensor(torch.stack((unique//r,unique%r)),amount.float()/state.grid_sizes[unique%r],size=(f,r),device=k.device).coalesce()
        return population,means,pooled,overlap
    state.prepare=prepare
    torch.cuda.synchronize();count.zero_();times.zero_();torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();wall=time.perf_counter()-begin
    assert len(state.records)==24 and torch.isfinite(raw['depth']).all() and torch.isfinite(raw['pose_enc']).all()
    events,groups,total,unique=rec.finish()
    used=int(count.item());assert used<capacity
    measurements=times[:used].cpu().tolist()
    graph=[]
    for index,row in enumerate(measurements):
        assert all(row[i+1]>=row[i] for i in range(6)),(index,row)
        graph.append(dict(index=index,layer=int(row[7]),gate_seconds=(row[1]-row[0])/1e9,
            loss_seconds=(row[2]-row[1])/1e9,selection_seconds=(row[3]-row[2])/1e9,
            pack_seconds=(row[4]-row[3])/1e9,transfer_seconds=(row[5]-row[4])/1e9,
            diagnostics_seconds=(row[6]-row[5])/1e9,total_seconds=(row[6]-row[0])/1e9))
    assert set(x['layer'] for x in graph)==set(range(1,24))
    result=dict(complete=True,wall_seconds=wall,global_attention_seconds=total,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,groups=groups,events=events,
        graph_windows=graph,graph_marker_count=used,sum_error_seconds=unique-total,
        config=config,input=canonical,hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        timing_scope='One warmed instrumented forward; graph subphases use validated GPU timestamps, outer scopes use CUDA Events; not uninstrumented benchmark median',
        quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},
        quality_reason='Timing diagnostic only; no pose/recon_unposed evaluation performed',dense_speedup=None,
        dense_reason='Unchanged Dense not rerun; no paired timing',instrumentation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save('profile.json',result)
    print(json.dumps(dict(wall_seconds=wall,global_attention_seconds=total,graph_windows=used,
        graph_seconds=sum(x['total_seconds'] for x in graph),groups=groups),indent=2),flush=True)


if __name__=='__main__':main()
