"""Heads128: every global-layer/head AV output, SAF3R and population Oracle vs Dense.

Projection-input hooks observe real outputs without changing a forward. Local
counterfactual Dense AV uses each sparse layer's own inputs; rollout reference
uses a separate original Dense trajectory. Neither comparison samples Queries.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record,quality_cells,LABELS

MODES=('dense','saf3r','oracle_population')
SCOPES=('all_queries','patch_queries','special_queries')


def _summary(s):
    aa,bb,ab,ee,cs,cn,rs,rn,count=s
    if any(not math.isfinite(t) for t in s):raise ValueError('Nonfinite output statistics')
    return dict(queries=int(count),relative_l2_frobenius=math.sqrt(ee/bb) if bb>0 else None,
        mean_per_query_cosine=cs/cn if cn else None,cosine_valid_queries=int(cn),
        flattened_cosine=ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None,
        mean_per_query_relative_l2=rs/rn if rn else None,
        output_norm_ratio=math.sqrt(aa/bb) if bb>0 else None,
        actual_energy=aa,reference_energy=bb,error_energy=ee,dot=ab)


def head_output_metrics(actual,reference,per_frame,heads=16,chunk=2048):
    """Stream full [1,N,H*D] outputs; FP32 products, FP64 accumulation."""
    import torch
    if actual.shape!=reference.shape or actual.ndim!=3 or actual.shape[0]!=1:
        raise ValueError('Expected matched batch-one projection inputs')
    n,c=actual.shape[1:]
    if c%heads or per_frame<6 or chunk<1:raise ValueError('Invalid output geometry')
    a=actual[0].reshape(n,heads,c//heads);b=reference[0].reshape_as(a)
    accum=torch.zeros((3,heads,9),device=actual.device,dtype=torch.float64)
    for start in range(0,n,chunk):
        x=a[start:start+chunk].float();y=b[start:start+chunk].to(actual.device).float()
        xx=x.square().sum(-1);yy=y.square().sum(-1);xy=(x*y).sum(-1)
        ee=(x-y).square().sum(-1);valid=(xx>1e-24)&(yy>1e-24);rv=yy>1e-24
        cosine=(xy/(xx*yy).clamp_min(1e-24).sqrt()).clamp(-1,1)
        rel=ee.sqrt()/yy.clamp_min(1e-24).sqrt()
        ones=torch.ones_like(xx)
        values=torch.stack((xx,yy,xy,ee,torch.where(valid,cosine,0),valid.float(),
            torch.where(rv,rel,0),rv.float(),ones),-1)
        patch=torch.arange(start,start+len(x),device=actual.device)%per_frame>=5
        accum[0]+=values.sum(0,dtype=torch.float64)
        accum[1]+=values[patch].sum(0,dtype=torch.float64)
        accum[2]+=values[~patch].sum(0,dtype=torch.float64)
    raw=accum.cpu().tolist();total=accum.sum(1).cpu().tolist()
    by_head=[{scope:_summary(raw[s][h]) for s,scope in enumerate(SCOPES)} for h in range(heads)]
    aggregate={scope:_summary(total[s]) for s,scope in enumerate(SCOPES)}
    # Aggregate query counts here count head-Query pairs, not unique Tokens.
    return by_head,aggregate


def original_attention(attn):
    return getattr(attn,'original',getattr(attn,'orig_module',attn))


class HeadOutputCapture:
    def __init__(self,args,model,routing,patterns):
        import torch
        from saf3r.patch.region_history_attention import prepare_qkv
        self.args=args;self.records=[];self.routing=routing;self.handles=[];self.pending={}
        self.out=args.output_dir/args.mode/'heads'
        self.ref=args.output_dir/'dense'/'heads'/'dense_av'
        shape=json.loads(args.baseline.read_text())['input']['shape']
        self.per=(shape[-2]//14)*(shape[-1]//14)+5
        for layer,block in enumerate(model.aggregator.global_blocks):
            base=original_attention(block.attn)
            if args.mode!='dense' and layer>=17:
                def before(module,inputs,kwargs,l=layer,o=base):
                    x=inputs[0];pos=kwargs.get('pos',inputs[1] if len(inputs)>1 else None)
                    q,k,v=prepare_qkv(o,x,pos);q,k=q.to(v.dtype),k.to(v.dtype)
                    dense=torch.nn.functional.scaled_dot_product_attention(
                        q[None],k[None],v[None],scale=o.scale)
                    self.pending[l]=dense.transpose(1,2).reshape(1,x.shape[1],-1).contiguous()
                self.handles.append(block.attn.register_forward_pre_hook(before,with_kwargs=True))
            def projected(module,inputs,l=layer):
                actual=inputs[0].detach()
                if actual.shape[1]!=args.frames*self.per:raise ValueError('Token geometry changed')
                if args.mode=='dense':
                    self.ref.mkdir(parents=True,exist_ok=True)
                    torch.save(actual.cpu(),self.ref/f'L{l:02d}.pt')
                    reference=actual
                else:
                    reference=torch.load(self.ref/f'L{l:02d}.pt',map_location='cpu',weights_only=True)
                metrics,aggregate=head_output_metrics(actual,reference,self.per)
                if args.mode!='dense' and l<17:
                    if any(m['all_queries']['relative_l2_frobenius']>1e-6 for m in metrics):
                        raise ValueError(f'Dense prefix mismatch at layer{l}')
                local,local_aggregate=(None,None)
                if l in self.pending:
                    local,local_aggregate=head_output_metrics(actual,self.pending.pop(l),self.per)
                head_records=[]
                for h,m in enumerate(metrics):
                    row=dict(head=h,pattern=patterns.get((l,h),dict(mode='dense')),
                        rollout_vs_dense=m)
                    if local is not None:row['same_input_vs_dense']=local[h]
                    head_records.append(row)
                self.records.append(dict(layer=l,heads=head_records,rollout_aggregate=aggregate,
                    same_input_aggregate=local_aggregate))
                save(self.out/'head_metrics.json',dict(method=args.mode,layers=self.records,
                    complete=len(self.records)==24))
                save(self.out/'capture_progress.json',dict(layer=l,heads=16))
                print('CAPTURE_HEAD_OUTPUTS',args.mode,l,flush=True)
            self.handles.append(base.proj.register_forward_pre_hook(projected))


def run(args):
    from scripts.evaluate_headwise_deep_oracles import run as evaluate
    from saf3r.patch.deep_saf3r_ablation import install_deep_saf3r,support_statistics
    from saf3r.patch.restore_native_heads import install_restored_heads
    profile=json.loads(args.profile.read_text());holder={}
    def installer(model,mode,query_chunk=64,progress=None):
        shape=json.loads(args.baseline.read_text())['input']['shape'];ph,pw=shape[-2]//14,shape[-1]//14
        routing=None;patterns={}
        if mode=='saf3r':
            selected=install_deep_saf3r(model,profile,ph,pw)
            for l in support_statistics(selected,args.frames,ph,pw):
                for g in l['patterns']:
                    for h,count,density in zip(g['heads'],g['key_counts'],g['head_densities']):
                        patterns[(l['layer'],h)]=dict(mode=g['mode'],keys=count,density=density,config=g['config'])
        elif mode=='oracle_population':
            routing=install_restored_heads(model,profile,'probe_oracle_population',query_chunk,progress)
            for l in range(17,24):
                for h in range(16):
                    patterns[(l,h)]=dict(mode='attention_oracle_per_query' if h in routing.routing[l]
                        else 'value_oracle_population',density=math.floor(.2*args.frames*(ph*pw+5))/(args.frames*(ph*pw+5)))
        recorder=HeadOutputCapture(args,model,routing,patterns);holder['capture']=recorder
        return recorder
    out=args.output_dir/args.mode/'heads'
    if args.mode!='dense':
        ref=args.output_dir/'dense'/'heads'/'input.json'
        if not ref.exists():raise ValueError('Run matching Dense capture first')
        if len(json.loads(ref.read_text())['image_paths'])!=args.frames:raise ValueError('Reference frame count mismatch')
    evaluate(args,installer=installer,extra_sources=('scripts/compare_deep_head_rollouts.py',
        'saf3r/patch/restore_native_heads.py','saf3r/patch/deep_saf3r_ablation.py',
        'saf3r/patch/patch_vggt.py','saf3r/patch/wrapper.py','saf3r/patch/sparse_attention.py',
        str(args.profile.resolve().relative_to(ROOT))),extra_config=dict(
            diagnostic_method=args.mode,captured_global_layers=list(range(24)),
            sparse_global_layers=[] if args.mode=='dense' else list(range(17,24)),
            dense_global_layers=list(range(24)) if args.mode=='dense' else list(range(17)),
            retention=.2 if args.mode=='oracle_population' else None,
            population_logit_compensation=args.mode=='oracle_population',
            saf3r_profile=str(args.profile),dino_topk_active=False,top128_active=False,
            native_saf3r_budgets=args.mode=='saf3r',
            output_scope='Per-head actual AV captured at projection input, before projection/residual',
            local_reference='Dense QK/softmax/AV on sparse-layer current inputs, global17–23 only',
            rollout_reference='Original Dense full forward; all24 global layers',
            query_sampling='All Query Tokens, no sampling; also patch/special scopes',
            timing_scope='Instrumented forward includes output I/O, comparisons and extra Dense counterfactuals; NOT a speed benchmark'))
    capture=holder['capture']
    if len(capture.records)!=24:raise ValueError('Missing global layer captures')
    if capture.routing is not None:save(out/'oracle_routing_runtime.json',capture.routing.records)
    for h in capture.handles:h.remove()
    if args.frames==128:
        p=out/'result.json';data=json.loads(p.read_text())
        data['diagnostic_method']=args.mode
        data['diagnostic_seconds_include']='CPU/disk capture, per-head comparisons and extra same-input Dense computations'
        if args.mode=='dense':
            data['quality']['speedup_vs_vggt']=1.
            data['quality']['speedup_unavailable']=None
        else:
            data['quality']['speedup_unavailable']='Instrumented accuracy/output diagnostic; no warmed paired full-forward timing'
        save(p,data);write_quality(out,data['quality'])
    report(args.output_dir)


def report(root):
    rows=[];quality=[];data={}
    for mode in MODES:
        directory=root/mode/'heads';p=directory/'head_metrics.json'
        if p.exists():
            x=json.loads(p.read_text())
            if not x['complete']:continue
            data[mode]=x
            for l in x['layers']:
                for h in l['heads']:
                    for comp in ('rollout_vs_dense','same_input_vs_dense'):
                        for scope,m in h.get(comp,{}).items():
                            rows.append(dict(method=mode,comparison=comp,scope=scope,layer=l['layer'],head=h['head'],
                                pattern=h['pattern']['mode'],**m))
        p=directory/'result.json'
        if p.exists():
            x=json.loads(p.read_text());quality.append(dict(method=mode,quality=x['quality'],input=x['input'],
                quality_protocol=x['quality_protocol'],diagnostic_seconds=x['diagnostic_seconds'],peak_gib=x['peak_gib']))
    if quality and any(q['input']!=quality[0]['input'] or q['quality_protocol']!=quality[0]['quality_protocol'] for q in quality):
        raise ValueError('Unmatched quality comparison')
    if rows:
        with (root/'head_comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    complete=all(m in data for m in MODES) and len(quality)==3
    save(root/'head_comparison.json',dict(complete=complete,methods=data,quality=quality))
    lines=['# Heads128: native deep SAF3R and population Value/Attention Oracle vs Dense','',
        f'Complete three-method quality/output comparison: {complete}.',
        'Only global17–23 sparse; frame/local and global0–16 unchanged. Zero-based layers/heads.',
        'All24 global layers x16 heads captured at actual projection inputs; all Queries included.',
        'Rollout reference: separate Dense trajectory. Same-input reference: Dense on each sparse layer\'s own Q/K/V.',
        'Relative L2=||sparse-Dense||_F/||Dense||_F; cosine averaged over per-Query head vectors.',
        'SAF3R uses native deep head profile, including original first-frame/special rules; not20%-matched.',
        'No DINO Top-K heads exist here; neither lazy Top128 nor paper-DINO Top4 is executed.',
        'No speedup from instrumented runs: extra Dense counterfactuals and output I/O are included in diagnostic time.',
        '', '| Method | '+' | '.join(LABELS)+' | Speedup |','|---|'+'---:|'*7]
    for q in quality:
        lines.append('| '+q['method']+' | '+' | '.join(quality_cells(q['quality']))+' | '+
            ('1.00x reference' if q['method']=='dense' else 'N/A: instrumented diagnostic')+' |')
    sparse=[m for m in ('saf3r','oracle_population') if m in data]
    for comp,agg,layers in [('Full-rollout versus Dense','rollout_aggregate',range(24)),
                            ('Same-input layer-local versus Dense','same_input_aggregate',range(17,24))]:
        lines+=['',f'## {comp}','', '| Layer | '+' | '.join(f'{m} rel L2 % | {m} mean cosine' for m in sparse)+' |',
            '|---|'+'---:|'*(2*len(sparse))]
        for layer in layers:
            cells=[]
            for m in sparse:
                v=next(l for l in data[m]['layers'] if l['layer']==layer)[agg]['all_queries']
                cells.extend([f"{100*v['relative_l2_frobenius']:.4f}",f"{v['mean_per_query_cosine']:.6f}"])
            lines.append('| '+str(layer)+' | '+' | '.join(cells)+' |')
    lines+=['','Per-head all/patch/special metrics: head_comparison.csv and head_comparison.json.',
        'Layer-level relative L2 pools head energies; it is not the arithmetic average of head relative errors.',
        'Reconstruction: unposed,threshold0.05m,downsample0.0078125m,CD=(ACC+Comp)/2 unsquared.',
        'One heads128 sequence; no dataset macro, no tuning or statistical significance claim.']
    (root/'head_comparison.md').write_text('\n'.join(lines)+'\n')
    if all(m in data for m in ('saf3r','oracle_population')):
        detail=['# Complete per-head deep output comparison','',
            'heads128,zero-based global layers/heads. All Queries; AV before projection/residual.',
            'Global0–16: all16 heads of both methods match Dense (relative L2=0,cosine=1).',
            'CSV additionally contains every0–23 layer and patch/special Query scopes.',
            'Same-input compares each method with Dense on its OWN current inputs; sparse methods do not share those inputs afterL17.']
        for comp in ('rollout_vs_dense','same_input_vs_dense'):
            detail+=['',f'## {comp}']
            for layer in range(17,24):
                detail+=['',f'### Global{layer}','',
                    '| Head | SAF3R pattern | SAF3R error % | SAF3R cosine | Oracle pattern | Oracle error % | Oracle cosine |',
                    '|---|---|---:|---:|---|---:|---:|']
                sa=next(l for l in data['saf3r']['layers'] if l['layer']==layer)
                oa=next(l for l in data['oracle_population']['layers'] if l['layer']==layer)
                for sh,oh in zip(sa['heads'],oa['heads']):
                    s=sh[comp]['all_queries'];o=oh[comp]['all_queries']
                    detail.append(f"|{sh['head']}|{sh['pattern']['mode']}|{100*s['relative_l2_frobenius']:.4f}|{s['mean_per_query_cosine']:.6f}|{oh['pattern']['mode']}|{100*o['relative_l2_frobenius']:.4f}|{o['mean_per_query_cosine']:.6f}|")
        (root/'per_head_deep.md').write_text('\n'.join(detail)+'\n')
        plot(root,data)


def plot(root,data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,title,scale in [('rollout_vs_dense','Full-rollout vs original Dense',1),
                            ('same_input_vs_dense','Same-input layer-local vs Dense',1)]:
        for metric,label,mult in [('relative_l2_frobenius','Relative L2 error (%)',100),
                                   ('mean_per_query_cosine','Mean per-Query cosine',1)]:
            arrays=[]
            for method in ('saf3r','oracle_population'):
                deep=[l for l in data[method]['layers'] if l['layer']>=17]
                arrays.append(np.array([[h[key]['all_queries'][metric]*mult for h in l['heads']] for l in deep]))
            vmax=max(float(a.max()) for a in arrays) if mult==100 else 1.
            vmin=0 if mult==100 else min(0.,min(float(a.min()) for a in arrays))
            fig,axes=plt.subplots(2,1,figsize=(17,9),layout='constrained')
            for ax,array,name in zip(axes,arrays,('SAF3R native deep profile','Attention Oracle + population Value')):
                im=ax.imshow(array,aspect='auto',vmin=vmin,vmax=vmax,cmap='magma' if mult==100 else 'viridis')
                ax.set_xticks(range(16),labels=range(16));ax.set_yticks(range(7),labels=range(17,24))
                ax.set_xlabel('Head (zero based)');ax.set_ylabel('Global layer');ax.set_title(name)
                for i in range(7):
                    for j in range(16):ax.text(j,i,f'{array[i,j]:.1f}' if mult==100 else f'{array[i,j]:.2f}',
                        ha='center',va='center',fontsize=8,color='white' if array[i,j]<(vmin+vmax)/2 else 'black')
            fig.suptitle(f'heads128 | {title} | {label}')
            fig.colorbar(im,ax=list(axes),shrink=.85,label=label)
            fig.savefig(root/f'{key}_{metric}.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--frames',choices=[2,128],type=int,default=128)
    p.add_argument('--scene',choices=['heads'],default='heads')
    p.add_argument('--query-chunk',type=int,default=64)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/global_attention/heads128_deep_head_comparison_20261020')
    p.add_argument('--baseline',type=Path,default=ROOT/'outputs/attention_first_residual_value/campaign_7scenes128/quality/dense/heads/result.json')
    p.add_argument('--profile',type=Path,default=ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    if args.report_only:report(args.output_dir)
    else:
        try:run(args)
        except Exception as exc:
            out=args.output_dir/args.mode/'heads'
            if not (out/'result.json').exists():
                write_quality(out,quality_record({},f'Head-output comparison failed: {type(exc).__name__}: {exc}'))
                save(out/'status.json',dict(stage='failed',error=str(exc)))
            raise
