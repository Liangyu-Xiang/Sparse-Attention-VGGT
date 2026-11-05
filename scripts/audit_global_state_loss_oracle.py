"""Whole-mask, globally budgeted key/state oracles on frozen real QKV.

No per-cluster key quota. Exact Dense probabilities are diagnostic targets only.
Straight-through discrete search is an approximate oracle, never a proof of a
globally optimal mask. Every evaluated hard mask obeys the same global budget.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--capture-only',action='store_true')
    p.add_argument('--replay-dir',type=Path)
    a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    from scripts.quality_metrics import quality_record,quality_markdown
    quality=quality_record({},'Frozen-QKV whole-mask oracle diagnostic; pose/recon_unposed not evaluated; no geometry truth is read')
    quality['speedup_unavailable']='Instrumented oracle optimization with diagnostic Dense QK; not warmed production forward timing'
    save(out/'quality.json',quality);(out/'quality.md').write_text(quality_markdown(quality))
    save(out/'status.json',dict(stage='loading'))
    import torch
    import torch.nn.functional as F
    import numpy as np
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch import reuse_residual_exploration as core
    from saf3r.patch import reuse_sampled_history as history
    from saf3r.patch import reuse_window_graph as window
    from saf3r.patch import reuse_inference_ops as rules
    from saf3r.triton import reuse_shared_value_states as shared
    from saf3r.triton.reuse_compact import budget
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads(a.config.read_text())
    assert config['shared_value_weighting']=='cluster_population'
    assert config['shared_value_features']=='head_mean'
    paths=json.loads(a.input.read_text())['image_paths']
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module');cfg.pop('sparse_config_path')
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    if a.replay_dir:
        metadata=json.loads((a.replay_dir/'input.json').read_text())
        assert metadata['config']==config and metadata['image_paths']==paths
        model=images=state=None;modules={}
        image_shape=metadata['image_shape'];checkpoint_sha=metadata['checkpoint_sha256']
    else:
        model=build_model(cfg,'cuda');images=load_images(model,'vggt',paths).cuda()
        state=core.install_reuse_residual_exploration(model,core.ReuseExploreConfig(**config))
        modules={id(block.attn.original):i for i,block in enumerate(model.aggregator.global_blocks)}
        image_shape=list(images.shape);checkpoint_sha=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest()
    layers={6,11,13,19};holder={};cases=[];key_records=out/'clusters.jsonl.gz'
    raw=gzip.open(key_records,'wt')
    per_frame=image_shape[-2]//14*(image_shape[-1]//14)+5
    save(out/'input.json',dict(image_paths=paths,image_shape=image_shape,config=config,teacher_forced=False,
         checkpoint_sha256=checkpoint_sha,replay_dir=str(a.replay_dir),
         source_sha256={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in
             ['saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_shared_value_states.py',
              'saf3r/patch/reuse_window_graph.py','scripts/audit_global_state_loss_oracle.py']},
         layers=sorted(layers),groups='1/4 and 3/4 of Query groups; all clusters/keys and all sparse-executed Queries in each selected group',
         query_exclusions='Camera queries already overridden by Dense correction do not depend on sparse mask',
         oracle='Approximate global hard-mask search, exact FP32 post-Wo objective; state oracle 2 initializations, free-key oracle 3',
         budget='Native actual number of selected patch keys; protected keys unchanged; no per-cluster quota',
         dense_reference='Local Dense QK on frozen sparse-trajectory QKV; not an independent Dense model quality run',
         gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),gpu=torch.cuda.get_device_name(),
         precision='BF16 model, FP32 oracle, TF32 off',saf3r_profile='N/A: user Value/reuse method',
         reconstruction_threshold=None,scope='RGB-only diagnostic; no dataset quality evaluation'))

    # Loss is always evaluated AFTER the global sparse softmax denominator and Wo.
    def evaluator(prob,value,special_num,special_den,projection,bias=None):
        def project(y):
            y=y.transpose(0,1).reshape(y.shape[1],-1)
            return F.linear(y,projection,bias) if projection is not None else y
        with torch.no_grad():
            target=project((prob@value+special_num)/(prob.sum(-1)+special_den)[...,None])
            scale=target.square().sum().clamp_min(1e-20)
        def loss(weights):
            weighted=prob*weights[None,None,:]
            y=(weighted@value+special_num)/(weighted.sum(-1)+special_den).clamp_min(1e-30)[...,None]
            return (project(y)-target).square().sum()/scale
        return loss

    def dual_choice(errors,costs,cap,upper=None):
        if upper is None:
            finite=torch.where(torch.isfinite(errors),errors,0.)
            upper=(finite.max()-finite.min()).reshape(1)*2+1e-7
        selected,count=budget(errors[None].contiguous(),costs[None].contiguous(),upper,cap,16)
        return selected[0],count[0]

    def audit(s,layer,start,local,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform,result):
        # Model inference tensors must be cloned into normal tensors for oracle autograd.
        with torch.inference_mode(False),torch.enable_grad(),torch.autocast('cuda',enabled=False):
            group=start+local;module=holder['module'];q,k,v=holder['qkv']
            query=torch.arange(group*256,min((group+1)*256,s.n),device=q.device)
            if config.get('dense_special_queries'):
                query=query[query%per_frame>=config['dense_special_queries']]
            patch=s.patches.clone();special=s.specials.clone();lab=labels.clone();rep=reps.clone()
            pop=population.float().clone();f=len(rep);n=len(lab)
            vv=v[:,patch].float().clone();vs=v[:,special].float().clone()
            prob=(q[:,query].float().clone()@k.float().clone().transpose(-1,-2)*module.scale).softmax(-1)
            pp=prob[:,:,patch].clone();sp=prob[:,:,special].clone();del prob
            wo=module.proj.weight.detach().float().clone();bias=module.proj.bias.detach().float().clone() if module.proj.bias is not None else None
            truth=evaluator(pp,vv,sp@vs,sp.sum(-1),wo,bias)
            est=prior[local].clone();mass=pi[local].clone();old=old_ids[local:local+1].clone()
            ctx=rules.gate(est[None],mass[None],pop,old,lab,rep)
            if config['shared_value_allow_observed_rep']:ctx['remove_rep']=torch.zeros_like(ctx['remove_rep'])
            eligible=ctx['qualified'][0].clone();hc=ctx['eligible_count'][0].float().clone()
            permitted=allowed[local].clone()
            copy_geom=dict(geometry)
            corr=shared.corrections(est[None],old,uniform[local:local+1].clone(),lab,copy_geom,ctx)
            errors,costs=shared.states(uniform[local:local+1],permitted[None],pop,copy_geom,ctx,corr)
            errors=errors[0].clone();costs=costs[0].clone()
            ids=result[0][local].long();weights=result[2][local].float();valid=ids>=0
            native=torch.zeros(n,device=q.device);native[ids[valid]]=1.
            cap=int(valid.sum())
            counts=torch.zeros(f,device=q.device).scatter_add_(0,lab,native)
            rep_mask=torch.zeros(n,device=q.device);rep_mask[rep]=1.
            hist_in_native=torch.zeros(f,device=q.device).scatter_add_(0,lab,native*eligible)
            native_state=torch.zeros(f,device=q.device,dtype=torch.long)
            is_rep=(counts==1)&native[rep].bool()
            is_hist=(counts>0)&(counts==hc)&(hist_in_native==hc)
            is_exact=(counts==pop)&permitted
            native_state[is_exact]=3;native_state[is_hist]=2;native_state[is_rep]=1
            valid_states=(counts==0)|is_rep|is_hist|is_exact
            if not bool(valid_states.all()):
                bad=torch.nonzero(~valid_states).flatten()[:12]
                save(out/'support_mismatch.json',dict(layer=layer,group=group,
                     examples=[dict(cluster=int(c),count=float(counts[c]),population=float(pop[c]),
                        history_count=float(hc[c]),history_overlap=float(hist_in_native[c]),
                        rep=int(rep[c]),rep_kept=bool(native[rep[c]]),allowed=bool(permitted[c]),
                        keys=ids[valid & (lab[ids.clamp_min(0)]==c)].cpu().tolist()) for c in bad]))
                raise AssertionError('Native support not expressible by stated 4-state vocabulary; see support_mismatch.json')
            def state_weights(choice):
                return (choice[lab,1]*rep_mask*pop[lab]+choice[lab,2]*eligible*pop[lab]/hc[lab].clamp_min(1)+choice[lab,3])
            native_w=state_weights(F.one_hot(native_state,4).float())
            torch.testing.assert_close(native_w[ids[valid]],weights[valid],rtol=2e-6,atol=2e-6)
            native_loss=float(truth(native_w))
            state_feasible=torch.isfinite(errors)
            # Diagnostic probability replacement freezes eligibility, features, costs and candidates.
            zz=z.float().clone();zd=zz.square().sum(-1);zr=(zz-zz[rep[lab]]).square().sum(-1)
            def legacy(probability):
                drop=torch.zeros(f,device=q.device).scatter_add_(0,lab,probability*zd)
                re=torch.zeros_like(drop).scatter_add_(0,lab,probability*zr)
                hist=torch.zeros_like(drop).scatter_add_(0,lab,probability*zd*(~eligible))
                e=torch.stack((drop,re,hist,torch.zeros_like(drop)),1)
                return e.masked_fill(~state_feasible,float('inf'))
            formula_errors=legacy(est)
            finite=state_feasible
            algebra_error=float((formula_errors[finite]-errors[finite]).abs().max())
            torch.testing.assert_close(formula_errors[finite],errors[finite],rtol=1e-3,atol=2e-5)
            probe0=int(holder['obs']['positions'][local,0])
            probe_index=int((query-probe0).abs().argmin())
            prob_group=pp.mean((0,1));prob_probe0=pp[:,probe_index].mean(0)
            prior_total_variation=float((est-prob_group).abs().sum()/2)
            true_mean_sel,_=dual_choice(legacy(prob_group),costs,cap,legacy(prob_group)[:,0].max().reshape(1)+1e-12)
            true_probe_sel,_=dual_choice(legacy(prob_probe0),costs,cap,legacy(prob_probe0)[:,0].max().reshape(1)+1e-12)
            obs=holder['obs']
            special_est=(obs['special_logits'][local,:,:1,:]-obs['coarse_logz'][local,:,:1,:]).exp().mean((0,1)).clone()
            pe=est[None,None,:];se=special_est[None,None,:]
            rms=v.mean(0,dtype=torch.float32)[patch].square().sum(-1).mean().sqrt().clamp_min(1e-12)
            zs=(v.mean(0,dtype=torch.float32)[special]/rms).clone()[None]
            z_est=evaluator(pe,zz[None],se@zs,se.sum(-1),None)
            wo_est=evaluator(pe.expand(len(v),-1,-1),vv,(se.expand(len(v),-1,-1)@vs),se.sum(-1).expand(len(v),-1),wo,bias)
            records={}
            for name,sel in [('legacy_true_group_probability',true_mean_sel),('legacy_true_probe_probability',true_probe_sel)]:
                w=state_weights(F.one_hot(sel,4).float())
                records[name]=dict(relative_l2=float(truth(w).sqrt()),keys=int(costs.gather(1,sel[:,None]).sum()),quality=quality)

            def optimize(name,objective,kind,seeds,steps):
                best_value=float('inf');best_w=None;best_state=None;curves=[]
                for seed_name,seed in seeds:
                    if kind=='states':
                        initial=F.one_hot(seed,4).float()*.1
                    else:
                        # Dummy slots allow unused budget; no cluster has a quota.
                        dummy=torch.zeros(cap,device=q.device)
                        dummy[:cap-int(seed.sum())]=1.
                        initial=(torch.cat((seed,dummy))*2-1)*.05
                        initial+=torch.linspace(0,1e-6,n+cap,device=q.device)
                    parameter=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([parameter],lr=.025,eps=1e-12)
                    curve=[]
                    for iteration in range(steps+1):
                        optimizer.zero_grad(set_to_none=True)
                        if kind=='states':
                            scores=parameter.masked_fill(~state_feasible,-1e6)
                            with torch.no_grad():
                                e=(-scores).masked_fill(~state_feasible,float('inf'))
                                chosen,count=dual_choice(e,costs,cap)
                            hard=F.one_hot(chosen,4).float();soft=scores.softmax(-1)
                            w=state_weights(hard+(soft-soft.detach()))
                            actual_count=int(costs.gather(1,chosen[:,None]).sum())
                        else:
                            hard=torch.zeros_like(parameter).scatter_(0,parameter.topk(cap).indices,1.)
                            soft=parameter[:n].sigmoid();gate=hard[:n]+(soft-soft.detach())
                            amount=torch.zeros(f,device=q.device).scatter_add_(0,lab,gate)
                            w=gate*pop[lab]/amount[lab].clamp_min(1.)
                            actual_count=int(hard[:n].sum());chosen=None
                        assert actual_count<=cap
                        loss=objective(w);value=float(loss.detach())
                        assert np.isfinite(value)
                        if value<best_value:
                            best_value=value;best_w=w.detach().clone()
                            best_state=chosen.detach().clone() if chosen is not None else None
                        curve.append(value)
                        if iteration<steps:loss.backward();optimizer.step()
                    curves.append(dict(seed=seed_name,objective=curve,best=min(curve)))
                actual=float(truth(best_w).sqrt())
                records[name]=dict(relative_l2=actual,fit_relative_l2=best_value**.5,keys=int((best_w>0).sum()),
                                   search='Approximate straight-through Adam; every hard iterate globally budget-feasible',curves=curves,quality=quality)
                return best_w,best_state

            est_z_w,est_z_s=optimize('joint_same_features_estimated_probability',z_est,'states',[('native',native_state)],a.steps)
            est_wo_w,est_wo_s=optimize('joint_full_wo_estimated_probability',wo_est,'states',[('native',native_state)],a.steps)
            oracle_state_w,oracle_state=optimize('state_space_oracle',truth,'states',
                 [('native',native_state),('oracle_shared_probability',true_mean_sel)],a.steps)
            top_mass=torch.zeros_like(native).scatter_(0,prob_group.topk(cap).indices,1.)
            oracle_w,_=optimize('free_key_oracle',truth,'keys',
                 [('native',native),('top_true_probability',top_mass),('state_oracle',(oracle_state_w>0).float())],a.steps)
            # Oracle never assigns a quota to any cluster; this distribution is an output.
            oracle=(oracle_w>0);oracle_count=torch.zeros(f,device=q.device).scatter_add_(0,lab,oracle.float())
            oracle_h=torch.zeros(f,device=q.device).scatter_add_(0,lab,(oracle&eligible).float())
            oracle_rep=oracle[rep]
            reachable=rep_mask.bool()|eligible|permitted[lab]
            unreachable=int((oracle&~reachable).sum())
            cover=torch.full((f,),float('inf'),device=q.device)
            cover[permitted]=pop[permitted]
            cover[(oracle_count==oracle_h)&(hc>0)]=torch.minimum(cover[(oracle_count==oracle_h)&(hc>0)],hc[(oracle_count==oracle_h)&(hc>0)])
            cover[(oracle_count==1)&oracle_rep]=1;cover[oracle_count==0]=0
            same_rep=(oracle_count==1)&oracle_rep
            same_hist=(oracle_count==hc)&(oracle_count==oracle_h)&(hc>0)
            same_exact=(oracle_count==pop)&permitted
            expressible=(oracle_count==0)|same_rep|same_hist|same_exact
            budget_replay,_=dual_choice(copy_geom['objective'][0],costs,cap,errors[:,0].max().reshape(1)+1e-12)
            replay_w=state_weights(F.one_hot(budget_replay,4).float())
            case=dict(layer=layer,group=group,queries=len(query),clusters=f,patch_keys=n,budget=cap,quality=quality,
                native_relative_l2=native_loss**.5,variants=records,loss_kernel_algebra_max_abs=algebra_error,
                estimated_vs_true_group_patch_probability_total_variation=prior_total_variation,
                native_replay_support_difference=int(((replay_w>0)!=(native_w>0)).sum()),
                oracle_keys_unreachable_by_any_existing_state=unreachable,
                oracle_keys_in_unexpressible_cluster_supports=int(oracle_count[~expressible].sum()),
                minimum_existing_state_cost_to_cover_oracle_keys=float(cover.sum()) if torch.isfinite(cover).all() else None,
                oracle_scope='All patch keys/clusters jointly, all non-overridden Queries in this group, full heads and Wo',
                optimality='No global optimality guarantee; reported approximate search best over all iterates and listed initializations')
            # Save every cluster's desired key set, native set, feasible state set and loss values.
            members=[[] for _ in range(f)]
            for key,c in enumerate(lab.cpu().tolist()):members[c].append(key)
            native_cpu=native.bool().cpu().tolist();oracle_cpu=oracle.cpu().tolist();elig_cpu=eligible.cpu().tolist()
            es=errors.cpu().tolist();cs=costs.cpu().tolist()
            native_s=native_state.cpu().tolist();os=oracle_state.cpu().tolist();ps=true_mean_sel.cpu().tolist()
            ez=est_z_s.cpu().tolist();ew=est_wo_s.cpu().tolist();rep_cpu=rep.cpu().tolist()
            for c,keys in enumerate(members):
                row=dict(layer=layer,group=group,cluster=c,population=len(keys),rep=rep_cpu[c],
                    native_keys=[k for k in keys if native_cpu[k]],oracle_keys=[k for k in keys if oracle_cpu[k]],
                    eligible_history_keys=[k for k in keys if elig_cpu[k]],
                    state_ids='0 Drop,1 Rep,2 History,3 Exact; diagnostic index, not production decoded enum',
                    native_state=native_s[c],state_oracle_state=os[c],true_probability_state=ps[c],
                    joint_same_features_state=ez[c],joint_full_wo_state=ew[c],
                    legacy_losses=[v if np.isfinite(v) else None for v in es[c]],costs=cs[c],
                    exact_allowed=bool(permitted[c]),oracle_support_expressible=bool(expressible[c]))
                raw.write(json.dumps(row,allow_nan=False)+'\n')
            raw.flush();cases.append(case);save(out/'cases.json',cases)
            print('ORACLE',layer,group,'native',case['native_relative_l2'],
                  {name:round(r['relative_l2'],6) for name,r in records.items()},flush=True)
            del prob_group,pp,sp,truth,z_est,wo_est

    if a.replay_dir:
        from types import SimpleNamespace
        begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try:
            for snapshot in sorted(a.replay_dir.glob('layer*_start*.pt')):
                data=torch.load(snapshot,map_location='cuda',weights_only=True)
                s=SimpleNamespace(**{k:data[k] for k in ('patches','specials','n','groups')})
                module=SimpleNamespace(proj=SimpleNamespace(weight=data['projection'],bias=data['bias']),scale=data['scale'])
                holder.update(module=module,qkv=data['qkv'],obs=data['obs'],start=data['start'])
                for group in data['targets']:
                    save(out/'status.json',dict(stage='oracle',layer=data['layer'],group=group))
                    audit(s,data['layer'],data['start'],group-data['start'],
                        *[data[k] for k in ('prior','pi','old_ids','allowed','z','reps','labels','population','geometry','uniform','result')])
                del data
            torch.cuda.synchronize()
            save(out/'result.json',dict(quality=quality,cases=len(cases),instrumented_seconds=time.perf_counter()-begin,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,scope='Offline frozen-QKV whole-mask oracle diagnostic'))
            save(out/'status.json',dict(stage='diagnostic_complete'))
        finally:raw.close()
        return

    original_prepare=core.prepare_qkv;original_observe=history.observe;original_select=window.select
    def prepare(module,x,pos):
        result=original_prepare(module,x,pos)
        holder.update(module=module,layer=modules[id(module)],qkv=tuple(t.to(result[2].dtype).contiguous() for t in result))
        return result
    def observe(s,q,k,start,g,scale):
        result=original_observe(s,q,k,start,g,scale);holder.update(obs=result,start=start);return result
    def select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None,**kwargs):
        result=original_select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=uniform,**kwargs)
        start=holder['start']
        targets=sorted(set([s.groups//4,3*s.groups//4]))
        if layer in layers:
            for group in targets:
                if start<=group<start+len(prior):
                    debug_window=next((w for w in getattr(s,'_reused_window_graphs',{}).values() if w.ids is result[0]),None)
                    if debug_window is not None:
                        save(out/'window_consistency.json',dict(layer=layer,group=group,
                            allowed_equal=bool(torch.equal(debug_window.inputs[3],allowed)),
                            labels_equal=bool(torch.equal(debug_window.constants[2],labels)),
                            reps_equal=bool(torch.equal(debug_window.constants[1],reps)),
                            prior_equal=bool(torch.equal(debug_window.inputs[0],prior))))
                    result=tuple(t.clone() if torch.is_tensor(t) else t for t in result)
                    if a.capture_only:
                        snapshot=out/f'layer{layer:02d}_start{start:04d}.pt'
                        if not snapshot.exists():
                            def cpu(value):
                                if torch.is_tensor(value):return value.detach().cpu()
                                if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                                if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
                                return value
                            torch.save(cpu(dict(layer=layer,start=start,targets=[g for g in targets if start<=g<start+len(prior)],
                                prior=prior,pi=pi,old_ids=old_ids,allowed=allowed,z=z,reps=reps,labels=labels,
                                population=population,geometry=geometry,uniform=uniform,result=result,qkv=holder['qkv'],obs=holder['obs'],
                                projection=holder['module'].proj.weight,bias=holder['module'].proj.bias,scale=holder['module'].scale,
                                patches=s.patches,specials=s.specials,n=s.n,groups=s.groups,per_frame=per_frame,config=config)),snapshot)
                        continue
                    save(out/'status.json',dict(stage='oracle',layer=layer,group=group))
                    audit(s,layer,start,group-start,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform,result)
        return result
    core.prepare_qkv=prepare;history.observe=observe;window.select=select
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(images)
        if a.capture_only:
            torch.save({key:prediction[key].detach().cpu() for key in ('pose_enc','depth','depth_conf')},out/'predictions.pt')
        torch.cuda.synchronize()
        save(out/'result.json',dict(quality=quality,cases=len(cases),instrumented_seconds=time.perf_counter()-begin,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,scope='Frozen-QKV oracle diagnostics, not model speed or geometry quality'))
        save(out/'status.json',dict(stage='diagnostic_complete'))
    finally:
        core.prepare_qkv=original_prepare;history.observe=original_observe;window.select=original_select;raw.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--output-dir' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output-dir')+1])
            if out.exists():
                from scripts.quality_metrics import quality_record,quality_markdown
                q=quality_record({},f'Failed oracle diagnostic: {type(exc).__name__}: {exc}')
                save(out/'quality.json',q);(out/'quality.md').write_text(quality_markdown(q))
                save(out/'status.json',dict(stage='failed',error=str(exc),traceback=traceback.format_exc()))
        raise
