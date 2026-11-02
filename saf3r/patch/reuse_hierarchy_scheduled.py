"""Fixed, no-GT layer schedule: low-error ends sparse, high-error middles Dense."""
from .reuse_residual_exploration import *
from . import reuse_hierarchy_grouped as grouped
import torch
DENSE_LAYERS=(4,5,6,7,8,10)

class ScheduledReuse(grouped.GroupedReuseResidualExploration):
    def forward(self,x,pos=None):
        o,s=self.original,self.state;c=s.config;GS=s.fast_group_size
        s.budget=math.floor(c.layer_retentions[self.layer]*s.n)-len(s.specials)
        if self.layer in DENSE_LAYERS:return dense_forward(self,x,pos)
        if self.training or len(x)!=1 or self.layer!=s.previous_layer+1:raise ValueError('Batch-one ordered inference required')
        q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
        with stage(s,'value_update'),torch.autocast(x.device.type,enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches]
            s.fine.update(features,self.layer)
        with stage(s,'pooling'),torch.autocast(x.device.type,enabled=False):
            population,means,pooled,overlap=s.prepare(k,features)
            if c.state_loss=="cosine_distortion":
                from .reuse_cosine_distortion import normalized_values,history_prior,cosine_states,transfer_weights,relative_logit_prior
                zvalues=normalized_values(features)
            geometry=None
            if c.compact and q.is_cuda:
                from saf3r.triton import reuse_compact as compact
                geometry=compact.prepare_geometry(zvalues,s.fine.labels,s.fine.reps,population)
            token_grid=None
            if c.history_mode=='all_relative_logit':
                token_grid=torch.zeros(s.n,device=x.device,dtype=torch.long)
                token_grid[s.patches]=s.grid
        s.parent_prepared=None
        if c.hierarchical and self.layer:
            from . import hierarchical_reuse as hierarchy
            with stage(s,'parent_structure_prepare'),torch.autocast(x.device.type,enabled=False):
                if bool(((zvalues.norm(dim=-1)-1).abs()<1e-3).all()):
                    s.parent_prepared=hierarchy.prepare(zvalues,s.fine.reps,s.fine.labels,geometry,c.parent_arity)
        order=None
        if c.query_shift:
            order=(torch.arange(s.n,device=x.device)+self.layer*c.query_shift%32)%s.n
            if self.layer:regroup_history(s,order)
            s.query_order=order
            q=q[:,order].contiguous()
        if c.history_mode=='dense_m5c':
            from . import reuse_m5c as m5c
            s.m5c_next={};s.m5c_reconstruction_qk=0;s.m5c_prior_errors=[];s.m5c_seed_errors=[];s.m5c_probe_qk=0
        output=torch.empty_like(v);history=[];weights=[];grids=[];real_grids=[];stats=[];capture=[];relative_evidence=[]
        candidates=max(1,math.ceil(s.fine.base_count*c.candidate_fraction))
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);first=start*GS;end=min(s.n,(start+g)*GS)
            if c.history_mode=='dense_m5c':
                with stage(s,'m5c_coarse_observation'),torch.autocast(x.device.type,enabled=False):
                    m5c_observation=m5c.observe(s,q,k,start,g,o.scale)
            if self.layer==0:
                seed=torch.arange(s.budget,device=x.device)*s.fine.base_count//s.budget
                ids=s.fine.reps[seed][None].expand(g,-1).contiguous()
                pops=population[seed][None].expand(g,-1).contiguous();real=torch.zeros_like(ids,dtype=torch.bool)
                record=dict(reuse_edges=torch.zeros(g,device=x.device),representatives=torch.full((g,),s.budget,device=x.device),
                    exact_edges=torch.zeros(g,device=x.device),dropped_clusters=torch.full((g,),s.fine.base_count-s.budget,device=x.device),
                    counts=torch.full((g,),s.budget,device=x.device),history_cap_discarded=torch.zeros(g,device=x.device))
            else:
                probe=(torch.arange(start,start+g,device=x.device)[:,None]*GS+
                       torch.tensor([GS//8,3*GS//8,5*GS//8,7*GS//8],device=x.device)).clamp_max(s.n-1)
                pq=q[:,probe]
                with stage(s,'coarse_observation'):
                    z=(pq@pooled[:,None].transpose(-1,-2)).float()*o.scale+s.grid_sizes.log()
                    normalizer=z.logsumexp(-1,keepdim=True)
                    probability=z.softmax(-1).mean((0,2))
                    previous=s.previous_grid[start:start+g].float()
                    beta=torch.where(previous>1e-12,(probability/previous.clamp_min(1e-12)).clamp_max(1),0)
                    demand=(probability-beta*s.previous_real_grid[start:start+g].float()).clamp_min(0)
                with stage(s,'residual_fine_probe'),torch.autocast(x.device.type,enabled=False):
                    pi=torch.sparse.mm(overlap,probability.T).T.contiguous()
                    residual=torch.sparse.mm(overlap,demand.T).T
                    candidate=residual.topk(candidates,dim=1).indices
                    allowed=torch.zeros_like(pi,dtype=torch.bool).scatter_(1,candidate,True)
                    if not c.compact:
                        ck=k[:,s.patches[s.fine.reps[candidate]]]
                        logits=(pq@ck.transpose(-1,-2)).float()*o.scale+population[candidate].log()[None,:,None]
                        fine_mass=(logits-normalizer).exp().mean((0,2))
                        pi_hat=pi.clone().scatter_(1,candidate,fine_mass)
                        pi_hat/=pi_hat.sum(1,keepdim=True).clamp_min(1e-30)
                with stage(s,'reuse_and_budget'),torch.autocast(x.device.type,enabled=False):
                    old_ids=None if s.history_ids is None else s.history_ids[start:start+g]
                    old_weights=None if s.history_weights is None else s.history_weights[start:start+g]
                    if c.history_mode=='dense_m5c':
                        prior,pi=m5c.build_prior(s,start,old_ids,m5c_observation)
                        old_weights=prior.gather(1,old_ids.long().clamp_min(0)).masked_fill(old_ids<0,0)
                        history_beta=torch.ones_like(beta)
                    elif c.history_mode=='all_relative_logit':
                        prior,pi=relative_logit_prior(probability,s.grid,s.grid_sizes,old_ids,
                            s.history_relative[start:start+g],s.fine.labels,s.fine.base_count)
                        old_weights=prior.gather(1,old_ids.long().clamp_min(0)).masked_fill(old_ids<0,0)
                        history_beta=torch.ones_like(beta)
                    else:history_beta=beta
                    if geometry is not None:
                        hids,discarded=compact.top_history(old_ids,prior,s.fine.reps,geometry,c.history_per_cluster)
                    else:hids,hmass,discarded=top_history(old_ids,old_weights,history_beta,s.grid,s.fine.labels,s.fine.reps,c.history_per_cluster)
                    if c.state_loss=="cosine_distortion":
                        if c.history_mode not in ('all_relative_logit','dense_m5c'):
                            prior=history_prior(pi,s.fine.labels,population,old_ids,old_weights,beta,s.grid,c.prior_floor_fraction)
                        if geometry is not None:
                            errors,costs=cosine_states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier,geometry=geometry)
                        else:errors,costs=cosine_states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier)
                    else:
                        errors,costs=operating_points(pi,pi_hat,means,features[s.fine.reps],population,
                                                    hids,hmass,features,allowed)
                    # Subtracting the Rep baseline is an exact gain reformulation.
                    # Keep the same feasible discrete dual solve (not a greedy exchange).
                    objective=geometry['objective'] if geometry is not None else (errors-errors[:,:,1:2] if c.rep_gain else errors)
                    if geometry is not None:
                        selected,count=compact.budget(objective,costs,errors[:,:,0].amax(1)+1e-12,s.budget,c.dual_iterations)
                    else:
                        selected,count=budget_states(objective,costs,s.budget,c.dual_iterations,maximize=c.allocation_objective=="max_error",
                            lambda_upper=errors[:,:,0].amax(1)+1e-12 if c.rep_gain else None)
                    if s.parent_prepared is not None:
                        with stage(s,'parent_joint_allocation'):
                            selected,parent_owners,parent_steps=hierarchy.allocate(errors,costs,pi,zvalues,s.fine.reps,hids,prior,selected,s.parent_prepared,s.budget,c.dual_iterations,c.parent_passes,geometry,c.parent_moment_mode)
                    if geometry is not None:
                        ids,real,record=compact.pack(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                    else:
                        ids,pops,real,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                    if c.state_loss=="cosine_distortion":
                        if c.mass_transfer:
                            if geometry is not None:pops=compact.transfer(selected,s.fine.labels,s.fine.reps,hids,prior,pi,ids,geometry)
                            else:pops,_=transfer_weights(selected,zvalues,s.fine.reps,s.fine.labels,hids,prior,ids)
                        if s.parent_prepared is not None:
                            with stage(s,'parent_cross_cluster_transfer'):
                                pops,parent_record=hierarchy.transfer(pops,ids,prior,pi,parent_owners,hids)
                                record.update(parent_record)
                                record['parent_proxy_loss']=parent_steps[-1]['proxy_loss']
                                record['parent_baseline_proxy_loss']=parent_steps[0]['baseline_proxy_loss']
                                record['parent_allocation_fallbacks']=torch.stack([step['fallback'] for step in parent_steps]).sum(0)
                                record['truly_dropped_clusters']=(selected==0).sum(1)-parent_record['delegated_clusters']
                        limit=c.history_per_cluster
                        if geometry is not None:
                            record.update(compact.diagnostics(errors,selected,hids,old_ids,s.fine.labels,s.fine.reps))
                        else:
                            reused=torch.where((selected>=2)&(selected<=limit+1),selected-1,
                                torch.where((selected>=limit+2)&(selected<=2*limit+1),selected-limit-1,0))
                            for amount in range(limit+1):record[f'selected_history_count_{amount}_clusters']=(reused==amount).sum(1)
                    if c.rep_gain and geometry is None:
                        observed=(old_ids>=0)
                        overlap_rep=observed&(old_ids.long().clamp_min(0)==s.fine.reps[s.fine.labels[old_ids.long().clamp_min(0)]])
                        gains=errors[:,:,1:2]-errors
                        finite=torch.isfinite(gains)
                        record.update(observed_history=observed.sum(1),history_rep_overlap=overlap_rep.sum(1),
                            history_candidates=(hids>=0).sum((1,2)),
                            positive_gain_states=((gains>0)&finite).sum((1,2)),
                            positive_gain_sum=gains.masked_fill(~finite,0).clamp_min(0).sum((1,2)),
                            negative_gain_sum=gains.masked_fill(~finite,0).clamp_max(0).sum((1,2)),
                            selected_gain=gains.gather(2,selected[:,:,None]).sum((1,2)),
                            gain_min=gains.masked_fill(~finite,torch.inf).amin((1,2)),
                            gain_max=gains.masked_fill(~finite,-torch.inf).amax((1,2)))
                        for state_id in range(gains.shape[-1]):
                            present=finite[:,:,state_id]
                            record[f'state_{state_id}_clusters']=(selected==state_id).sum(1)
                            record[f'gain_{state_id}']=gains[:,:,state_id].masked_fill(~present,0).sum(1)/present.sum(1).clamp_min(1)
                    record['history_cap_discarded']=discarded
                    if geometry is None:
                        record['proxy_drop_error']=(errors[:,:,0]).sum(1)
                        record['proxy_selected_error']=errors.gather(2,selected[:,:,None]).sum((1,2))
            actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous()
            if c.disable_population_compensation:
                pops=(ids>=0).to(dtype=pops.dtype)
            bias=pops.log().contiguous()
            with stage(s,'execution_and_evidence'):
                if q.is_cuda:
                    from saf3r.triton.reuse_grouped_attention_fast import weighted_attention
                    reference=None
                    if c.history_mode=='all_relative_logit':
                        reference=(pooled,token_grid)
                    execution=weighted_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale,
                        single_qk=c.evidence_mode=="saved_logits",relative_reference=reference,group_size=GS)
                    if reference is None:result,mass=execution
                    else:
                        result,mass,delta=execution
                        relative_evidence.append(delta)
                else:
                    result,mass=reference_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale)
                    if c.history_mode=='all_relative_logit':
                        safe=actual.long().clamp_min(0)
                        raw=(q[:,first:end].float().mean(1)[:,None,:]*k[:,safe].float()).sum(-1)*o.scale
                        ref=(q[:,first:end].float().mean(1)[:,None,None,:]*pooled[:,s.grid[ids.clamp_min(0)]].float()).sum(-1)*o.scale
                        relative_evidence.append((raw-ref).mean(0).masked_fill(ids<0,0))
            output[:,first:end]=result
            if c.history_mode=='dense_m5c':
                with stage(s,'m5c_evidence_and_host_copy'),torch.autocast(x.device.type,enabled=False):
                    m5c.save_evidence(s,q,k,start,ids,actual,m5c_observation,o.scale)
            with stage(s,'history_summary'):
                mass=mass/mass.sum(1,keepdim=True).clamp_min(1e-30)
                key_grid=s.grid[ids.clamp_min(0)]
                grid=torch.zeros((g,s.regions),device=x.device).scatter_add_(1,key_grid,mass)
                # A selected history Key may represent other members. Only its
                # unit-Key probability is genuine evidence, not transferred mass.
                evidence=mass/pops.clamp_min(1) if c.state_loss=="cosine_distortion" else mass
                if c.history_mode!='genuine_probability':real=ids>=0
                real_grid=torch.zeros_like(grid).scatter_add_(1,key_grid,evidence*real)
                history.append(ids.masked_fill(~real,-1).int());weights.append((evidence*real).half())
                grids.append(grid.half());real_grids.append(real_grid.half())
            stats.append(record)
            if c.capture:capture.append(dict(ids=actual.clone(),population=pops.clone(),genuine=real.clone(),
                                             mass=mass.clone(),reuse=record['reuse_edges'].clone()))
        s.history_ids=torch.cat(history);s.history_weights=torch.cat(weights)
        if relative_evidence:s.history_relative=torch.cat(relative_evidence)
        if c.history_mode=='dense_m5c':
            if s.m5c_previous:raise RuntimeError('Unconsumed M5c history windows')
            s.m5c_previous=s.m5c_next;s.m5c_next={}
            s.m5c_previous_q=q.detach().clone();s.m5c_previous_k=k.detach().clone();s.m5c_previous_scale=o.scale
            s.history_pooled_fp32=None
        if c.rep_gain:
            # Layer 0 has no candidate allocation; keep a consistent diagnostic schema.
            diagnostic_keys=('observed_history','history_rep_overlap','history_candidates','positive_gain_states',
                'positive_gain_sum','negative_gain_sum','selected_gain','gain_min','gain_max')
            for record in stats:
                for key in diagnostic_keys:record.setdefault(key,torch.zeros_like(record['counts']))
                for state_id in range(2*c.history_per_cluster+3):
                    record.setdefault(f'state_{state_id}_clusters',record['counts'] if state_id==1 else torch.zeros_like(record['counts']))
                    record.setdefault(f'gain_{state_id}',torch.zeros_like(record['counts']))
        s.previous_grid=torch.cat(grids);s.previous_real_grid=torch.cat(real_grids)
        s.previous_layer=self.layer
        joined={key:torch.cat([r[key] for r in stats]) for key in stats[0]}
        rows=(s.n-torch.arange(s.groups,device=x.device)*GS).clamp_max(GS)
        s.records.append(dict(layer=self.layer,query_offset=(self.layer*c.query_shift)%32,patch_budget=s.budget,special_keys=len(s.specials),
            fine_clusters=s.fine.base_count,canonical_regions=s.regions,
            **{key+'_mean':val.float().mean() for key,val in joined.items()},
            actual_unique_density=((joined['counts']+len(s.specials))*rows).sum()/s.n**2,
            unused_budget_mean=(s.budget-joined['counts']).float().mean(),
            canonical_qk_per_head=0 if self.layer==0 else 4*s.groups*s.regions,
            fine_probe_qk_per_head=0 if self.layer==0 or c.compact else 4*s.groups*candidates,
            compact=c.compact,geometry_width=0 if geometry is None else geometry['width'],
            geometry_cache_bytes=0 if geometry is None else sum(geometry[name].numel()*geometry[name].element_size() for name in ('members','rank','distances','rep_distance')),
            final_unique_qk_per_head=((joined['counts']+len(s.specials))*rows).sum(),
            evidence_extra_qk_per_head=(joined['counts']*rows).sum() if c.evidence_mode=='second_qk' else 0,
            evidence_scratch_peak_bytes=min(c.window_groups*32,s.n)*q.shape[0]*s.budget*4 if c.evidence_mode=='saved_logits' else 0,
            stored_history_bytes=s.history_ids.numel()*4+s.history_weights.numel()*2,
            stored_grid_bytes=s.previous_grid.numel()*4,
            population_compensation=not c.disable_population_compensation and (c.state_loss=="output_proxy" or not c.mass_transfer),head_aware=False,
            history_mode=c.history_mode,drop_multiplier=c.drop_multiplier,rep_gain=c.rep_gain,
            relative_logit_evidence_bytes=0 if s.history_relative is None else s.history_relative.numel()*4,
            state_loss=c.state_loss,mass_transfer="estimated assigned mass / own prior mass" if c.state_loss=="cosine_distortion" and c.mass_transfer else "original population weights",
            selector=('Rep-relative weighted cosine distortion, discrete dual budget search; not exact knapsack' if c.rep_gain else 'separable vector-output proxy, discrete dual budget search; not exact knapsack'),
            history=c.history_mode+'; compact relative logits FP32, probabilities FP16; no fixed retention ratio',
            evidence=c.evidence_mode,allocation_objective=c.allocation_objective,query_shift=c.query_shift))
        if c.clustering_mode!='source_anchored':s.records[-1]['clustering']=s.fine.records[-1]
        if c.hierarchical:
            s.records[-1].update(hierarchical=True,parent_arity=c.parent_arity,parent_passes=c.parent_passes,parent_moment_mode=c.parent_moment_mode,parent_active=s.parent_prepared is not None,
                parent_groups=0 if s.parent_prepared is None else len(s.parent_prepared['parents']),
                parent_fallback_reason=('L0 retains the native Rep seed' if not self.layer else 'non-unit or zero normalized Value; native flat allocation retained') if s.parent_prepared is None else None,
                state0_meaning='no locally emitted keys; delegated vs truly ignored clusters recorded separately',
                exact_parent_proxy='child Rep and most distant member, both real emitted Exact keys')

        if c.history_mode=='dense_m5c':
            s.records[-1].update(history_estimator='D2 + M5c',beta_down_exponent=1.,beta_up_exponent=2.,
                probability_normalization='all Keys including specials; no patch renormalization',
                history_cache_device='cpu_grid_statistics_gpu_previous_qk',history_cache_dtype='float32_statistics_model_qk_dtype',
                history_estimator_head_aware=True,history='D2 full-Key probabilities reconstructed exactly from previous Q/K; M5c beta(1,2)',
                history_gpu_reconstruction_bytes=s.m5c_previous_q.numel()*s.m5c_previous_q.element_size()+s.m5c_previous_k.numel()*s.m5c_previous_k.element_size(),
                extra_previous_layer_probe_qk_per_head=s.m5c_reconstruction_qk,history_cpu_bytes=m5c.host_bytes(s.m5c_previous),
                seed_normalization_max_error=torch.stack(s.m5c_seed_errors).max(),
                prior_patch_mass_max_error=torch.stack(s.m5c_prior_errors).max() if s.m5c_prior_errors else torch.tensor(0.,device=x.device),
                extra_known_probe_qk_per_head=s.m5c_probe_qk,extra_coarse_probe_qk_per_head=4*s.groups*(s.regions+len(s.specials)))
        if c.capture:s.captured[self.layer]=dict(windows=capture,specials=s.specials.clone(),
            query_order=None if order is None else order.clone(),canonical=s.grid.clone(),fine_labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
        if order is not None:
            restored=torch.empty_like(output);restored[:,order]=output;output=restored
        camera=s.specials.reshape(s.frames,5)[:,0]
        with stage(s,'exact_camera_queries'):
            dense_camera=torch.nn.functional.scaled_dot_product_attention(q[:,camera][None],k[None],v[None],scale=o.scale)[0]
            output[:,camera]=dense_camera
        count=joined['counts'];extra=(s.n-count[camera//GS]-len(s.specials)).sum()
        s.records[-1]['final_unique_qk_per_head']+=extra
        s.records[-1].update(execution_backend='hierarchical_scheduled_fast',layer_nominal_retention=c.layer_retentions[self.layer],layer_execution='sparse',state_selection_scope='actual sparse support',routing_group_size=GS,attention_query_tile=64,camera_queries_dense=True,camera_dense_extra_edges=extra,history_mass_scope='four unique deterministic probes, all heads; not full query-group mean',
            m5c_seed_reuses_execution_probe_qk=True,evidence_extra_qk_per_head=4*joined['counts'].sum(),
            actual_unique_density=s.records[-1]['actual_unique_density']+extra/s.n**2,
            history_cache_device='gpu_statistics_gpu_previous_qk',history_gpu_statistics_bytes=sum(t.numel()*t.element_size() for w in s.m5c_previous.values() for t in w.values()),history_cpu_bytes=0,
            routing_approximation='four M5c probes per enlarged contiguous query group; output and softmax remain per query')
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install(group_size=2048):
    grouped.install(group_size)
    from . import reuse_residual_exploration as implementation
    implementation.ReuseResidualExploration=ScheduledReuse

def dense_forward(self,x,pos=None):
    from . import reuse_m5c
    from saf3r.triton.history_probe_logits import selected_probe_logits
    o,s=self.original,self.state;c=s.config
    if self.training or len(x)!=1 or self.layer!=s.previous_layer+1:raise ValueError('Ordered inference required')
    q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
    with stage(s,'value_update'),torch.autocast(x.device.type,enabled=False):
        features=v.mean(0,dtype=torch.float32)[s.patches];s.fine.update(features,self.layer)
    with stage(s,'pooling'),torch.autocast(x.device.type,enabled=False):population,means,pooled,overlap=s.prepare(k,features)
    if s.history_ids is None:raise ValueError('Scheduled Dense layers require a retained preceding support')
    s.m5c_previous.clear();s.m5c_next={};s.m5c_seed_errors=[];s.m5c_probe_qk=0;s.m5c_reconstruction_qk=0
    grids=[];real_grids=[];weights=[]
    with stage(s,'dense_layer_history_refresh'),torch.autocast(x.device.type,enabled=False):
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);obs=grouped.observe(s,q,k,start,g,o.scale)
            ids=s.history_ids[start:start+g];valid=ids>=0;regions=s.grid[ids.long().clamp_min(0)]
            actual=s.patches[ids.long().clamp_min(0)].masked_fill(~valid,-1).int().contiguous()
            count=torch.zeros((g,s.regions),device=x.device).scatter_add_(1,regions,valid.float());remaining=s.grid_sizes[None]-count
            if bool((remaining<0).any()):raise ValueError('Retained history duplicates grid members')
            raw=selected_probe_logits(q,k,actual,obs['positions'],o.scale)
            missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]==0,-torch.inf)
            logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),raw.logsumexp(-1)),obs['special_logits'].logsumexp(-1))[:,:,:,None]
            p=(raw-logz).exp();uniform=(obs['grid_logits']-logz).exp()
            known=torch.zeros_like(uniform).scatter_add_(-1,regions[:,None,None,:].expand_as(p),p)
            corrected=uniform*remaining[:,None,None,:]+known;special=(obs['special_logits']-logz).exp()
            s.m5c_seed_errors.append((corrected.sum(-1)+special.sum(-1)-1).abs().max())
            cache=dict(history_dense_log_normalizer=logz,previous_coarse_grid_probability=obs['coarse_grid_probability'],previous_corrected_grid_probability=corrected,previous_corrected_special_probability=special,unobserved_uniform_probability=uniform)
            s.m5c_next[start]={key:value.detach() for key,value in cache.items()}
            grid=reuse_m5c.mean_probes(corrected,obs['weights']);patch_mass=grid.sum(-1,keepdim=True).clamp_min(1e-30)
            grids.append((grid/patch_mass).half());real_grids.append((reuse_m5c.mean_probes(known,obs['weights'])/patch_mass).half())
            weights.append((reuse_m5c.mean_probes(p,obs['weights'])/patch_mass).masked_fill(~valid,0.).half());s.m5c_probe_qk+=int(valid.sum())*4
    with stage(s,'exact_dense_layer'):
        output=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None],scale=o.scale)[0]
    s.history_weights=torch.cat(weights);s.previous_grid=torch.cat(grids);s.previous_real_grid=torch.cat(real_grids)
    s.m5c_previous=s.m5c_next;s.m5c_next={};s.m5c_previous_q=q.detach().clone();s.m5c_previous_k=k.detach().clone();s.m5c_previous_scale=o.scale;s.history_pooled_fp32=None;s.previous_layer=self.layer
    s.records.append(dict(layer=self.layer,execution_backend='hierarchical_scheduled_fast',layer_execution='dense',layer_nominal_retention=1.,actual_unique_density=1.,final_unique_qk_per_head=s.n*s.n,
        fine_clusters=s.fine.base_count,fine_partition_retention=c.retention,canonical_regions=s.regions,routing_group_size=s.fast_group_size,patch_budget=None,special_keys=len(s.specials),
        state_selection_unavailable='Dense layer executes no budget solver or sparse state selection',history_retention='Previous sparse history subset retained; all its keys genuinely executed by current Dense Attention',
        history_mass_scope='four unique deterministic probes, all heads',history_estimator='D2+M5c',history_cache_device='gpu_statistics_gpu_previous_qk',history_cpu_bytes=0,
        history_gpu_statistics_bytes=sum(t.numel()*t.element_size() for w in s.m5c_previous.values() for t in w.values()),seed_normalization_max_error=torch.stack(s.m5c_seed_errors).max(),prior_patch_mass_max_error=None,
        extra_known_probe_qk_per_head=s.m5c_probe_qk,extra_previous_layer_probe_qk_per_head=0,evidence_extra_qk_per_head=0,clustering=s.fine.records[-1]))
    return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))
