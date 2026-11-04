"""Independent same-input diagnostic operators; none feed back into the model."""
import ast,inspect,math,statistics,textwrap,time
from pathlib import Path
import torch,triton
from scripts.direct_detail_profile import csvout,save

def timed(fn,repeats=7):
 for _ in range(2):result=fn()
 samples=[]
 for _ in range(repeats):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();result=fn();en.record();en.synchronize();samples.append(st.elapsed_time(en)/1000)
 return statistics.median(samples),result

def probability(rec,out):
 from saf3r.triton import reuse_direct_sparse_attention as direct
 from saf3r.triton import reuse_probability_history as history
 from saf3r.triton import reuse_direct_legacy_observation as legacy
 args,kw=rec.snapshot;q,k,v,ids,special,scale=args[:6];g=min(4,len(ids));r=min(q.shape[1],g*32);q=q[:,:r];ids=ids[:g].clone();bias=kw['logit_bias'][:g].clone();pos=kw['probe_positions'][:g].clone();h=q.shape[0];d=q.shape[-1];a=ids.shape[1];s=len(special)
 def execute(qv=q,positions=pos,export=True):
  return direct.direct_sparse_attention(qv,k,v,ids,special,scale,logit_bias=bias,probe_positions=positions,return_history_probability=export,history_export='debiased_log_lse_probe',num_warps=4,num_stages=2)
 base_t,(av,evidence)=timed(lambda:execute(export=False));lse=evidence['lse'];rows=[dict(variant='Direct AV only',seconds=base_t,probe_qk_edges=0,kernels=1,output_bytes=av.numel()*av.element_size()+lse.numel()*4,probability_relative_l1=None,scope='Independent microbenchmark;4 fixed groups; full K/V bank; no writeback')]
 truth=[]
 with torch.autocast('cuda',enabled=False):
  for group in range(g):
   logits=q[:,pos[group]].float()@k.float().transpose(-1,-2)*scale
   dense=logits.softmax(-1);actual=ids[group].long().clamp_min(0);truth.append(dense[:,:,actual].mean(0).masked_fill(ids[group][None]<0,0.))
 target=torch.stack(truth)
 def probe(qv,positions,normalizer):
  p=positions.shape[1];output=torch.empty((g,p,s+a),device=q.device,dtype=torch.float32)
  direct._probe[(triton.cdiv(s+a,128),g)](qv,k,ids,special,bias,positions,normalizer,output,h,r,k.shape[1],a,s,d,qv.stride(0),k.stride(0),scale,p,max(16,triton.next_power_of_2(d)),128,True,True,num_warps=4,num_stages=1)
  return output
 estimate4=None
 for name,qv,positions,normalizer in [('4 Probe export',q,pos,lse),('1 Probe export',q,pos[:,:1].contiguous(),lse)]:
  seconds,value=timed(lambda:probe(qv,positions,normalizer));estimate=value[:,:,s:].exp();tar=target if len(positions[0])==4 else target[:,:1]
  error=float((estimate-tar).abs().sum()/tar.sum().clamp_min(1e-30))
  rows.append(dict(variant=name,seconds=seconds,probe_qk_edges=g*h*positions.shape[1]*(s+a),logical_qk_flops=2*g*h*positions.shape[1]*(s+a)*d,kernels=1,output_bytes=value.numel()*4,probability_relative_l1=error,scope='QK+LSE read+logmeanexp+store are fused; allocations included in wrapper event interval'))
  if name.startswith('4'):estimate4=estimate
 meanq=q.reshape(h,g,32,d).mean(2,keepdim=True).expand(-1,-1,32,-1).reshape(h,r,d).contiguous();meanpos=(torch.arange(g,device=q.device)*32)[:,None].int()
 _,meane=execute(meanq,meanpos,False);seconds,value=timed(lambda:probe(meanq,meanpos,meane['lse']))
 targetmean=target.mean(1,keepdim=True);error=float((value[:,:,s:].exp()-targetmean).abs().sum()/targetmean.sum())
 rows.append(dict(variant='Mean Query independent QK',seconds=seconds,probe_qk_edges=g*h*(s+a),logical_qk_flops=2*g*h*(s+a)*d,kernels=1,output_bytes=value.numel()*4,probability_relative_l1=error,scope='Mean Q and its own sparse LSE prepared outside timing; compares to four-probe Dense mean; approximation only'))
 combo,_=timed(lambda:execute());rows.append(dict(variant='Direct AV +4 Probe',seconds=combo,probe_qk_edges=g*h*4*(s+a),kernels=2,output_bytes=(g*4*(s+a)+h*r)*4+av.numel()*av.element_size(),scope='Two kernels and wrapper allocation; same QKV/mask/bias'))
 # Reproduce legacy all32 observation on the same support and pooled keys.
 state=rec.state;grid=state.grid;regions=grid[torch.zeros_like(ids).long()] # overwritten using global-patch inverse below
 inverse=torch.full((k.shape[1],),-1,device=q.device,dtype=torch.long);inverse[state.patches]=torch.arange(len(state.patches),device=q.device)
 relative=inverse[ids.long().clamp_min(0)];regions=grid[relative.clamp_min(0)].int().contiguous()
 pooled=torch.zeros((h,state.regions,d),device=q.device);pooled.index_add_(1,grid,k[:,state.patches].float());pooled/=state.grid_sizes[None,:,None]
 pq=q[:,pos];base=torch.einsum('hgpd,hrd->ghpr',pq.float(),pooled)*scale
 norm=torch.linalg.vector_norm(k,dim=-1,dtype=torch.float32).amax(-1)
 sec,_=timed(lambda:legacy.observe(q,k,ids,regions,bias,pos,lse,base,state.grid_sizes,scale,norm,pq))
 rows.append(dict(variant='Legacy all32 history observation',seconds=sec,probe_qk_edges=g*h*32*a,additional_probe_edges='Legacy conditional fallback may execute additional four-probe QK',kernels='Multiple;see source',probability_relative_l1=None,scope='Actual legacy observation implementation; no model writeback'))
 # Grid conditional comparison against exact Dense on the same probes.
 sizes=state.grid_sizes;cache=torch.empty((g,1,4,a),device=q.device,dtype=torch.float16)
 missing=torch.logsumexp(base-lse[:,pos].permute(1,0,2)[:,:,:,None],1)-math.log(h)
 scores=estimate4.clamp_min(1e-30).log()
 sec,(_,counts,den)=timed(lambda:history.store(scores,missing,relative.int().masked_fill(ids<0,-1),grid,sizes,cache))
 dense_grid=torch.zeros((g,4,state.regions),device=q.device)
 with torch.autocast('cuda',enabled=False):
  for group in range(g):
   pp=(q[:,pos[group]].float()@k.float().transpose(-1,-2)*scale).softmax(-1).mean(0)[:,state.patches]
   dense_grid[group].scatter_add_(1,grid[None].expand(4,-1),pp)
 truthcond=target/dense_grid.gather(2,regions.long().clamp_min(0)[:,None].expand(-1,4,-1)).clamp_min(1e-30)
 cond=cache[:,0].float().exp().masked_fill(ids[:,None]<0,0.)
 rows.append(dict(variant='4 Probe Grid conditional and cache',seconds=sec,probe_qk_edges=0,output_bytes=cache.numel()*2,probability_relative_l1=float((cond-truthcond).abs().sum()/truthcond.sum().clamp_min(1e-30)),scope='Same-input conditional diagnostic; score exponent/log introduces extra roundoff; model cache unchanged'))
 csvout(out/'probability_variants.csv',rows)
 save(out/'probability_micro_input.json',dict(layer=12,groups=g,q_shape=list(q.shape),k_shape=list(k.shape),ids_shape=list(ids.shape),dtype=str(q.dtype),same_input=True,selected_slots_including_padding=a,specials=s,edge_count_kind='Logical valid capacity bound; padding may skip valid loads, all dot tiles still scheduled'))

def components(rec,out):
 from saf3r.triton import reuse_fused_states as fused
 from saf3r.triton import reuse_compact as compact
 from scripts.direct_detail_profile import _stamp
 window=rec.state_snapshot
 if isinstance(window,(list,tuple)):
  args=list(window);prior,z,reps,labels,pop,allowed,geometry,context=args
  g=min(4,len(prior));prior=prior[:g];allowed=allowed[:g];context={k:(v[:g] if torch.is_tensor(v) and v.ndim and v.shape[0]==len(args[0]) else v) for k,v in context.items()}
 else:
  from saf3r.patch.reuse_inference_ops import gate
  prior,pi,old_ids,allowed=window.inputs;z,reps,labels,pop=window.constants;geometry=window.geometry;g=min(4,len(prior));prior=prior[:g];allowed=allowed[:g];context=gate(prior,pi[:g],pop,old_ids[:g],labels,reps)
 # Re-execute exact current fused producer for owner/ground-reference outputs.
 seconds,(errors,costs)=timed(lambda:fused.states(prior,z,reps,labels,pop,allowed,geometry,context),3)
 n,d=z.shape;f=len(reps);rows=[dict(component='Fused State Loss',seconds=seconds,kind='Production fused kernel and wrapper on fixed input',mathematical_change=False)]
 # Derive diagnostic kernels from identical source. Unused values are eliminated
 # by Triton; each writes one output, so independent costs are not additive.
 source=(getattr(fused._states,'kernel',None) or fused._states).src
 module=__import__('saf3r.triton.reuse_fused_states',fromlist=['_states']);namespace=module.__dict__.copy();namespace['_stamp']=_stamp
 mode={'Drop Moment/Loss':'drop','Rep Loss':'rep','History Distance Gather':'tl.sum(tl.sum(tl.where(distance < float("inf"), distance, 0.),2),1)','History Nearest Owner':'tl.sum(tl.where(has,owner,0).to(tl.float32),1)','History Loss':'hist','History Eligible Mass':'tl.sum(tl.where(own,weight,0.),1)','History Value Contribution and Remaining Moment':'tl.sum(hv_diag*hv_diag+remaining_diag*remaining_diag,1)','Exact Eligibility':'exact'}
 for index,(name,expr) in enumerate(mode.items()):
  tree=ast.parse(textwrap.dedent(source));fn=tree.body[0];fn.decorator_list=[];fn.name='_diagnostic_component_'+str(index);fn.args.args.insert(0,ast.arg(arg='OUT'))
  fn.body=[node for node in fn.body if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='tl.store')]
  if name=='History Value Contribution and Remaining Moment':fn.body+=ast.parse('hv_diag=tl.sum(tl.where(own[:,:,None],weight[:,:,None]*values,0.),1)\nremaining_diag=moment-hv_diag').body
  fn.body+=ast.parse('tl.store(OUT+g*F+c, '+expr+', ci<COUNT)').body;ast.fix_missing_locations(tree);file=out/(fn.name+'.py');file.write_text(ast.unparse(tree));exec(compile(tree,str(file),'exec'),namespace);kernel=triton.jit(namespace[fn.name]);output=torch.empty((g,f),device=prior.device)
  def execute():
   for bi,(size,clusters) in enumerate(geometry['drop_bins']):
    c=min(16,max(1,128//size));pointer='drop_bin_counts' in geometry;count=geometry['drop_bin_counts'][bi] if pointer else len(clusters)
    kernel[(g,triton.cdiv(len(clusters),c))](output,prior,z,geometry['members'],geometry['distances'],geometry['rep_distance'],context['qualified'],context['remove_rep'],context['eligible_count'],allowed.contiguous(),pop,clusters,errors,costs,geometry['objective'],context['owners'],n,f,d,geometry['width'],count,c,size,triton.next_power_of_2(d),pointer,num_warps=8 if size>=16 else 4,enable_fp_fusion=False)
   return output
  sec,result=timed(execute,3);row=dict(component=name,seconds=sec,kind='Isolated diagnostic kernel; compiler eliminates unrelated fused work; costs do not sum to fused time',mathematical_change=False)
  if name=='Drop Moment/Loss':row['max_abs_error']=float((result-errors[:,:,0]).abs().max())
  if name=='Rep Loss':
   valid=torch.isfinite(errors[:,:,1]);row['max_abs_error']=float((result[valid]-errors[:,:,1][valid]).abs().max())
  if name=='History Loss':
   valid=torch.isfinite(errors[:,:,2]);row['max_abs_error']=float((result[valid]-errors[:,:,2][valid]).abs().max())
  rows.append(row)
 # Shared geometry is ALREADY present in this implementation. There is no
 # redundant per-group recomputation of pairwise member distances to remove.
 rows.append(dict(component='Shared Cluster Geometry (existing)',seconds=None,kind='Already production prepare_geometry once per layer; same fused kernel consumes it; no new algorithm',mathematical_change=False))
 # Exact cluster-grid sufficient-statistic contraction: baseline per-grid plus
 # arbitrary residual. Fixed grid + prior, arbitrary historical residual kept.
 grid=rec.state.grid;pair=labels*rec.state.regions+grid;unique,inverse=pair.unique(sorted=True,return_inverse=True);pc=unique//rec.state.regions;pr=unique%rec.state.regions
 sums=torch.zeros((len(unique),d),device=z.device).index_add_(0,inverse,z);dsum=torch.zeros(len(unique),device=z.device).scatter_add_(0,inverse,geometry['rep_distance'])
 baseline=torch.full((g,rec.state.regions),float('inf'),device=z.device).scatter_reduce_(1,grid[None].expand(g,-1),prior,reduce='amin',include_self=True);delta=prior-baseline[:,grid]
 def sufficient():
  weight=baseline[:,pr];mom=torch.zeros((g,f,d),device=z.device);rep=torch.zeros((g,f),device=z.device)
  mom.index_add_(1,pc,weight[:,:,None]*sums[None]);rep.index_add_(1,pc,weight*dsum[None])
  for begin in range(0,n,8192):
   end=min(n,begin+8192);mom.index_add_(1,labels[begin:end],delta[:,begin:end,None]*z[None,begin:end]);rep.index_add_(1,labels[begin:end],delta[:,begin:end]*geometry['rep_distance'][None,begin:end])
  return mom.square().sum(-1).sqrt(),rep
 sec,(drop,rep)=timed(sufficient,3);valid=torch.isfinite(errors[:,:,1]);maxerr=max(float((drop-errors[:,:,0]).abs().max()),float((rep[valid]-errors[:,:,1][valid]).abs().max()))
 candidate=errors.clone();candidate[:,:,0]=drop;candidate[:,:,1]=rep.masked_fill(~valid,float('inf'))
 objective=geometry['objective'][:g];candidate_objective=candidate-rep[:,:,None];selected,_=compact.budget(objective,costs,errors[:,:,0].amax(1)+1.e-12,rec.state.budget,16);proposed,_=compact.budget(candidate_objective,costs,drop.amax(1)+1.e-12,rec.state.budget,16)
 rows.append(dict(component='Cluster-Grid sufficient stats plus exact arbitrary prior residual',seconds=sec,kind='Independent implementation; precompute excluded; no prior-uniformity shortcut',max_abs_error=maxerr,changed_states=int((selected!=proposed).sum()),mathematical_change=False))
 csvout(out/'state_loss_breakdown.csv',rows)

def run(rec,out):
 try:probability(rec,out)
 except Exception as exc:save(out/'probability_micro_error.json',dict(error=repr(exc)));raise
 try:components(rec,out)
 except Exception as exc:save(out/'state_micro_error.json',dict(error=repr(exc)));raise
