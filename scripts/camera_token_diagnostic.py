"""Camera attention/feature tracing only: full matched inputs, no method/GT tuning."""
import argparse,copy,gc,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def similarity(a,b,detail=True):
 import torch
 a=a.float().reshape(-1,a.shape[-1]);b=b.float().reshape_as(a)
 sums=torch.zeros(4,device=a.device,dtype=torch.float64);cs=[];rs=[]
 for lo in range(0,len(a),4096):
  x=a[lo:lo+4096];y=b[lo:lo+4096];xx=x.square().sum(-1);yy=y.square().sum(-1);xy=(x*y).sum(-1);dd=(x-y).square().sum(-1)
  sums+=torch.stack((xx.sum(),yy.sum(),xy.sum(),dd.sum())).double()
  cs.append(xy/(xx*yy).clamp_min(1e-30).sqrt());rs.append((dd/yy.clamp_min(1e-30)).sqrt())
 xx,yy,xy,dd=sums.cpu().tolist();cs=torch.cat(cs);rs=torch.cat(rs)
 result=dict(cosine=xy/max(math.sqrt(xx*yy),1e-30),relative_l2=math.sqrt(dd/max(yy,1e-30)),norm_ratio=math.sqrt(xx/max(yy,1e-30)),mean_token_cosine=float(cs.mean()),token_relative_l2_p90=float(torch.quantile(rs,.9)))
 if detail:result.update(per_token_cosine=cs.cpu().tolist(),per_token_relative_l2=rs.cpu().tolist())
 return result

def probabilities(q,k,scale,kind,frames,per_frame):
 import torch
 with torch.autocast('cuda',enabled=False):
  if kind=='global':return (q.float()@k.float().transpose(-1,-2)*scale).softmax(-1)
  kk=k.reshape(k.shape[0],frames,per_frame,k.shape[-1]).float()
  return (torch.einsum('htd,htpd->htp',q.float(),kk)*scale).softmax(-1)

def apply_probability(p,v,kind,frames,per_frame):
 import torch
 with torch.autocast('cuda',enabled=False):
  if kind=='global':return p@v.float()
  return torch.einsum('htp,htpd->htd',p,v.reshape(v.shape[0],frames,per_frame,v.shape[-1]).float())

def qkv(module,x,pos,kind):
 b,n,c=x.shape
 q,k,v=module.qkv(x).reshape(b,n,3,module.num_heads,module.head_dim).permute(2,0,3,1,4).unbind(0)
 q,k=module.q_norm(q),module.k_norm(k)
 if module.rope is not None:q,k=module.rope(q,pos),module.rope(k,pos)
 if kind=='global':return q[0],k[0],v[0]
 return tuple(t.permute(1,0,2,3).reshape(module.num_heads,-1,module.head_dim) for t in (q,k,v))

def saf_camera_probability(wrapper,q_all,k,v,own,frames,per_frame):
 """Effective Value-source probabilities; broadcast scoring Key ownership separately recorded."""
 import torch
 qc=q_all[:,::per_frame];actual=torch.zeros_like(own);counts=torch.zeros(own.shape[:2],device=k.device);score_owner=[]
 n=k.shape[1];special=(torch.arange(frames,device=k.device)[:,None]*per_frame+torch.arange(5,device=k.device)[None]).flatten()
 for h,cfg in enumerate(wrapper.head_cfg):
  mode=cfg['mode'];owner=list(range(frames));keys=None
  if mode in ['full','dino_topk']:
   step=1 if mode=='full' else cfg.get('dino_sp2sp_stride',1)
   keys=torch.arange(0,n,step,device=k.device)
  elif mode=='all_to_first':keys=torch.arange(per_frame,device=k.device)
  elif mode=='stride':
   step=cfg.get('stride',4);start=h%step if cfg.get('shift',False) else 0;keys=torch.arange(start,n,step,device=k.device)
   if cfg.get('include_as',True):keys=torch.cat((keys,special,torch.arange(per_frame,device=k.device))).unique(sorted=True)
  elif mode=='q_probe_topk':
   qh=q_all[h:h+1][None];kh=k[h:h+1][None]
   ratio=cfg.get('q_sample_ratio',1)
   if ratio<1:qh=qh[:,:,torch.linspace(0,n-1,steps=max(1,int(n*ratio))).long(),:]
   # Match the ORIGINAL selector's autocast arithmetic and topk settings.
   bar=qh.mean(2,keepdim=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):scores=(bar.float()@kh.transpose(-1,-2).float()/math.sqrt(k.shape[-1])).squeeze()
   amount=max(1,n//cfg['stride']) if 'stride' in cfg else min(cfg.get('topk',1024),n)
   keys=scores.topk(amount,sorted=False).indices
   if cfg.get('include_as',False):keys=torch.cat((keys,special,torch.arange(per_frame,device=k.device))).unique(sorted=True)
   else:keys=keys.sort().values
  elif mode=='global_to_frame':
   for f in range(frames):
    ids=torch.arange(f*per_frame,(f+1)*per_frame,device=k.device)
    with torch.autocast('cuda',enabled=False):p=(qc[h,f:f+1].float()@k[h,ids].float().T*wrapper.orig_module.scale).softmax(-1)
    actual[h,f,ids]=p[0];counts[h,f]=per_frame
  elif mode=='broadcast_first':
   with torch.autocast('cuda',enabled=False):p=(qc[h,0:1].float()@k[h,:per_frame].float().T*wrapper.orig_module.scale).softmax(-1)[0]
   for f in range(frames):actual[h,f,f*per_frame:(f+1)*per_frame]=p
   counts[h]=per_frame;owner=[0]*frames
  elif mode=='skip':pass
  else:raise ValueError('Unimplemented SAF3R camera mode: '+mode)
  if keys is not None:
   with torch.autocast('cuda',enabled=False):p=(qc[h].float()@k[h,keys].float().T*wrapper.orig_module.scale).softmax(-1)
   actual[h,:,keys]=p;counts[h]=len(keys)
  score_owner.append(dict(head=h,mode=mode,query_owner_frames=owner,score_key_frame_rule='first-frame Keys with per-target-frame Values' if mode=='broadcast_first' else 'K and V share token IDs',config=cfg))
 return actual,counts,score_owner

def distribution_metrics(p,r):
 import torch
 with torch.autocast('cuda',enabled=False):
  xx=(p*p).sum(-1);yy=(r*r).sum(-1);dot=(p*r).sum(-1);dd=((p-r)**2).sum(-1);m=(p+r)*.5
  js=.5*(torch.where(p>0,p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log()),0.).sum(-1)+torch.where(r>0,r*(r.clamp_min(1e-30).log()-m.clamp_min(1e-30).log()),0.).sum(-1))
  cos=dot/(xx*yy).clamp_min(1e-30).sqrt();rel=(dd/yy.clamp_min(1e-30)).sqrt();tv=.5*(p-r).abs().sum(-1)
 return dict(cosine=cos,relative_l2=rel,js_nats=js,total_variation=tv)

def dist_summary(p,r):
 vals=distribution_metrics(p,r)
 return {name:float(x.mean()) for name,x in vals.items()}, {name:x.cpu().tolist() for name,x in vals.items()}

class Trace:
 def __init__(self,model,method,images,out,reference):
  import torch
  self.torch=torch;self.model=model;self.method=method;self.images=images;self.out=out;self.ref=reference;self.cache={} if reference is None else reference
  self.frames=len(images);self.per=(images.shape[-2]//14)*(images.shape[-1]//14)+5;self.cam=torch.arange(self.frames,device='cuda')*self.per
  self.reg=(self.cam[:,None]+torch.arange(1,5,device='cuda')[None]).flatten();self.patch=torch.arange(self.frames*self.per,device='cuda');self.patch=self.patch[self.patch%self.per>=5]
  self.handles=[];self.context={};self.features=[];self.attention=[];self.local=[];self.counterfactual=[];self.raw={};self.trunk_count={}
 def camera(self,x):return x.reshape(-1,x.shape[-1])[self.cam]
 def register(self,x):return x.reshape(-1,x.shape[-1])[self.reg]
 def feature(self,key,x):
  cam=self.camera(x).detach();reg=self.register(x).detach();self.raw[key]=dict(camera=cam.cpu(),register=reg.cpu())
  if self.method=='dense':self.cache[key]=self.raw[key]
  else:
   ref=self.cache[key];self.features.append(dict(stage=key,camera=similarity(cam,ref['camera'].cuda()),register=similarity(reg,ref['register'].cuda())))
 def full_raw(self,key,kind,layer,x):
  if self.method=='dense':
   if kind=='global' or (key=='input' and layer==0):self.cache[f'{kind}_{layer:02d}_{key}_raw']=x.detach().cpu()
  else:
   if kind=='global':r=self.cache[f'global_{layer:02d}_{key}_raw'].cuda().reshape_as(x)
   elif key=='input':r=self.cache['frame_00_input_raw' if layer==0 else f'global_{layer-1:02d}_output_raw'].cuda().reshape_as(x)
   else:r=self.cache[f'global_{layer:02d}_input_raw'].cuda().reshape_as(x)
   if key=='input':self.context[(kind,layer)]['ref_raw']=r
   self.features.append(dict(stage=f'{kind}_{layer:02d}_block_{key}_all_tokens',camera=similarity(self.camera(x),self.camera(r)),register=similarity(self.register(x),self.register(r)),patch=similarity(x.reshape(-1,x.shape[-1])[self.patch],r.reshape(-1,r.shape[-1])[self.patch],detail=False)))
 def install(self):
  for kind,blocks in [('frame',self.model.aggregator.frame_blocks),('global',self.model.aggregator.global_blocks)]:
   for layer,block in enumerate(blocks):
    self.handles.append(block.register_forward_pre_hook(lambda mod,args,kind=kind,layer=layer:self.block_pre(kind,layer,args[0])))
    self.handles.append(block.norm2.register_forward_pre_hook(lambda mod,args,kind=kind,layer=layer:self.feature(f'{kind}_{layer:02d}_post_attention_residual',args[0])))
    self.handles.append(block.register_forward_hook(lambda mod,args,out,kind=kind,layer=layer:self.block_post(kind,layer,out)))
    attn=block.attn;original=getattr(attn,'original',getattr(attn,'orig_module',attn))
    self.handles.append(attn.register_forward_pre_hook(lambda mod,args,kwargs,kind=kind,layer=layer,block=block,original=original:self.attn_pre(kind,layer,block,mod,original,args[0],kwargs.get('pos')),with_kwargs=True))
    self.handles.append(original.proj.register_forward_pre_hook(lambda mod,args,kind=kind,layer=layer,original=original:self.pre_proj(kind,layer,original,args[0])))
    self.handles.append(attn.register_forward_hook(lambda mod,args,out,kind=kind,layer=layer:self.feature(f'{kind}_{layer:02d}_attention_post_Wo',out)))
  self.handles.append(self.model.camera_head.register_forward_pre_hook(self.head_pre))
  self.handles.append(self.model.camera_head.token_norm.register_forward_hook(lambda m,a,o:self.head_feature('camera_head_normalized_input',o)))
  for i,b in enumerate(self.model.camera_head.trunk):
   self.handles.append(b.register_forward_hook(lambda m,a,o,i=i:self.trunk(i,o)))
 def block_pre(self,kind,layer,x):
  self.context[(kind,layer)]={};self.feature(f'{kind}_{layer:02d}_block_input',x);self.full_raw('input',kind,layer,x)
 def block_post(self,kind,layer,x):
  self.feature(f'{kind}_{layer:02d}_block_output',x);self.full_raw('output',kind,layer,x);self.context.pop((kind,layer),None)
 def attn_pre(self,kind,layer,block,wrapper,original,x,pos):
  torch=self.torch;ctx=self.context[(kind,layer)];self.feature(f'{kind}_{layer:02d}_attention_input_norm',x)
  q,k,v=qkv(original,x,pos,kind);qc=q[:,self.cam];own=probabilities(qc,k,original.scale,kind,self.frames,self.per)
  counts=torch.full(own.shape[:2],own.shape[-1],device=x.device);owners=None
  if self.method=='saf3r' and kind=='global':actual,counts,owners=saf_camera_probability(wrapper,q,k,v,own,self.frames,self.per)
  else:actual=own
  actual_o=apply_probability(actual,v,kind,self.frames,self.per);own_o=apply_probability(own,v,kind,self.frames,self.per)
  ctx.update(actual_o=actual_o,own_o=own_o)
  entropy=-(actual*actual.clamp_min(1e-30).log()).sum(-1)
  if kind=='global':
   mass=actual.reshape(actual.shape[0],self.frames,self.frames,self.per).sum(-1)
   special=actual[:,:,self.cam].sum(-1);register=actual[:,:,self.reg].sum(-1)
  else:
   mass=torch.diag_embed(torch.ones(actual.shape[:2],device=x.device));special=actual[:,:,0];register=actual[:,:,1:5].sum(-1)
  score_mass=mass.clone()
  if owners:
   for entry in owners:
    if entry['mode']=='broadcast_first':score_mass[entry['head']].zero_();score_mass[entry['head'],:,0]=1.
  small=dict(frame_score_key_mass=score_mass.cpu(),frame_value_mass=mass.cpu(),camera_key_mass=special.cpu(),register_key_mass=register.cpu(),patch_key_mass=(actual.sum(-1)-special-register).cpu(),entropy_nats=entropy.cpu(),support_counts=counts.cpu(),head_modes=owners)
  self.raw[f'{kind}_{layer:02d}_attention_summary']=small
  if self.method=='dense':
   self.cache[f'{kind}_{layer:02d}_prob']=own.cpu()
   if kind=='global':self.cache[f'global_{layer:02d}_kv']=(qc.detach().cpu(),k.detach().cpu(),v.detach().cpu())
  else:
   refp=self.cache[f'{kind}_{layer:02d}_prob'].cuda();summary,detail=dist_summary(actual,refp);local,_=dist_summary(actual,own)
   self.attention.append(dict(kind=kind,layer=layer,against_dense=summary,per_head_camera=detail,local_same_qkv=local,head_modes=owners))
   if kind=='global':qr,kr,vr=(a.cuda() for a in self.cache[f'global_{layer:02d}_kv'])
   else:
    rx=block.norm1(ctx['ref_raw']);rq,kr,vr=qkv(original,rx,pos,kind);qr=rq[:,self.cam]
   self.features.append(dict(stage=f'{kind}_{layer:02d}_qkv_drift',camera_query=similarity(qc.permute(1,0,2).reshape(self.frames,-1),qr.permute(1,0,2).reshape(self.frames,-1)),camera_key=similarity(k[:,self.cam].permute(1,0,2).reshape(self.frames,-1),kr[:,self.cam].permute(1,0,2).reshape(self.frames,-1)),camera_value=similarity(v[:,self.cam].permute(1,0,2).reshape(self.frames,-1),vr[:,self.cam].permute(1,0,2).reshape(self.frames,-1)),patch_key=similarity(k[:,self.patch].permute(1,0,2).reshape(len(self.patch),-1),kr[:,self.patch].permute(1,0,2).reshape(len(self.patch),-1),detail=False),patch_value=similarity(v[:,self.patch].permute(1,0,2).reshape(len(self.patch),-1),vr[:,self.patch].permute(1,0,2).reshape(len(self.patch),-1),detail=False)))
   dense_o=apply_probability(refp,vr,kind,self.frames,self.per)
   def flat(a):return a.permute(1,0,2).reshape(self.frames,-1)
   def projected(a):
    import torch.nn.functional as F
    with torch.autocast('cuda',enabled=False):return F.linear(flat(a).float(),original.proj.weight.float(),None if original.proj.bias is None else original.proj.bias.float())
   cf={'own_dense_all_qkv_drift':own_o,'current_Q_only':apply_probability(probabilities(qc,kr,original.scale,kind,self.frames,self.per),vr,kind,self.frames,self.per),'current_K_only':apply_probability(probabilities(qr,k,original.scale,kind,self.frames,self.per),vr,kind,self.frames,self.per),'current_V_only':apply_probability(refp,v,kind,self.frames,self.per),'current_QK_dense_V':apply_probability(own,vr,kind,self.frames,self.per),'actual_operator':actual_o}
   self.counterfactual.append(dict(kind=kind,layer=layer,against_dense={name:similarity(projected(value),projected(dense_o)) for name,value in cf.items()},note='Counterfactuals use same trained projections/token alignment; effects are nonlinear and must not be summed as percentages of causation'))
   del refp,kr,vr,qr,cf
  del q,k,v,own,actual
 def pre_proj(self,kind,layer,original,x):
  ctx=self.context[(kind,layer)];cam=self.camera(x);pred=ctx['actual_o'].permute(1,0,2).reshape(self.frames,-1);own=ctx['own_o'].permute(1,0,2).reshape(self.frames,-1)
  self.local.append(dict(kind=kind,layer=layer,reconstruction=similarity(cam,pred),actual_vs_dense_same_qkv=similarity(cam,own)))
  # This verifies reconstruction of actual SAF3R support/query-sharing, not a guessed dense map.
  if float((cam.float()-pred).norm()/pred.norm().clamp_min(1e-30))>.03:raise RuntimeError(f'Camera attention reconstruction mismatch: {self.method} {kind} {layer}')
  self.feature(f'{kind}_{layer:02d}_attention_pre_Wo',x)
 def head_feature(self,key,x):
  val=x.detach().reshape(self.frames,-1);self.raw[key]=val.cpu()
  if self.method=='dense':self.cache[key]=val.cpu()
  else:self.features.append(dict(stage=key,camera=similarity(val,self.cache[key].cuda())))
 def head_pre(self,module,args):self.head_feature('camera_head_input',args[0][-1][:,:,0])
 def trunk(self,i,out):
  iteration=self.trunk_count.get(i,0);self.trunk_count[i]=iteration+1;self.head_feature(f'camera_head_iteration{iteration}_block{i}',out)
 def close(self):
  for h in self.handles:h.remove()

def run_scene(model,cfg,scene,root,checkpoint_sha=None):
 import numpy as np,torch
 from addict import Dict
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from saf3r.patch import patch_model,update_args_per_forward
 from scripts.quality_metrics import quality_record,quality_markdown
 inp=json.loads((ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105/quality/A_all_specials_cluster20/inputs'/f'{scene}.json').read_text());checkpoint_sha=checkpoint_sha or sha(Path(cfg.ckpt_path));assert checkpoint_sha==inp['checkpoint_sha256'];images=load_images(model,'vggt',inp['image_paths']).cuda();assert list(images.shape)==inp['shape']
 originals=[b.attn for b in model.aggregator.global_blocks];original_forward=model.aggregator.forward;reference=None
 method_config=json.loads((ROOT/'configs/evaluation/reuse_rep_history_A_all_specials_cluster20.json').read_text())
 for method in ['dense','ours','saf3r']:
  out=root/scene/method;out.mkdir(parents=True,exist_ok=False)
  q=quality_record({},'Explicit instrumented camera-token attention/features diagnostic; pose/recon_unposed not evaluated. No production speed measurement; historical quality remains separate.')
  q['speedup_unavailable']='Instrumented attention re-computation and CPU reference copies; not a warmed production forward benchmark'
  save(out/'quality.json',q);(out/'quality.md').write_text(quality_markdown(q));save(out/'status.json',dict(stage='loading'))
  torch.manual_seed(0);np.random.seed(0);state=None
  if method=='ours':state=install_reuse_residual_exploration(model,ReuseExploreConfig(**method_config))
  if method=='saf3r':patch_model(model,'vggt',copy.deepcopy(cfg));update_args_per_forward(model,'vggt',*images.shape[-2:])
  trace=Trace(model,method,images,out,reference);trace.install();save(out/'input.json',dict(inp,checkpoint_sha256=checkpoint_sha,method=method,method_config=method_config if method=='ours' else None,saf3r_profile=str(cfg.sparse_config_path) if method=='saf3r' else None,saf3r_topk=4 if method=='saf3r' else None,gpu= torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid)))
  save(out/'status.json',dict(stage='instrumented_forward'));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
  try:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
   torch.cuda.synchronize();elapsed=time.perf_counter()-begin
   assert len(trace.local)==48 and all(torch.isfinite(pred[k]).all() for k in ['pose_enc','depth'])
   if method=='dense':reference=trace.cache
   save(out/'result.json',dict(method=method,scene=scene,complete=True,quality=q,features=trace.features,attention=trace.attention,local_operator_checks=trace.local,counterfactuals=trace.counterfactual,instrumented_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,scope='All48 Frame/Global layers,all16 heads,all camera tokens,all actual retained Key positions; Dense references held only in CPU memory; no GT tuning or production method changes'))
   torch.save(dict(camera_features_and_attention_summaries=trace.raw,pose_enc=pred['pose_enc'].detach().cpu()),out/'camera_tensors.pt')
   save(out/'status.json',dict(stage='diagnostic_complete'));print('DONE',scene,method,elapsed,flush=True)
  finally:trace.close()
  if state is not None:state.reset_handle.remove();del model.reuse_explore_state
  for block,original in zip(model.aggregator.global_blocks,originals):block.attn=original
  model.aggregator.forward=original_forward
  del trace,pred;gc.collect();torch.cuda.empty_cache()
 del reference,images;gc.collect();torch.cuda.empty_cache()

def main():
 import torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
 dense_cfg=copy.deepcopy(cfg);dense_cfg.pop('patch_module');dense_cfg.pop('sparse_config_path');model=build_model(dense_cfg,'cuda')
 run_scene(model,cfg,a.scene,a.output)
if __name__=='__main__':main()
