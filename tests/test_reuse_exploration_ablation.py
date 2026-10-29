import pytest
import torch
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration,regroup_history,reference_attention
from saf3r.models.official_vggt.layers.attention import Attention

@pytest.mark.parametrize('n',[33,64,69,133])
def test_shift_history_overlap_and_duplicate_keys(n):
 from types import SimpleNamespace
 g=(n+31)//32
 ids=torch.tensor([[0,i+1,-1] for i in range(g)],dtype=torch.int32)
 weights=torch.tensor([[.25,.75,0.] for _ in range(g)],dtype=torch.float16)
 s=SimpleNamespace(n=n,groups=g,query_order=torch.arange(n),patches=torch.arange(100),history_ids=ids.clone(),history_weights=weights.clone(),previous_grid=torch.eye(g).half(),previous_real_grid=torch.eye(g).half())
 order=(torch.arange(n)+8)%n
 regroup_history(s,order)
 for group in range(g):
  old=order[group*32:(group+1)*32]//32
  expected=torch.bincount(old,minlength=g).float()/len(old)
  torch.testing.assert_close(s.previous_grid[group].float(),expected,atol=.001,rtol=.001)
  observed={int(k):float(w) for k,w in zip(s.history_ids[group],s.history_weights[group]) if k>=0 and w>0}
  wanted={0:.25}
  for source in range(g):
   if expected[source]>0:wanted[source+1]=float(expected[source])*.75
  assert observed.keys()==wanted.keys()
  for k in wanted:assert abs(observed[k]-wanted[k])<.001

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
@pytest.mark.parametrize('length',[3,35,65])
def test_single_qk_evidence_matches_reference_and_second_qk(length):
 from saf3r.triton.reuse_explore_av import weighted_attention
 torch.manual_seed(22);device='cuda';h,d=2,32;g=(length+31)//32
 q=torch.randn(h,length+7,d,device=device,dtype=torch.bfloat16)[:,3:3+length]
 k=torch.randn(h,100,d,device=device,dtype=torch.bfloat16);v=torch.randn_like(k)
 ids=torch.arange(5,76,device=device,dtype=torch.int32)[None].repeat(g,1);ids[:,-3:]=-1
 pop=torch.rand(g,71,device=device)*9+1;pop[:,-3:]=0
 specials=torch.arange(5,device=device)
 expected,mass=reference_attention(q,k,v,ids,pop.log(),specials,d**-.5)
 second,old_mass=weighted_attention(q,k,v,ids,pop.log(),specials,d**-.5)
 actual,new_mass=weighted_attention(q,k,v,ids,pop.log(),specials,d**-.5,single_qk=True)
 torch.testing.assert_close(actual,second,atol=0,rtol=0)
 torch.testing.assert_close(actual.float(),expected.float(),atol=.015,rtol=.025)
 torch.testing.assert_close(new_mass,mass,atol=.001,rtol=.02)
 torch.testing.assert_close(new_mass,old_mass,atol=1e-6,rtol=1e-5)
 assert (new_mass[ids<0]==0).all()

@pytest.mark.parametrize('variant',['shift','max_error','saved_logits','grid4','grid8'])
@torch.inference_mode()
def test_variant_all_layers_finite_and_budget_cpu(variant):
 torch.set_num_threads(1);torch.manual_seed(5)
 cfg=ReuseExploreConfig(source_width=4,window_groups=2,capture=True,grid_y=4 if variant=='grid4' else (8 if variant=='grid8' else 2),grid_x=4 if variant=='grid4' else (8 if variant=='grid8' else 2),query_shift=8 if variant=='shift' else 0,allocation_objective='max_error' if variant=='max_error' else 'min_error',evidence_mode='saved_logits' if variant=='saved_logits' else 'second_qk')
 s=ReuseExploreState(cfg);s.reset(1,8,16,'cpu')
 for layer in range(24):
  o=Attention(dim=32,num_heads=2).eval();w=ReuseResidualExploration(o,s,layer).eval()
  x=torch.randn(1,133,32)
  with torch.inference_mode():out=w(x)
  assert torch.isfinite(out).all()
  from saf3r.patch.region_history_attention import prepare_qkv
  q,k,v=prepare_qkv(o,x,None)
  if variant=='shift':
   order=s.captured[layer]['query_order'];assert torch.equal(order,(torch.arange(133)+(layer*8)%32)%133)
   permuted=q[:,order];expected=[]
   for wi,window in enumerate(s.captured[layer]['windows']):
    start=wi*cfg.window_groups*32;end=min(133,start+len(window['ids'])*32)
    part,_=reference_attention(permuted[:,start:end],k,v,window['ids'],window['population'].log(),s.specials,o.scale);expected.append(part)
   permuted_out=torch.cat(expected,1);restored=torch.empty_like(permuted_out);restored[:,order]=permuted_out
   expected_out=o.proj_drop(o.proj(restored.transpose(0,1).reshape(1,133,32)))
   torch.testing.assert_close(out,expected_out,atol=1e-6,rtol=1e-5)
  for window in s.captured[layer]['windows']:
   for ids in window['ids']:
    valid=ids[ids>=0];assert len(valid.unique())==len(valid) and len(valid)<=s.budget
  assert (s.previous_grid.sum(1).float()-1).abs().max()<.01


def test_maximum_error_objective_keeps_invalid_states_unavailable_and_budget():
 from saf3r.patch.reuse_residual_exploration import budget_states
 errors=torch.tensor([[[1.,10.,float('inf')],[1.,2.,float('inf')]]])
 costs=torch.tensor([[[0.,1.,0.],[0.,1.,0.]]])
 chosen,count=budget_states(errors,costs,1,24,maximize=True)
 assert chosen.tolist()==[[1,0]] and count.item()==1
 chosen,count=budget_states(errors,costs,1,24)
 assert chosen.tolist()==[[0,0]] and count.item()==0
 # Drop can be the worst-error state; unused budget is intentional.
 chosen,count=budget_states(torch.tensor([[[10.,1.,0.]]]),torch.tensor([[[0.,1.,4.]]]),4,maximize=True)
 assert chosen.tolist()==[[0]] and count.item()==0


@pytest.mark.parametrize('size',[2,4,8])
def test_canonical_grid_resolution_preserves_frames_and_pixel_coverage(size):
 from saf3r.patch.reuse_residual_exploration import canonical_layout
 frames,height,width=3,28,37
 labels,regions=canonical_layout(frames,height,width,size,size,'cpu')
 assert regions==frames*size*size and labels.numel()==frames*height*width
 counts=torch.bincount(labels,minlength=regions)
 assert (counts>0).all() and counts.sum()==frames*height*width
 assert torch.equal(labels.reshape(frames,-1)//(size*size),torch.arange(frames)[:,None].expand(frames,height*width))
 for y in range(height):
  for x in range(width):assert labels[y*width+x]==(y*size//height)*size+x*size//width
