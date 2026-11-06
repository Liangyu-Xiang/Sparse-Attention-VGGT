import types,pytest,torch
from saf3r.patch.teacher_guided_history import prepare_keys
from saf3r.patch.head_value_sketch import projection
from saf3r.triton import teacher_history_probability as p
from saf3r.triton.teacher_history_probability_tiles import attention
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('n,probes',[(512,4),(259,4),(356,2),(480,1)])
def test_observe_predict_update(n,probes):
 torch.manual_seed(41);dev='cuda';h,d=2,32;g=(n+255)//256;specials=torch.tensor([0,5],device=dev);patches=torch.tensor([j for j in range(n) if j not in [0,5]],device=dev);np=len(patches);q=torch.randn(h,n,d,device=dev,dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);scale=d**-.5
 s=types.SimpleNamespace(config=types.SimpleNamespace(history_key_block_size=64,teacher_history_samples=8),patches=patches,specials=specials,_value_sketch_matrix=projection(h,d,4,dev))
 prep=prepare_keys(s,k);obs=p.observe(q,prep,0,g,256,probes,scale);pos=obs['positions'];qp=q[:,pos].float();r=prep['regions'];m=8
 raw=torch.einsum('hgpd,hrd->hgpr',qp,prep['bank'].float())*scale+prep['bias'];coarse=raw.softmax(-1).mean((0,2))[:,:r]
 torch.testing.assert_close(obs['coarse'],coarse,rtol=1e-4,atol=2e-6)
 raw=torch.einsum('hgpd,hkd->hgpk',qp,prep['anchor_keys'].float()).reshape(h,g,probes,r,m)*scale;raw.masked_fill_(~prep['valid'].reshape(1,1,1,r,m),-torch.inf);logs=raw.softmax(-1).mean((0,2)).clamp_min(1e-30).log()
 torch.testing.assert_close(obs['sample_logprob'],logs,rtol=2e-5,atol=2e-5)
 oldphi=torch.rand(g,np,device=dev);oldphi=oldphi/oldphi.sum(1,keepdim=True)*.8;oldmass=p.block_mass(oldphi,64);oldcoarse=coarse*.9;oldlog=logs+torch.randn_like(logs)*.2
 old=dict(dense_probability=oldphi,block_mass=oldmass,coarse=oldcoarse,sample_logprob=oldlog);pred,mass=p.predict(old,obs,prep['assignment'],prep['sizes'],64,.75,1.)
 transport=oldmass*coarse/oldcoarse;sp=(1-oldmass.sum(1))*(1-coarse.sum(1))/(1-oldcoarse.sum(1));transport/= (transport.sum(1)+sp)[:,None];beta=2*(coarse*oldcoarse).sqrt()/(coarse+oldcoarse)
 grid=torch.arange(np,device=dev)//64;delta=(logs-oldlog)[:,grid,prep['assignment']];tilted=oldphi*delta.clamp(-20,20).exp();tilt_mass=p.block_mass(tilted,64);ref=.75*transport[:,grid]*(beta[:,grid]*tilted/tilt_mass[:,grid]+(1-beta[:,grid])/prep['sizes'][grid])+.25*coarse[:,grid]/prep['sizes'][grid]
 torch.testing.assert_close(pred,ref,rtol=3e-5,atol=1e-7);torch.testing.assert_close(mass,p.block_mass(pred,64),rtol=1e-5,atol=1e-7)
 # Selected posterior gets unweighted conditional evidence from compensated AV.
 ids=torch.stack([torch.randperm(np,device=dev)[:37] for _ in range(g)]).int();ids[-1,-1]=-1;actual=patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int();weights=torch.exp(torch.randn(g,37,device=dev)*2+3)
 _,e=attention(q,k,v,actual,specials,weights,scale,probes,True);out=torch.empty_like(pred);p.update(pred.clone(),mass,ids,e,out,2,64,1.)
 keys=torch.cat((specials[None].expand(g,-1),actual),1).long();raw=torch.einsum('hgpd,hgad->hgpa',qp,k[:,keys.clamp_min(0)].float())*scale;raw.masked_fill_((keys<0)[None,:,None,:],-torch.inf);conditional=raw.softmax(-1).mean((0,2))[:,2:]
 valid=ids>=0;selected_mass=(1-pred.sum(1)+(pred.gather(1,ids.clamp_min(0).long())*valid).sum(1)).clamp(1e-7,1-1e-7);expected=pred.clone()
 for i in range(g):expected[i,ids[i,valid[i]].long()]=conditional[i,valid[i]]*selected_mass[i]
 expected*=mass[:,grid]/p.block_mass(expected,64)[:,grid]
 torch.testing.assert_close(out,expected,rtol=.006,atol=2e-5);torch.testing.assert_close(p.block_mass(out,64),mass,rtol=1e-5,atol=1e-7)
 # Changing historical fine density changes future prediction,even with same coarse input.
 altered=dict(old,dense_probability=oldphi.roll(1,1));other,_=p.predict(altered,obs,prep['assignment'],prep['sizes'],64,.75,1.)
 assert not torch.allclose(other,pred)
