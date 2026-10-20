import math
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
@pytest.mark.parametrize('offset',[0,23])
@pytest.mark.parametrize('regions',[False,True])
def test_direct_descriptors_unified_softmax_value_history_dedup_and_history(offset,regions):
    from saf3r.triton.parent_descriptor_attention import remaining_history,descriptor_attention,descriptor_history
    torch.manual_seed(18);device='cuda';h,n,d,r=2,97,32,45;pp=92
    q=torch.randn(h,n,d,device=device,dtype=torch.bfloat16)
    k=torch.randn_like(q);v=torch.randn_like(q)
    patches=torch.arange(5,n,device=device);member=torch.arange(pp,device=device,dtype=torch.int32)
    starts=torch.tensor([0,25,50,75],device=device);sizes=torch.tensor([25,25,25,17],device=device)
    source=torch.repeat_interleave(torch.arange(4,device=device),sizes)
    value_ids=torch.tensor([0,26,54,76],device=device);is_value=torch.zeros(pp,device=device,dtype=torch.bool);is_value[value_ids]=True
    common=torch.cat((patches[value_ids],torch.arange(5,device=device)))
    clean=torch.tensor([[-1,2,30,60,80,-1],[3,28,51,80,84,-1]],device=device,dtype=torch.int32)
    ids=torch.tensor([[0,1,2,-1],[1,2,3,-1]],device=device,dtype=torch.int32)
    if not regions:ids.fill_(-1);clean=clean[:,:0]
    ends=(sizes[ids.long().clamp_min(0)]*(ids>=0)).cumsum(1).contiguous()
    hist,hc=remaining_history(clean,ids,source)
    if regions:assert hist[0,:hc[0]].tolist()==[80] and hist[1,:hc[1]].tolist()==[3]
    rawcap=pp if regions else 0
    out,e,lse,counts=descriptor_attention(q,k,v,k[:,common].contiguous(),v[:,common].contiguous(),hist,hc,
        ids,ends,starts,member,patches,is_value,4,rawcap,query_start=40,query_rows=r,query_offset=offset,evidence=True,
        parent_sizes=sizes,parent_stride=64,parent_limit=4 if regions else 0)
    expected=[];tile_candidates=[]
    for group in range(2):
        parent_tiles=[]
        for parent in ids[group].tolist():
            if parent>=0:
                raw=member[starts[parent]:starts[parent]+sizes[parent]].tolist()
                parent_tiles.append([i for i in raw if not is_value[i]])
            elif regions:parent_tiles.append([])
        ht=[hist[group,:int(hc[group])].long().tolist()] if hist.shape[1] else []
        tile_candidates.append([value_ids.tolist()]+ht+parent_tiles)
    for row in range(r):
        group=row//32
        patch_ids=[i for tile in tile_candidates[group][1:] for i in tile]
        support=torch.cat((common,patches[torch.tensor(patch_ids,device=device,dtype=torch.long)]))
        assert len(support.unique())==len(support)
        real=(40+row+offset)%n
        z=(q[:,real,None].float()*k[:,support].float()).sum(-1)/math.sqrt(d)
        p=z.softmax(-1)
        expected.append((p[:,:,None]*v[:,support].float()).sum(1))
        for tile,local in enumerate(tile_candidates[group]):
            if local:
                key_ids=patches[torch.tensor(local,device=device)]
                positions=(support[:,None]==key_ids[None]).any(1)
                mass=p[:,positions].sum(1)
            else:mass=torch.zeros(h,device=device)
            torch.testing.assert_close((e[:,row,tile]-lse[:,row]).exp(),mass,atol=.001,rtol=.015)
    expected=torch.stack(expected,1)
    assert (out.float()-expected).norm()/expected.norm()<.012
    for group,tiles in enumerate(tile_candidates):assert counts[group].tolist()==[len(t) for t in tiles]
    hn=65 if regions else 4
    history=descriptor_history(e,lse,counts,common,hist,hc,ids,ends,starts,member,is_value,4,hn,pp,
        parent_sizes=sizes,parent_stride=64)
    for group in range(2):
        allowed={i for tile in tile_candidates[group] for i in tile}
        assert len(history[group].unique())==hn and set(history[group].tolist()).issubset(allowed)
        mass=(e[:,group*32:min(r,(group+1)*32)]-lse[:,group*32:min(r,(group+1)*32),None]).exp().sum((0,1))
        mass=mass.masked_fill(counts[group]==0,-1)
        order=mass.argsort(descending=True,stable=True).tolist()
        expected_history=[key for tile in order for key in tile_candidates[group][tile]][:hn]
        assert history[group].tolist()==expected_history


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
@pytest.mark.parametrize('mode',['mixed','value','attention'])
def test_descriptor_all24_shifted_outputs_budget_history_and_reset(mode):
    from saf3r.patch.value_region_av import ValueRegionAVConfig,ValueRegionAVState,ValueRegionAVAttention
    from saf3r.patch.region_history_attention import prepare_qkv
    from saf3r.models.official_vggt.layers.attention import Attention
    torch.manual_seed(4)
    ratios=(0.,)*24 if mode=='value' else (0.,)+(1.,)*23 if mode=='attention' else ValueRegionAVConfig().attention_ratios
    config=ValueRegionAVConfig(attention_ratios=ratios,parent_only=True,parent_topk=4,parent_descriptors=True,query_shift_step=1,
        capture=True,source_width=4,region_child_width=2,window_groups=2)
    s=ValueRegionAVState(config);s.reset(1,8,16,'cuda')
    for layer in range(24):
        original=Attention(dim=32,num_heads=2).cuda().eval();x=torch.randn(1,133,32,device='cuda')
        incoming=None if s.history is None else s.history.clone()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            actual=ValueRegionAVAttention(original,s,layer).eval()(x)
            data=s.captured[layer];mask=torch.zeros(133,133,device='cuda',dtype=torch.bool)
            for group,row in enumerate(data['attention']):
                support=torch.cat((data['common'],row[row>=0].long()))
                assert len(support.unique())==len(support) and len(support)>=math.floor(.2*133)
                if s.ka[layer]:assert len(support)-math.floor(.2*133)<int(s.parent_sizes.max())
                qr=(torch.arange(group*32,min(133,(group+1)*32),device='cuda')+layer)%133
                mask[qr[:,None],support[None]]=True
                if incoming is not None:
                    previous=((qr-layer+1)%133//32).unique()
                    old=incoming[previous].flatten();old=old[old>=0].long()
                    assert set(s.patches[old].tolist()).issubset(support.tolist())
            q,k,v=prepare_qkv(original,x,None);q=q.to(v.dtype);k=k.to(v.dtype)
            with torch.autocast('cuda',enabled=False):
                z=(q.float()@k.float().transpose(-1,-2))*original.scale
                expected=(z.masked_fill(~mask[None],-torch.inf).softmax(-1)@v.float()).to(v.dtype)
            expected=original.proj_drop(original.proj(expected.transpose(0,1).reshape(1,133,32)))
        torch.testing.assert_close(actual.float(),expected.float(),rtol=.03,atol=.012)
        assert s.records[-1]['expanded_parent_token_indices']==0
    s.reset(1,8,16,'cuda');assert s.history is None and s.previous_layer==-1
