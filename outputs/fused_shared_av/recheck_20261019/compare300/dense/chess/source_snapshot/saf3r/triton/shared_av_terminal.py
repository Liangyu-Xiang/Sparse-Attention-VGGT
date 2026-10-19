"""Four real queries/head, local cell softmax, choose one REAL Key/cell."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pick(Q,K,FINE,MEMBERS,MISSING,OUT, R:tl.constexpr,N:tl.constexpr,D:tl.constexpr,
          H:tl.constexpr,WIDTH:tl.constexpr,E:tl.constexpr,QH:tl.constexpr,QS:tl.constexpr,
          KH:tl.constexpr,SCALE:tl.constexpr,BQ:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,BHP:tl.constexpr):
    group,pos=tl.program_id(0),tl.program_id(1)
    if pos<tl.load(MISSING+group):
        cell=tl.load(FINE+group*E+pos)
        member=tl.arange(0,BM);ids=tl.load(MEMBERS+cell*WIDTH+member,member<WIDTH,-1)
        flat=tl.arange(0,BHP*BM);dd=tl.arange(0,BD)
        hp=flat//BM;head=hp//4;probe=hp%4
        member_ids=tl.load(MEMBERS+cell*WIDTH+flat%BM,flat%BM<WIDTH,-1)
        row=tl.minimum(group*BQ+(2*probe+1)*(BQ//8),R-1)
        qq=tl.load(Q+head[:,None]*QH+row[:,None]*QS+dd[None,:],
                   (head[:,None]<H)&(dd[None,:]<D),0).to(tl.float32)
        kk=tl.load(K+head[:,None]*KH+tl.maximum(member_ids,0)[:,None]*D+dd[None,:],
                   (head[:,None]<H)&(member_ids[:,None]>=0)&(dd[None,:]<D),0).to(tl.float32)
        z=tl.reshape(tl.sum(qq*kk,1)*SCALE,(BHP,BM))
        z=tl.where(ids[None,:]>=0,z,-float('inf'))
        weights=tl.exp(z-tl.max(z,1)[:,None]);weights=weights/tl.sum(weights,1)[:,None]
        scores=tl.sum(tl.where(tl.arange(0,BHP)[:,None]<H*4,weights,0.),0)/(H*4)
        best=tl.argmax(scores,0)
        chosen=tl.sum(tl.where(member==best,ids,0),0)
        tl.store(OUT+group*E+pos,chosen)
    else:tl.store(OUT+group*E+pos,-1)


def terminal_keys(q,k,selected,members,missing,scale,query_block=32):
    if selected.shape[1]==0:return selected.int()
    out=torch.empty_like(selected,dtype=torch.int32)
    _pick[(len(selected),selected.shape[1])](q,k,selected,members,missing,out,
        q.shape[1],k.shape[1],q.shape[2],q.shape[0],members.shape[1],selected.shape[1],
        q.stride(0),q.stride(1),k.stride(0),scale,query_block,
        triton.next_power_of_2(q.shape[2]),triton.next_power_of_2(members.shape[1]),triton.next_power_of_2(q.shape[0]*4),num_warps=8)
    return out


@triton.jit
def _pick_tiles(Q,K,KEYMAP,FINE,MEMBERS,MISSING,OUT,R:tl.constexpr,N:tl.constexpr,D:tl.constexpr,
                H:tl.constexpr,WIDTH:tl.constexpr,E:tl.constexpr,QH:tl.constexpr,QS:tl.constexpr,
                KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,FC:tl.constexpr):
    group,tile=tl.program_id(0),tl.program_id(1)
    first=tile*FC
    if first<tl.load(MISSING+group):
        col=tl.arange(0,FC*BM);pos=first+col//BM
        valid_cell=(pos<E)&(pos<tl.load(MISSING+group))
        cell=tl.load(FINE+group*E+pos,valid_cell,0)
        local=tl.load(MEMBERS+cell*WIDTH+col%BM,valid_cell&(col%BM<WIDTH),-1)
        ids=tl.load(KEYMAP+tl.maximum(local,0),local>=0,-1)
        dd=tl.arange(0,BD);pr=tl.arange(0,16)
        qr=tl.minimum(group*32+(2*pr+1)*4,R-1)
        scores=tl.zeros((FC*BM,),tl.float32)
        for head in range(H):
            qq=tl.load(Q+head*QH+qr[:,None]*QS+dd[None,:],(pr[:,None]<4)&(dd[None,:]<D),0)
            kk=tl.load(K+head*KH+dd[:,None]+tl.maximum(ids,0)[None,:]*D,(dd[:,None]<D)&(ids[None,:]>=0),0)
            z=tl.dot(qq,kk)*SCALE
            z=tl.where(ids[None,:]>=0,z,-float('inf'))
            cells=tl.reshape(z,(16*FC,BM))
            maximum=tl.max(cells,1);maximum=tl.where(maximum==-float('inf'),0.,maximum)
            weights=tl.exp(cells-maximum[:,None]);weights=weights/tl.maximum(tl.sum(weights,1)[:,None],1e-30)
            weights=tl.reshape(weights,(16,FC*BM))
            scores+=tl.sum(tl.where(pr[:,None]<4,weights,0.),0)
        by_cell=tl.reshape(scores,(FC,BM));best=tl.argmax(by_cell,1)
        local=tl.reshape(local,(FC,BM));member=tl.arange(0,BM)
        chosen=tl.sum(tl.where(member[None,:]==best[:,None],local,0),1)
        positions=first+tl.arange(0,FC)
        tl.store(OUT+group*E+positions,chosen,(positions<E)&(positions<tl.load(MISSING+group)))


def terminal_keys_tensorcore(q,k,patch_keys,selected,members,missing,scale):
    if selected.shape[1]==0:return selected.int()
    out=torch.full_like(selected,-1,dtype=torch.int32)
    _pick_tiles[(len(selected),triton.cdiv(selected.shape[1],8))](q,k,patch_keys,selected,members,missing,out,
        q.shape[1],k.shape[1],q.shape[-1],q.shape[0],members.shape[1],selected.shape[1],q.stride(0),q.stride(1),k.stride(0),
        scale,max(16,triton.next_power_of_2(q.shape[-1])),triton.next_power_of_2(members.shape[1]),8,num_warps=4,num_stages=2)
    return out
