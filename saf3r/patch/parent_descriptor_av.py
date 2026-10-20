"""Direct whole-Parent execution; no per-Query-window Parent member expansion."""
import math
import torch
from .fused_shared_av import pack_disjoint_history,shared_scores,stage
from .value_region_av import aligned_history
from .parent_region_selection import parent_prefix


def descriptor_layer(wrapper,q,k,v,pk,pv,common,value_ids,merged,is_value):
    from saf3r.triton.parent_descriptor_attention import remaining_history,descriptor_attention,descriptor_history
    s=wrapper.state;c=s.config;layer=wrapper.layer;o=wrapper.original
    if not q.is_cuda:raise ValueError('Direct Parent descriptor execution requires CUDA; use expanded CPU reference for tests')
    device=q.device;ka=s.ka[layer];kv=len(value_ids);groups=math.ceil(s.n/32)
    next_history=min(s.total[layer],math.floor(c.history_fraction*s.ka[layer+1])) if layer<23 else 0
    output=torch.empty_like(v);histories=[];all_counts=[];raw_counts=[];hc_counts=[];parent_counts=[];stats=[];captures=[]
    if ka:
        value_parent=torch.zeros(s.parent_count,device=device,dtype=torch.long)
        value_parent.index_add_(0,s.patch_parent[value_ids],torch.ones(kv,device=device,dtype=torch.long))
        starts=s.parent_starts;member=s.member_bank;parent_sizes=s.parent_sizes
    else:
        starts=torch.zeros(s.parent_count,device=device,dtype=torch.long)
        member=torch.zeros(1,device=device,dtype=torch.int32)
        parent_sizes=torch.zeros(s.parent_count,device=device,dtype=torch.long)
    # Conservative raw-member bound includes duplicates later skipped in-kernel.
    raw_capacity=s.total[layer]+s.capacity-1 if ka else 0
    # No device scalar synchronization: a conservative bound on the number of
    # selected Parents follows from source geometry and merged-Value coverage.
    fine_min=(s.height*s.width)//math.ceil(s.value.base_count/s.frames)
    parent_min=min(c.source_width,s.value.base_count%c.source_width or c.source_width)*fine_min
    value_max=min(c.source_width,math.ceil(c.source_width*kv/s.value.base_count)+1) if kv else 0
    nonvalue_min=max(1,parent_min-value_max)
    parent_limit=min(s.parent_count,math.ceil(ka/nonvalue_min)+1) if ka else 0
    for g in range(0,groups,c.window_groups):
        length=min(c.window_groups,groups-g);begin=g*32;end=min(s.n,(g+length)*32)
        with stage(s,'query_history_realign'):old=aligned_history(s,g,length)
        clean,kept=pack_disjoint_history(old,is_value)
        if clean is None:
            clean=torch.empty((length,0),device=device,dtype=torch.int32)
            kept=torch.zeros(length,device=device,dtype=torch.long)
        if ka:
            with stage(s,'parent_available'):
                available=(s.parent_sizes-value_parent)[None].expand(length,-1).clone()
                if clean.shape[1]:
                    valid=clean>=0;parents=s.patch_parent[clean.long().clamp_min(0)]
                    available.scatter_add_(1,parents,-valid.long())
            with stage(s,'region_parent'):
                scores=shared_scores(q,s.pooled_parent,o.scale,s.parent_log_population,g,length,query_offset=s.query_offset)
                order,chosen,extra=parent_prefix(scores,available,ka-kept,c.parent_topk)
            ids=order.masked_fill(~chosen,-1).int().contiguous()
            sizes=s.parent_sizes[order.clamp_min(0)]*chosen
            ends=sizes.cumsum(1).contiguous()
            counts=kept+(available.gather(1,order.clamp_min(0))*chosen).sum(1)
            stats.append(dict(incoming_history=kept.sum(),parent_regions=chosen.sum(),**extra))
        else:
            ids=torch.full((length,s.parent_count),-1,device=device,dtype=torch.int32)
            ends=torch.zeros((length,s.parent_count),device=device,dtype=torch.long)
            counts=torch.zeros(length,device=device,dtype=torch.long)
        with stage(s,'parent_history_filter'):
            h,hcounts=remaining_history(clean,ids,s.patch_parent)
        with stage(s,'execution'):
            result,e,lse,tile_counts=descriptor_attention(q,k,v,pk,pv,h,hcounts,ids,ends,starts,member,s.patches,
                is_value,kv,raw_capacity,query_start=begin,query_rows=end-begin,query_offset=s.query_offset,
                scale=o.scale,evidence=bool(next_history),parent_sizes=parent_sizes,
                parent_stride=s.capacity,parent_limit=parent_limit)
        dest=(begin+s.query_offset)%s.n;first=min(end-begin,s.n-dest)
        output[:,dest:dest+first]=result[:,:first]
        if first<end-begin:output[:,:end-begin-first]=result[:,first:]
        if next_history:
            with stage(s,'history'):
                history=descriptor_history(e,lse,tile_counts,common,h,hcounts,ids,ends,starts,member,is_value,
                    kv,next_history,s.height*s.width,parent_sizes=parent_sizes,parent_stride=s.capacity)
            histories.append(history)
        all_counts.append(counts);raw_counts.append(ends[:,-1]);hc_counts.append(hcounts);parent_counts.append((ids>=0).sum(1))
        if c.capture:
            # Offline small-input validation only; never on the primary path.
            rows=[]
            for row in range(length):
                selected=[h[row,:int(hcounts[row])].long()]
                for parent in ids[row].tolist():
                    if parent<0:continue
                    a=int(starts[parent]);n=int(s.parent_sizes[parent]);local=member[a:a+n].long()
                    selected.append(local[~is_value[local]])
                rows.append(s.patches[torch.cat(selected)].int())
            captures.extend(rows)
    s.history=torch.cat(histories) if next_history else None;s.history_offset=s.query_offset;s.previous_layer=layer
    counts=torch.cat(all_counts);raw=torch.cat(raw_counts);hcounts=torch.cat(hc_counts);pn=torch.cat(parent_counts)
    rows=(s.n-torch.arange(groups,device=device)*32).clamp_max(32)
    unique=((counts+len(common))*rows).sum()
    padded=32*64*(groups*math.ceil(len(common)/64)+pn.sum()*math.ceil(s.capacity/64)+((hcounts+63)//64).sum())
    record=dict(layer=layer,attention_edges_target=ka,value_edges=kv,special_keys=len(s.specials),
        total_edges_target=ka+len(common),actual_attention_min=counts.min(),actual_attention_max=counts.max(),
        overshoot_max=(counts-ka).max(),overshoot_mean=(counts-ka).float().mean(),actual_density=unique/s.n**2,
        final_unique_qk_per_head=unique,final_padded_qk_per_head=padded,attention_ratio=c.attention_ratios[layer],
        parent_only=True,parent_topk=c.parent_topk,parent_descriptors=True,history_out=next_history,
        query_offset=s.query_offset,query_shift_step=c.query_shift_step,base_clusters=s.value.base_count,
        query_groups=groups,terminal_member_qk_per_head=0,source='direct_Parent_descriptors',
        expanded_parent_token_indices=0,parent_seen_bitmap_bytes=0,remaining_history_keys=hcounts.sum(),
        descriptor_layout='per_parent_tiles',parent_descriptor_limit=parent_limit,selected_parent_max=pn.max(),
        child_regions=0,fine_regions=0,child_refined_groups=0,fine_refined_groups=0,
        routing_qk_per_head=4*groups*s.parent_count if ka else 0)
    if ka:
        record.update(max_parent_population=s.parent_sizes.max(),max_fine_population=s.fine_sizes.max())
        for name in stats[0]:record[name]=sum(r[name] for r in stats)
    s.records.append(record)
    if c.capture:
        width=max((len(row) for row in captures),default=0)
        attention=torch.full((groups,width),-1,device=device,dtype=torch.int32)
        for row,ids in enumerate(captures):attention[row,:len(ids)]=ids
        s.captured[layer]=dict(common=common.clone(),attention=attention,base_labels=s.value.labels.clone(),
            value_labels=None if merged is None else merged.clone(),value_representatives=value_ids.clone(),
            query_offset=s.query_offset,regions=[],member_bank=None)
    if ka:
        s.pooled_parent=None;s.pooled_child=None;s.fine_keys=None;s.member_bank=None
    return output
