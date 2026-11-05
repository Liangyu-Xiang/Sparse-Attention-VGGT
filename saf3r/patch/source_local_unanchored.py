"""Fixed source candidates, centroid Lloyd, and temporary local empty repair.

Algorithm variant: source membership is immutable during one forward, but every
member (including yesterday's representative) can change fine cluster.
"""
import math
import torch


def _initial_layout(frames, height, width, count, source_width, layout, temporal_span):
    n = frames * height * width
    if not frames <= count <= n or min(source_width, temporal_span) < 1:
        raise ValueError('Invalid source partition')
    # Identical fine initialization for every layout: vary only source membership.
    quotas = torch.full((frames,), count // frames, dtype=torch.long)
    quotas += torch.arange(frames) < count % frames
    offsets = quotas.cumsum(0) - quotas
    labels = (torch.arange(height * width)[None] * quotas[:, None] // (height * width)
              + offsets[:, None]).flatten()
    if layout == 'sequential':
        return labels, labels // source_width
    if layout != 'spatiotemporal':
        raise ValueError(layout)
    ids = torch.arange(n)
    coordinates=torch.stack((ids//(height*width),ids//width%height,ids%width),1).double()
    centers=torch.zeros((count,3),dtype=torch.double).index_add_(0,labels,coordinates)
    centers/=torch.bincount(labels,minlength=count)[:,None]
    target_tiles=max(1,math.ceil(count/frames*min(frames,temporal_span)/source_width))
    tiles_y=min(height,max(1,round(math.sqrt(target_tiles*height/width))))
    tiles_x=min(width,max(1,math.ceil(target_tiles/tiles_y)))
    tile_t=centers[:,0].long()//temporal_span
    tile_y=(centers[:,1]*tiles_y/height).long().clamp_max(tiles_y-1)
    tile_x=(centers[:,2]*tiles_x/width).long().clamp_max(tiles_x-1)
    tile=(tile_t*tiles_y+tile_y)*tiles_x+tile_x
    # Fine IDs already follow frame/raster order; stable sorting preserves this
    # locality within each tile. Oversized tiles split into <=source_width IDs.
    order=tile.argsort(stable=True)
    _,sizes=torch.unique_consecutive(tile[order],return_counts=True)
    cluster_source=torch.empty(count,dtype=torch.long)
    offset=0; group=0
    for size in sizes.tolist():
        cluster_source[order[offset:offset+size]]=torch.arange(size)//source_width+group
        offset+=size; group+=math.ceil(size/source_width)
    return labels,cluster_source[labels]


class SourceLocalUnanchoredValue:
    def __init__(self, frames, height, width, base_count, device, source_width=32,
                 iterations=2, layout='sequential', temporal_span=1, batch_groups=128, fused_scatter=False):
        labels,sources=_initial_layout(frames,height,width,base_count,source_width,layout,temporal_span)
        self.fused_scatter=fused_scatter
        self.base_count=base_count; self.iterations=iterations; self.batch_groups=batch_groups
        self.layout=layout; self.temporal_span=temporal_span; self.merges={}; self.records=[]
        groups=int(sources.max())+1
        counts=torch.bincount(sources,minlength=groups)
        capacity=int(counts.max())
        order=sources.argsort(stable=True)
        starts=counts.cumsum(0)-counts
        table=torch.full((groups,capacity),len(labels),dtype=torch.long)
        table[sources[order],torch.arange(len(labels))-starts[sources[order]]]=order
        cluster_sources=torch.empty(base_count,dtype=torch.long)
        cluster_sources[labels]=sources
        candidate_count=torch.bincount(cluster_sources,minlength=groups)
        candidates=torch.full((groups,int(candidate_count.max())),base_count,dtype=torch.long)
        ordered=cluster_sources.argsort(stable=True)
        starts=candidate_count.cumsum(0)-candidate_count
        candidates[cluster_sources[ordered],torch.arange(base_count)-starts[cluster_sources[ordered]]]=ordered
        self.labels=labels.to(device); self.source=sources.to(device)
        self.table=table.to(device); self.candidates=candidates.to(device)
        self.cluster_source=cluster_sources.to(device)
        self.reps=torch.full((base_count,),len(labels),device=device,dtype=torch.long)
        self.reps.scatter_reduce_(0,self.labels,torch.arange(len(labels),device=device),reduce='amin')

    def _centers(self,x):
        population=torch.bincount(self.labels,minlength=self.base_count)
        centers=torch.zeros((self.base_count,x.shape[1]),device=x.device,dtype=torch.float32)
        centers.index_add_(0,self.labels,x)
        return centers/population.clamp_min(1)[:,None]

    def update(self,x,layer):
        # Remove the common offset to reduce cancellation in squared distances.
        x=x.float(); x=x-x.mean(0); n=len(x); k=self.base_count
        padded=torch.cat((x,torch.zeros_like(x[:1])))
        for _ in range(self.iterations):
            centers=self._centers(x)
            centers=torch.cat((centers,torch.zeros_like(centers[:1])))
            new=self.labels.clone(); costs=torch.empty(n+1,device=x.device)
            if self.fused_scatter:
                from saf3r.triton.source_local_scatter import assignment,repair
                padded_labels=torch.cat((self.labels,self.labels[:1]))
            for begin in range(0,len(self.table),self.batch_groups):
                table=self.table[begin:begin+self.batch_groups]
                candidate=self.candidates[begin:begin+self.batch_groups]
                xx=padded[table]; cc=centers[candidate]
                dist=(xx.square().sum(-1)[:,:,None]+cc.square().sum(-1)[:,None]
                      -2*torch.bmm(xx,cc.transpose(1,2))).clamp_min_(0)
                dist.masked_fill_(candidate[:,None]==k,torch.inf)
                best,choice=dist.min(-1); assigned=candidate.gather(1,choice)
                old=(padded_labels if self.fused_scatter else torch.cat((self.labels,self.labels[:1])))[table]
                own=(xx-centers[old]).square().sum(-1)
                assigned=torch.where(own<=best,old,assigned)
                valid=table<n
                if self.fused_scatter:
                    assignment(table,assigned,torch.minimum(own,best),new,costs,n)
                else:
                    new[table[valid]]=assigned[valid]
                    costs[table[valid]]=torch.minimum(own,best)[valid]
            # Each donor keeps one ordinary member ONLY during this repair.
            population=torch.bincount(new,minlength=k)
            first=torch.full((k+1,),n,device=x.device,dtype=torch.long)
            first.scatter_reduce_(0,new,torch.arange(n,device=x.device),reduce='amin')
            eligible=torch.arange(n,device=x.device)!=first[new]
            score=costs[:n].masked_fill(~eligible,-torch.inf)
            score=torch.cat((score,score.new_full((1,),-torch.inf)))
            empty=torch.cat((population==0,torch.zeros(1,device=x.device,dtype=torch.bool)))
            for begin in range(0,len(self.table),self.batch_groups):
                table=self.table[begin:begin+self.batch_groups]
                candidate=self.candidates[begin:begin+self.batch_groups]
                missing=empty[candidate]
                # Rank empties and high-error donors within the SAME source.
                ranks=missing.long().cumsum(1)-1
                donors=table.gather(1,score[table].argsort(dim=1,descending=True,stable=True))
                target=donors.gather(1,ranks.clamp_min(0))
                if self.fused_scatter:repair(target,candidate,missing,new)
                else:new[target[missing]]=candidate[missing]
            self.labels=new
        centers=self._centers(x)
        distance=(x-centers[self.labels]).square().sum(-1)
        minimum=torch.full((k,),torch.inf,device=x.device)
        minimum.scatter_reduce_(0,self.labels,distance,reduce='amin')
        ids=torch.where(distance==minimum[self.labels],torch.arange(n,device=x.device),n)
        self.reps=torch.full((k,),n,device=x.device,dtype=torch.long)
        self.reps.scatter_reduce_(0,self.labels,ids,reduce='amin')
        self.records.append(dict(layer=layer,mode='source_local_unanchored',layout=self.layout,
                                 temporal_span=self.temporal_span,anchor_constraint=False))

    def representatives(self,x,count):
        if count==self.base_count:return self.reps,self.labels
        from .fused_shared_av import FullCoverageValue
        return FullCoverageValue.representatives(self,x,count)
