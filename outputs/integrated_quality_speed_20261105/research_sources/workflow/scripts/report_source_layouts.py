"""Static geometry metadata for the predeclared source layouts (no inference)."""
from pathlib import Path
import importlib.util,json,math
import torch
root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('layout',root/'saf3r/patch/source_local_unanchored.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);torch.set_num_threads(2)
rows=[]
for layout in ('sequential','spatiotemporal'):
 for width in (16,32,64):
  for span in ((1,) if layout=='sequential' else (1,2,4)):
   labels,sources=m._initial_layout(128,28,37,26009,width,layout,span)
   groups=int(sources.max())+1;cs=torch.empty(26009,dtype=torch.long);cs[labels]=sources
   counts=torch.bincount(cs,minlength=groups);t=torch.arange(len(labels))//(28*37)
   lo=torch.full((groups,),128,dtype=torch.long).scatter_reduce_(0,sources,t,reduce='amin')
   hi=torch.zeros(groups,dtype=torch.long).scatter_reduce_(0,sources,t,reduce='amax')
   occupied=torch.zeros((groups,128),dtype=torch.bool);occupied[sources,t]=True
   target=max(1,math.ceil(26009/128*span/width));gy=min(28,max(1,round(math.sqrt(target*28/37))));gx=min(37,max(1,math.ceil(target/gy)))
   rows.append(dict(layout=layout,source_width=width,temporal_span=span,source_groups=groups,clusters_min=int(counts.min()),clusters_mean=float(counts.float().mean()),clusters_max=int(counts.max()),frame_extent_max=int((hi-lo+1).max()),mean_distinct_frames=float(occupied.sum(1).float().mean()),tile_grid=[gy,gx] if layout=='spatiotemporal' else None))
out=root/'outputs/source_local_quality_20261105';(out/'source_layouts.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['Static source layouts, after the same fine initialization. Width bounds candidate clusters, not Query tokens.','', '| Layout | Width | Temporal bin | Spatial tile grid | Source groups | Fine clusters mean/max | Mean distinct frames | Max frame extent |','|---|---|---|---|---|---|---|---|']
for r in rows:lines.append(f"| {r['layout']} | {r['source_width']} | {r['temporal_span']} | {r['tile_grid']} | {r['source_groups']} | {r['clusters_mean']:.2f}/{r['clusters_max']} | {r['mean_distinct_frames']:.2f} | {r['frame_extent_max']} |")
lines+=['','Spatial tile size adjusts with temporal bin to keep the candidate-cluster cap fixed. These are combined layout variants, not a clean ablation of temporal span alone. Temporal span is an upper bound, not the actual number of frames in every group. Source spatial tiles are independent of the unchanged 2x2 per-frame coarse/history grid. Initial fine cells can cross tile boundaries because sources are assigned from fine-cell mean image coordinates.']
(out/'source_layouts.md').write_text('\n'.join(lines)+'\n')
print('Saved source layout geometry')
