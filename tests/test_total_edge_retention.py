import math,types,pytest
from saf3r.patch.layer_retention_schedule import total_edge_patch_budget,layer_budget,patch_budget
@pytest.mark.parametrize('frames,patches_per_frame',[(128,1036),(14,1036),(38,1036),(76,1036)])
@pytest.mark.parametrize('special_queries',[0,1,5])
def test_exact_budget(frames,patches_per_frame,special_queries):
 n=frames*(patches_per_frame+5);s=frames*5;d=frames*special_queries
 for rate in [.06,.08,.10,1/14,.12,9/70,.14]:
  cap=math.floor(rate*n*n);b=total_edge_patch_budget(n,s,d,rate);actual=d*n+(n-d)*(s+b)
  assert actual<=cap and d*n+(n-d)*(s+b+1)>cap
  c=types.SimpleNamespace(total_edge_retention=True,dense_special_queries=special_queries,layer_retentions=[rate]*24,retention=.2)
  assert layer_budget(c,0,n,s)==b
  c.total_edge_retention=False;assert layer_budget(c,0,n,s)==patch_budget(n,s,rate)
