"""Kernel-only audit: special Keys always retained; all special Queries see every Key."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import torch
 from saf3r.patch.protected_frame_keys import protected_layout
 from saf3r.patch.dense_special_queries import correct_output
 from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
 from scripts.quality_metrics import quality_record,quality_markdown
 out=ROOT/'outputs/special_token_isolation_audit_20261105';out.mkdir(parents=True,exist_ok=True)
 quality=quality_record({},'Synthetic attention-kernel diagnostic only; no model pose/recon_unposed quality or paired full-forward Dense timing')
 (out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n');(out/'quality.md').write_text(quality_markdown(quality))
 patches,specials,semantic,_=protected_layout(2,4,4,'cuda',False)
 assert len(specials)==10 and not set(patches.tolist())&set(specials.tolist())
 h,n,d=2,42,16
 q=torch.zeros(h,n,d,device='cuda',dtype=torch.float16);k=q.clone();v=q.clone()
 # Special Key contributes to ordinary Queries, including across frames.
 v[:,specials[0],0]=2
 # An ordinary Key excluded from sparse support must still affect ALL special Queries.
 omitted=patches[2];v[:,omitted,1]=3
 ids=patches[:2].int()[None].expand(2,-1).contiguous()
 sparse,_=direct_sparse_attention(q,k,v,ids,specials,d**-.5)
 torch.testing.assert_close(sparse[:,:,0].float(),torch.full((h,n),2/12,device='cuda'),atol=3e-4,rtol=1e-3)
 assert torch.count_nonzero(sparse[:,:,1])==0
 full=(q.float()@k.float().transpose(-1,-2)*d**-.5).softmax(-1)@v.float()
 rows=[]
 for filename in ['reuse_layout_fixed_all_specials_dense.json','reuse_partial_states_all_specials_dense.json']:
  cfg=json.loads((ROOT/'configs/evaluation'/filename).read_text());assert cfg['dense_special_queries']==5
  result,dense_ids=correct_output(q,k,v,sparse.clone(),semantic,cfg['dense_special_queries'],d**-.5)
  assert torch.equal(dense_ids,semantic)
  torch.testing.assert_close(result[:,semantic].float(),full[:,semantic],atol=3e-4,rtol=1e-3)
  assert bool((result[:,semantic,1]>0).all()),'Omitted ordinary Key must participate for camera AND register Queries'
  assert torch.equal(result[:,patches],sparse[:,patches])
  rows.append(dict(config=filename,all_special_keys_retained=True,specials_excluded_from_sparse_patch_domain=True,all_camera_and_register_queries_dense=True,omitted_patch_key_affects_every_special_query=True,ordinary_queries_receive_special_key_values=True,max_abs_special_output_error=float((result[:,semantic].float()-full[:,semantic]).abs().max())))
 (out/'audit.json').write_text(json.dumps(dict(scope='Synthetic two-frame attention; not a full-model or dataset quality evaluation',gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),quality=quality,checks=rows),indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
