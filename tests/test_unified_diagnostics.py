import torch
import numpy as np
from scripts.diagnose_history_pipeline import support_metrics
from scripts.summarize_value_compressibility import rank

def test_pipeline_qk_shape_and_support_reference():
    q=torch.randn(5,3);k=torch.randn(5,3)
    p=(q@k.T).softmax(-1)
    target=p.topk(2,dim=-1).indices
    candidates=torch.arange(5)[None]
    r=support_metrics(p[None],target[None],candidates,torch.arange(5),torch.arange(5))
    assert r['recall']==1 and abs(r['mass']-1)<1e-6 and r['density']==1

def test_spearman_ties_have_equal_rank():
    r=rank(np.array([2.,1.,1.,3.]))
    assert r.tolist()==[3.,1.5,1.5,4.]
