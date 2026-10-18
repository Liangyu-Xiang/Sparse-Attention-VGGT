import sys
import torch
from saf3r.patch.region_history_attention import select_regions,spatial_search_layout
from scripts.visualize_region_attention_stages import selector_stages


def test_return_observer_matches_real_selector_and_stage_subsets():
    torch.manual_seed(771)
    layout=spatial_search_layout(1,4,6,'cpu')
    args=(torch.randn(2,2,3),torch.randn(2,24,3),layout,
        torch.tensor([[.8,.2],[.8,.2]]),torch.zeros(24,dtype=torch.long),
        torch.tensor([[0,1],[-1,-1]]),4,'B')
    reference=select_regions(*args)
    observed={}
    def callback(frame,event,result):
        if event=='return' and frame.f_code is select_regions.__code__:
            observed.update(selector_stages(frame.f_locals))
    old=sys.getprofile();sys.setprofile(callback)
    try:actual=select_regions(*args)
    finally:sys.setprofile(old)
    assert torch.equal(reference[0],actual[0])
    assert observed['history_filtered'].sum(1).tolist()==[2,0]
    assert observed['new_representatives'].sum(1).tolist()==[2,4]
    assert observed['history_plus_new'].sum(1).tolist()==[4,4]
    assert observed['history_covered_fine'][0].sum()==4
    assert not (observed['selected_fine']&observed['history_covered_fine']).any()
