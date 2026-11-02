"""Isolate ready-history implementation from the frozen 28.18s reference."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/tmp/zdrop_m5c_pipeline_dynamic_20261102'));p.add_argument('--destination',type=Path,default=Path('/tmp/zdrop_ready_history_local'));a=p.parse_args()
    if a.destination.exists():raise ValueError('Fresh destination required')
    hashes=json.loads((ROOT/'outputs/m5c_pipeline_dynamic_heads128_20261102/manifest.json').read_text())['source']['after']['sha256']
    assert all(hashlib.sha256((a.source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    shutil.copytree(a.source,a.destination,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files=['saf3r/patch/reuse_ready_history.py','saf3r/triton/reuse_ready_history.py','saf3r/triton/reuse_explore_av_ready.py']
    for n in files:shutil.copy2(ROOT/n,a.destination/n)
    path=a.destination/'saf3r/patch/reuse_residual_exploration.py';s=path.read_text()
    s=s.replace('self.previous_layer=-1;self.records=[]','self.ready_allocator_trims=0;self.ready_written={};self.previous_layer=-1;self.records=[]')
    s=s.replace('from . import reuse_m5c_pipeline as m5c','from . import reuse_ready_history as m5c\n            if self.layer==0:m5c.ensure_cache(s,q)')
    s=s.replace('from saf3r.triton.reuse_explore_av import weighted_attention','from saf3r.triton.reuse_explore_av_ready import weighted_attention')
    s=s.replace('single_qk=c.evidence_mode=="saved_logits",relative_reference=reference)', 'single_qk=c.evidence_mode=="saved_logits",relative_reference=reference,emit_evidence=self.layer<23)')
    s=s.replace('if reference is None:result,mass=execution','if reference is None:result,mass,ready_logits=execution;del execution')
    old="            if c.history_mode=='dense_m5c':\n                with stage(s,'m5c_evidence_and_host_copy'),torch.autocast(x.device.type,enabled=False):\n                    m5c.save_evidence(s,q,k,start,ids,actual,m5c_observation,o.scale)"
    assert old in s;s=s.replace(old,"            if c.history_mode=='dense_m5c' and self.layer<23:\n                m5c_observation['ready_logits']=ready_logits\n                with stage(s,'m5c_evidence_and_host_copy'),torch.autocast(x.device.type,enabled=False):\n                    m5c.save_evidence(s,q,k,start,ids,actual,m5c_observation,o.scale)\n                del ready_logits,m5c_observation['ready_logits']")
    s=s.replace("if bool(torch.stack(s.m5c_count_errors).any()):", "if s.m5c_count_errors and bool(torch.stack(s.m5c_count_errors).any()):")
    s=s.replace('s.m5c_previous_q=q.detach().clone();s.m5c_previous_k=k.detach().clone();s.m5c_previous_scale=o.scale', 's.m5c_previous_q=None;s.m5c_previous_k=None;s.m5c_previous_scale=None\n            if self.layer==23:m5c.release(s)')
    old='s.m5c_previous_q.numel()*s.m5c_previous_q.element_size()+s.m5c_previous_k.numel()*s.m5c_previous_k.element_size()';assert old in s;s=s.replace(old,'0')
    s=s.replace('m5c.device_bytes(s.m5c_previous)','m5c.device_bytes(s)')
    s=s.replace('torch.stack(s.m5c_seed_errors).max(),',"torch.stack(s.m5c_seed_errors).max() if s.m5c_seed_errors else torch.tensor(0.,device=x.device),")
    s=s.replace("history_cache_dtype='float32_statistics_model_qk_dtype'","history_cache_dtype='positive_fp16_u14_edges_float32_grid_statistics'")
    s=s.replace("history_estimator='D2 + M5c'","history_estimator='Ready D2 conditionals + M5c'")
    s=s.replace("history='D2 full-Key probabilities reconstructed exactly from previous Q/K; M5c beta(1,2)'", "history='Prepared preceding-layer conditionals; FP16 edge cache; no old-QK reconstruction; M5c beta(1,2)'")
    s=s.replace("history_cache_device='gpu_grid_statistics_gpu_previous_qk'","history_cache_device='gpu_prepared_conditionals'")
    s=s.replace('history_calibration_gpu_bytes=m5c.device_bytes(s),', 'history_calibration_gpu_bytes=m5c.device_bytes(s),ready_probe_scratch_bytes=min(c.window_groups,s.groups)*q.shape[0]*4*s.budget*4 if self.layer<23 else 0,ready_edge_cache_bytes=0 if s.ready_edges is None else s.ready_edges.numel()*s.ready_edges.element_size(),')
    s=s.replace('            s.history_pooled_fp32=None','            s.history_pooled_fp32=None\n            m5c.trim(s)')
    s=s.replace('ready_edge_cache_bytes=0 if s.ready_edges is None else s.ready_edges.numel()*s.ready_edges.element_size(),','ready_edge_cache_bytes=0 if s.ready_edges is None else s.ready_edges.numel()*s.ready_edges.element_size(),ready_allocator_trims=s.ready_allocator_trims,')
    s=s.replace('m5c.host_bytes(s.m5c_previous)','m5c.host_bytes(s)')
    path.write_text(s)
    evaluator=a.destination/'scripts/evaluate_zdrop_history_states.py';text=evaluator.read_text();extra=files+['saf3r/patch/reuse_m5c_equivalent.py','saf3r/triton/reuse_m5c_equivalent.py','saf3r/patch/reuse_m5c_pipeline.py','saf3r/triton/reuse_drop_moment.py'];text=text.replace('names=set(sources)|{','names=set(sources)|{'+','.join(repr(n) for n in extra)+',');evaluator.write_text(text)
    print(a.destination)


if __name__=='__main__':main()
