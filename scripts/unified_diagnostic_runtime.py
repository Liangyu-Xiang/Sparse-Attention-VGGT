"""Diagnostic-only compatibility and required N/A quality reporting.

Production selector files are not modified. The probe implementation is the
already archived, four-effective-probe compatibility kernel from the prior run.
"""
import atexit
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
_installed=False

def install(diagnostic_reporting=True):
    global _installed
    if _installed:return
    _installed=True
    import torch
    import saf3r.triton.reuse_explore_av as target
    source=ROOT/'outputs/global_attention/unified_refactor_57bdb78/quality128/chess/source_snapshot/saf3r/triton/reuse_explore_av.py'
    spec=importlib.util.spec_from_file_location('_diagnostic_four_probe_compat',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    target._probe_mass=module._probe_mass
    original=torch.sparse.mm
    def fp32_sparse_mm(a,b,*args,**kwargs):
        with torch.autocast(a.device.type,enabled=False):return original(a,b,*args,**kwargs)
    torch.sparse.mm=fp32_sparse_mm
    if diagnostic_reporting:atexit.register(report)

def pytest_configure(config):install()

def report():
    if '--output-dir' not in sys.argv:return
    out=Path(sys.argv[sys.argv.index('--output-dir')+1])
    if not out.exists():return
    from scripts.quality_metrics import quality_record
    from scripts.evaluate_attention_first_residual_value import write_quality
    q=quality_record({},'Stage1 diagnostic/microbenchmark only: pose and recon_unposed not evaluated; no paired Dense timing')
    write_quality(out,q)
    (out/'diagnostic_provenance.json').write_text(json.dumps(dict(
        upstream='1a7bb41d955da284eded195a2e0b11faf49c2ec3',argv=sys.argv,
        compatibility='Diagnostic-only FP32 sparse.mm and archived four-effective-probe padded kernel; production files unchanged',
        quality_evaluation=False,speedup_vs_dense=None),indent=2)+'\n')
