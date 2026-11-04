"""Synthetic temporary fixtures test reporting arithmetic, not model quality."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_seven_scene_macro_and_ratio_of_mean_seconds_and_missing_scene(tmp_path):
    root=Path(__file__).resolve().parents[1]
    scenes=['chess','fire','heads','office','pumpkin','redkitchen','stairs']
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(datasets={'7scenes':scenes},methods={'candidate':{}})))
    runs=tmp_path/'runs';reports=tmp_path/'reports';times=[4,8,4,8,4,8,4]
    for scene,seconds in zip(scenes,times):
        inputs=dict(image_paths=['synthetic RGB fixture'],frame_indices=[0],shape=[1,3,14,14],checkpoint_sha256='synthetic')
        data=dict(frames=1,input=inputs,hardware=dict(gpu_uuid='synthetic-GPU',torch='fixture',cuda='fixture'),
                  timing=dict(median_seconds=seconds,samples=[dict(peak_allocated_gib=1)],warmup=1,repeats=3,capture=False),
                  quality=dict(values=dict(auc03=.1,auc30=.9,acc=.02,comp=.04,cd=.03,f1=.8),complete=True,unavailable={}))
        for method in ['candidate','dense','saf3r']:
            folder=runs/'7scenes'/scene/method;folder.mkdir(parents=True)
            copy=json.loads(json.dumps(data))
            if method!='candidate':copy['timing']['median_seconds']=16 if method=='dense' else 8
            (folder/('result.json' if method=='candidate' else 'timing_result.json')).write_text(json.dumps(copy))
            (runs/f'7scenes_{scene}_{method}.monitor.json').write_text(json.dumps(dict(same_gpu_overlap_detected=False,returncode=0)))
    command=[sys.executable,str(root/'scripts/report_integrated_final.py'),'--plan',str(plan),'--runs',str(runs),'--output',str(reports)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    result=json.loads((reports/'summary.json').read_text());macro=result['macro'][0]
    assert result['complete'] and macro['quality_complete']
    mean=sum(times)/7
    assert macro['speedup_vs_dense']==pytest.approx(16/mean)
    assert macro['time_ratio_vs_saf3r']==pytest.approx(mean/8)
    assert macro['cd']==pytest.approx(.03)
    (runs/'7scenes/stairs/candidate/result.json').unlink()
    subprocess.run(command,check=True,capture_output=True,text=True)
    incomplete=json.loads((reports/'summary.json').read_text())['macro'][0]
    assert not incomplete['quality_complete'] and incomplete['completed_scenes']==6
    assert all(incomplete[key] is None for key in ['auc03','auc30','acc','comp','cd','f1'])
