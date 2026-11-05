"""Evaluate saved independent predictions on CPU; no model/control forward rerun."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    a=p.parse_args()
    import torch
    import numpy as np
    import open3d as o3d
    from addict import Dict
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from evaluate_da3bench import load_scene_data,evaluate
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from scripts.quality_metrics import quality_record,quality_markdown
    assert not torch.cuda.is_available(),'This evaluator must run with CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(2);np.random.seed(0);o3d.utility.random.seed(0)
    metadata=json.loads((a.run/'input.json').read_text())
    assert not metadata.get('teacher_forced',False),'Counterfactual trajectories are not quality candidates'
    dataset=MV_REGISTRY.get('eth3d')()
    scene,gt=load_scene_data(dataset,'eth3d',a.scene,128,'uniform',seed=0)
    assert [str(Path(x).resolve()) for x in scene.image_files]==[str(Path(x).resolve()) for x in metadata['image_paths']]
    raw=torch.load(a.run/'predictions.pt',weights_only=True)
    images=torch.load(a.images,weights_only=True)
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    pred=Dict(extrinsics=ext.squeeze().float().numpy(),intrinsics=intr.squeeze().float().numpy(),
              depth=raw['depth'].squeeze().float().numpy(),conf=raw['depth_conf'].squeeze().float().numpy(),
              processed_images=images.squeeze().permute(0,2,3,1).numpy())
    start=time.perf_counter();metrics={};errors={}
    for task in ('pose','recon_unposed'):
        try:
            np.random.seed(0);o3d.utility.random.seed(0)
            metrics.update(evaluate(dataset,gt,pred,[task]))
        except Exception as exc:errors[task]=f'{type(exc).__name__}: {exc}'
    quality=quality_record(metrics,str(errors) if errors else 'Task unavailable')
    quality['speedup_unavailable']='Saved-prediction quality postprocessing, no paired warmed Dense timing'
    out=a.run/'saved_quality';out.mkdir(exist_ok=False)
    record=dict(scene=a.scene,frames=len(images),quality=quality,metrics=metrics,errors=errors,
                input=metadata,quality_protocol=dict(tasks=['pose','recon_unposed'],f1_threshold_m=dataset.eval_threshold,
                downsample_m=dataset.down_sample,alignment='repository ETH3D _prep_unposed',seed=0),
                cpu_evaluation_seconds=time.perf_counter()-start)
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    (out/'quality.md').write_text(quality_markdown(quality))
    print(a.scene,a.run.name,quality,flush=True)


if __name__=='__main__':main()
