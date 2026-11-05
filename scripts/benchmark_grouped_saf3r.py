"""Matched original/grouped SAF3R full-forward timing and numerical diagnostic."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch import update_args_per_forward
    from saf3r.patch.grouped_saf3r import install_grouped_saf3r
    from scripts.quality_metrics import quality_record,quality_markdown
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    quality=quality_record({},'Original/grouped SAF3R RGB-only timing and numerical diagnostic; no pose/recon_unposed GT evaluation and no paired Dense timing')
    save(a.output_dir/'quality.json',quality);(a.output_dir/'quality.md').write_text(quality_markdown(quality));save(a.output_dir/'status.json',dict(stage='loading'))
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
    model=build_model(cfg,'cuda');paths=json.loads(a.input.read_text())['image_paths'];images=load_images(model,'vggt',paths).cuda()
    save(a.output_dir/'input.json',dict(image_paths=paths,shape=list(images.shape),checkpoint_sha256=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest(),topk=4,topk_source='lazy first correspondence head',profile=cfg.sparse_config_path,profile_sha256=hashlib.sha256(Path(cfg.sparse_config_path).read_bytes()).hexdigest(),cpu_affinity=sorted(os.sched_getaffinity(0))))
    reference=None;rows=[]
    for variant in ['original_saf3r','grouped_saf3r']:
        if variant=='grouped_saf3r':install_grouped_saf3r(model)
        save(a.output_dir/'status.json',dict(stage='forward',variant=variant));samples=[];cold=[]
        for i in range(4):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            update_args_per_forward(model,'vggt',*images.shape[-2:])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
            torch.cuda.synchronize();sample=dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            (samples if i else cold).append(sample)
            if i==3:prediction={k:raw[k].float().cpu() for k in ['pose_enc','depth','depth_conf']}
            del raw
        compare=None
        if reference is not None:compare={k:dict(exact=bool(torch.equal(v,reference[k])),relative_l2=float((v-reference[k]).norm()/reference[k].norm().clamp_min(1e-12)),max_abs=float((v-reference[k]).abs().max())) for k,v in prediction.items()}
        else:reference=prediction
        row=dict(variant=variant,quality=quality,warmup=1,repeats=3,samples=samples,cold_samples=cold,median_seconds=float(np.median([s['seconds'] for s in samples])),deviation_from_original=compare,hardware=dict(gpu=torch.cuda.get_device_name(),uuid=str(torch.cuda.get_device_properties(0).uuid),cpu_affinity=sorted(os.sched_getaffinity(0))),scope='Full forward includes native per-frame lazy Top-K reset/recomputation and grouped static-geometry initialization on cache miss')
        if variant=='grouped_saf3r':row['layers']=[b.attn.grouped_execution_record for b in model.aggregator.global_blocks]
        rows.append(row);save(a.output_dir/'summary.json',dict(rows=rows));print(variant,row['median_seconds'],compare,flush=True)
    save(a.output_dir/'status.json',dict(stage='diagnostic_complete'))
if __name__=='__main__':main()
