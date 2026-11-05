"""Independent Dense trajectories and causal sparse ablations; no geometry GT."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--configs',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
    from saf3r.patch.region_history_attention import prepare_qkv
    from scripts.quality_metrics import quality_record,quality_markdown
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    quality=quality_record({},'Independent Dense trajectory / causal numerical diagnostic; no pose/recon_unposed ground truth or warmed speed comparison')
    save(a.output_dir/'quality.json',quality);(a.output_dir/'quality.md').write_text(quality_markdown(quality))
    save(a.output_dir/'status.json',dict(stage='loading'))
    inp=json.loads(a.input.read_text());paths=inp['image_paths']
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',paths).cuda()
    frames=len(images);per_frame=images.shape[-2]//14*(images.shape[-1]//14)+5;n=frames*per_frame
    all_ids=torch.arange(n,device='cuda');camera=all_ids[all_ids%per_frame==0]
    register=all_ids[(all_ids%per_frame>0)&(all_ids%per_frame<5)];patch=all_ids[all_ids%per_frame>=5]
    def sample(ids,count):return ids[torch.arange(min(len(ids),count),device='cuda')*len(ids)//min(len(ids),count)]
    groups={'camera':sample(camera,16),'register':sample(register,16),'patch':sample(patch,64)}
    ids=torch.cat(list(groups.values()));ranges={};offset=0
    for name,ix in groups.items():ranges[name]=slice(offset,offset+len(ix));offset+=len(ix)
    save(a.output_dir/'input.json',dict(image_paths=paths,shape=list(images.shape),checkpoint_sha256=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest(),sampled_queries={k:v.cpu().tolist() for k,v in groups.items()},hardware=dict(gpu=torch.cuda.get_device_name(),uuid=str(torch.cuda.get_device_properties(0).uuid))))
    originals=[b.attn for b in model.aggregator.global_blocks]
    def descriptor(raw):
        from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
        ext,_=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
        return dict(pose=raw['pose_enc'].float().cpu().numpy(),extrinsics=ext.float().squeeze(0).cpu().numpy(),log_depth=raw['depth'].float().squeeze(0)[...,0][:,::16,::16].clamp_min(1e-8).log().cpu().numpy())
    def pair_pose_error(student,teacher):
        i,j=np.triu_indices(len(student),1)
        def pair(ext):
            r=ext[j,:3,:3]@ext[i,:3,:3].transpose(0,2,1)
            t=ext[j,:3,3]-(r@ext[i,:3,3,None])[...,0]
            return r,t
        rs,ts=pair(student);rt,tt=pair(teacher)
        rotation=np.degrees(np.arccos(np.clip((np.trace(rs@rt.transpose(0,2,1),axis1=1,axis2=2)-1)/2,-1,1)))
        sn=np.linalg.norm(ts,axis=1);tn=np.linalg.norm(tt,axis=1);valid=(sn>1e-8)&(tn>1e-8)
        angle=np.degrees(np.arccos(np.clip(np.sum(ts[valid]*tt[valid],axis=1)/(sn[valid]*tn[valid]),-1,1)))
        return dict(rotation_mean_degrees=float(rotation.mean()),rotation_p90_degrees=float(np.percentile(rotation,90)),translation_direction_mean_degrees=float(angle.mean()))
    def relative(x,y):return float((x.float()-y.float()).square().sum()/y.float().square().sum().clamp_min(1e-12))**.5
    def run(label,config=None,reference=None):
        folder=a.output_dir/label;folder.mkdir();save(folder/'quality.json',quality);(folder/'quality.md').write_text(quality_markdown(quality))
        state=None;captured={};local={};structure={};handles=[]
        if config is not None:state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config))
        for layer,block in enumerate(model.aggregator.global_blocks):
            def attn_hook(module,args,kwargs,output,layer=layer):
                x=args[0];captured.setdefault(layer,{})['input']=x[:,ids].detach().cpu()
                captured[layer]['attention']=output[:,ids].detach().cpu()
                if config is not None:
                    original=module.original;pos=args[1] if len(args)>1 else kwargs.get('pos')
                    q,k,v=prepare_qkv(original,x,pos);q=q.to(v.dtype);k=k.to(v.dtype)
                    dense=torch.nn.functional.scaled_dot_product_attention(q[:,ids].unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),dropout_p=0.,scale=original.scale)[0]
                    dense=dense.transpose(0,1).reshape(len(ids),-1)
                    dense=torch.nn.functional.linear(dense,original.proj.weight,original.proj.bias)
                    local[layer]={name:relative(output[0,ids][sl],dense[sl]) for name,sl in ranges.items()}
                else:
                    pos=args[1] if len(args)>1 else kwargs.get('pos')
                    qq,kk,vv=prepare_qkv(module,x,pos)
                    logits=(qq[:,ids].float()@kk.float().transpose(-1,-2))*module.scale
                    probability=logits.softmax(-1)
                    structure[layer]={name:probability[:,sl,:per_frame].sum(-1).mean(1).cpu().tolist() for name,sl in ranges.items()}
            def block_hook(module,args,output,layer=layer):
                captured[layer]['block']=(output[0] if isinstance(output,tuple) else output)[:,ids].detach().cpu()
            handles.append(block.attn.register_forward_hook(attn_hook,with_kwargs=True))
            handles.append(block.register_forward_hook(block_hook))
        save(a.output_dir/'status.json',dict(stage='diagnostic_forward',variant=label))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;desc=descriptor(raw)
        np.savez_compressed(folder/'predictions.npz',**desc)
        row=dict(label=label,config=config,quality=quality,instrumented_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                 scope='Instrumented diagnostic including extra sampled Dense QK after sparse mask/output; not performance comparison',layers=[])
        if reference is not None:
            for layer in range(24):
                r=dict(layer=layer,local_same_qkv=local[layer],independent_dense={})
                for kind in ['input','attention','block']:
                    r['independent_dense'][kind]={name:relative(captured[layer][kind][0,sl],reference[0][layer][kind][0,sl]) for name,sl in ranges.items()}
                row['layers'].append(r)
            teacher=reference[1]
            row['prediction_deviation']=dict(pose_relative_l2=float(np.linalg.norm(desc['pose']-teacher['pose'])/max(1e-12,np.linalg.norm(teacher['pose']))),
                log_depth_rmse=float(np.mean((desc['log_depth']-teacher['log_depth'])**2)**.5))
            row['prediction_deviation'].update(pair_pose_error(desc['extrinsics'],teacher['extrinsics']))
        else:row['first_frame_dense_mass_by_head']=structure
        save(folder/'result.json',row)
        if state is not None:
            def cpu(v):
                if torch.is_tensor(v):return v.detach().cpu().tolist()
                if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
                if isinstance(v,list):return [cpu(x) for x in v]
                return v
            save(folder/'layers.json',cpu(state.records));state.reset_handle.remove()
        for handle in handles:handle.remove()
        for block,original in zip(model.aggregator.global_blocks,originals):block.attn=original
        if hasattr(model,'reuse_explore_state'):del model.reuse_explore_state
        del raw,state;gc.collect();torch.cuda.empty_cache()
        print('DONE',label,row.get('prediction_deviation'),flush=True)
        return captured,desc,row
    teacher=run('dense_reference');rows=[]
    for path in a.configs:
        captured,desc,row=run(path.stem,json.loads(path.read_text()),teacher);rows.append(row)
        save(a.output_dir/'summary.json',dict(rows=rows,quality=quality))
    repeat=run('dense_repeat')
    save(a.output_dir/'reference_repeat.json',{key:dict(equal=bool(np.array_equal(teacher[1][key],repeat[1][key])),max_abs=float(np.max(np.abs(teacher[1][key]-repeat[1][key])))) for key in ['pose','log_depth']})
    save(a.output_dir/'status.json',dict(stage='diagnostic_complete'))

if __name__=='__main__':main()
