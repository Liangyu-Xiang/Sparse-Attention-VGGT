"""Audit full ETH3D six-metric quality and all-layer camera features for10/25 retention."""
import csv,gzip,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_A_cluster20_early10_late25_20261106';BASE=ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105';DENSE=ROOT/'outputs/camera_token_comparison_20261105';FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def report():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from scripts.run_eth3d_layout_fix_evaluation import verify
 from scripts.quality_metrics import quality_markdown
 plan=read(OUT/'frozen_plan.json');verify(plan);cfg=read(Path(plan['config']));assert cfg==read(ROOT/'configs/evaluation/reuse_rep_history_A_all_specials_cluster20.json')|dict(layer_retentions=[.10]*6+[.25]*18)
 assert not read(OUT/'execution.json')['failures'];queue=read(OUT/'queue.json');assert not queue['pending'] and len(queue['claims'])==11 and all(c['returncode']==0 for c in queue['claims'])
 rows=[];fr=[];resources=[];audits=[];repeat=[];protocol=None
 (OUT/'raw_metrics').mkdir(exist_ok=True)
 for scene in plan['scenes']:
  folder=OUT/'runs'/scene;after=read(folder/'result.json');before=read(BASE/'quality/A_all_specials_cluster20/runs'/scene/'result.json');cam=read(folder/'camera/result.json');dense_input=read(DENSE/'full'/scene/'dense/input.json')
  assert after['quality']['complete'] and cam['complete'] and before['quality']['complete'];assert after['quality_protocol']==before['quality_protocol'];protocol=after['quality_protocol']
  for k in ['image_paths','frame_indices','shape','checkpoint_sha256']:assert after['input'][k]==before['input'][k]==dense_input[k]==cam['input'][k]
  assert all(after['config'][k]==v for k,v in cfg.items());assert all(after['source_sha256'][k]==v for k,v in plan['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
  g=next(c['gpu'] for c in queue['claims'] if c['scene']==scene);mon=read(OUT/f'worker{g}_gpu.json');assert g in [4,5,6] and mon['returncode']==0 and not mon['same_gpu_overlap_detected'];assert after['hardware']['gpu_uuid'].removeprefix('GPU-')==mon['gpu_uuid'].removeprefix('GPU-')
  layers=read(folder/'layers.json');assert len(layers)==24
  for l in layers:
   n=after['frames']*((after['input']['shape'][-2]//14)*(after['input']['shape'][-1]//14)+5)
   assert l['patch_budget']==int(n*cfg['layer_retentions'][l['layer']])-l['special_keys']
   assert l['dense_special_query_count']==5*after['frames'] and l['semantic_special_keys']==5*after['frames'] and l['special_keys']>=l['semantic_special_keys'] and l.get('native_cost_count_error_mean',0)==0
  for m,d in [('previous_10_22_10',before),('new_10_25_25',after)]:
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10;rows.append(dict(method=m,scene=scene,frames=d['frames'],**v))
  for r in cam['features']:
   for token,v in r.items():
    if isinstance(v,dict) and 'relative_l2' in v:fr.append(dict(scene=scene,stage=r['stage'],token=token,**{k:v[k] for k in ['cosine','relative_l2','norm_ratio','mean_token_cosine','token_relative_l2_p90']}))
  for k,v in cam['normal_vs_diagnostic'].items():repeat.append(dict(scene=scene,output=k,**v))
  assert next(r for r in cam['features'] if r['stage']=='frame_00_block_input')['camera']['relative_l2']==0
  resources.append(dict(scene=scene,gpu=g,frames=after['frames'],cold_full_forward_seconds=after['timing']['median_seconds'],peak_allocated_gib=max(r['peak_allocated_gib'] for r in after['timing']['samples']),actual_unique_density=statistics.mean(l['actual_unique_density'] for l in layers),diagnostic_seconds=cam['instrumented_seconds'],dense_speedup=None,reason='ETH3D quality and camera diagnostics; no paired warmed Dense timing'))
  audits.append(dict(scene=scene,matched_input_checkpoint_preprocessing=True,matched_quality_protocol=True,frozen_native_source=True,frozen_configuration=True,all24_layers=True,all_special_queries_keys=True,permitted_gpu_isolated=True,initial_camera_exact=True))
  for name,path in [('quality',folder/'result.json'),('camera',folder/'camera/result.json')]:
   raw=path.read_bytes();(OUT/'raw_metrics'/f'{scene}_{name}.json.gz').write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
 macro={m:{k:statistics.mean(r[k] for r in rows if r['method']==m) for k in FIELDS} for m in ['previous_10_22_10','new_10_25_25']};delta={k:macro['new_10_25_25'][k]-macro['previous_10_22_10'][k] for k in FIELDS}
 groups={}
 for r in fr:groups.setdefault((r['stage'],r['token']),[]).append(r)
 fm=[dict(stage=stage,token=token,**{k:statistics.mean(r[k] for r in rs) for k in ['cosine','relative_l2','norm_ratio','mean_token_cosine','token_relative_l2_p90']},scenes=len(rs)) for (stage,token),rs in groups.items()]
 assert len(audits)==11 and sum(r['frames'] for r in resources)==377 and all(r['scenes']==11 for r in fm)
 csvsave(OUT/'quality_per_scene.csv',rows);csvsave(OUT/'camera_per_scene.csv',fr);csvsave(OUT/'camera_macro.csv',fm);csvsave(OUT/'resources.csv',resources);csvsave(OUT/'normal_vs_diagnostic.csv',repeat)
 def feature(stage,k,m='new'):
  if m=='new':return next(r[k] for r in fm if r['stage']==stage and r['token']=='camera')
  old=read(DENSE/'summary.json')['feature_macro'];return next(r[k] for r in old if r['stage']==stage and r['token']=='camera' and r['method']=='ours')
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for row,kind in enumerate(['frame','global']):
  for col,metric in enumerate(['cosine','relative_l2']):
   for m,label,color in [('old','Previous 10/22/10','#2471a3'),('new','New 10/25/25','#d35400')]:axs[row,col].plot(range(1,25),[feature(f'{kind}_{i:02d}_block_output',metric,m) for i in range(24)],label=label,color=color)
   axs[row,col].set_title(f'{kind}: camera block output {metric} vs Dense');axs[row,col].set_xlabel('Layer (1-based)');axs[row,col].grid(alpha=.2);axs[row,col].legend()
 fig.savefig(OUT/'camera_layers.png',dpi=170);fig.savefig(OUT/'camera_layers.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
 for col,metric in enumerate(['cosine','relative_l2']):
  for stage in ['block_input','attention_pre_Wo','attention_post_Wo','post_attention_residual','block_output']:axs[col].plot(range(1,25),[feature(f'global_{i:02d}_{stage}',metric) for i in range(24)],label=stage)
  axs[col].set_title(f'New global camera stages: {metric} vs Dense');axs[col].set_xlabel('Layer');axs[col].legend(fontsize=8);axs[col].grid(alpha=.2)
 fig.savefig(OUT/'camera_stages.png',dpi=170);plt.close(fig)
 speed=None;speed_path=OUT/'speed_heads128/timing_result.json'
 if speed_path.exists():
  d=read(speed_path);mon=read(OUT/'speed_gpu.json');assert mon['returncode']==0 and mon['gpu']==6 and not mon['same_gpu_overlap_detected'];assert d['timing']['warmup']==2 and d['timing']['repeats']==5
  speed=dict(seconds=d['timing']['median_seconds'],peak_allocated_gib=max(r['peak_allocated_gib'] for r in d['timing']['samples']),dense_speedup=None,reason='Changed method only; unchanged controls reused from historical batches; no contemporaneous paired Dense timing',scope='7Scenes heads128 only; GPU6;2warmups5repeats complete forward')
 summary=dict(complete=True,adopted=None,adoption_reason='User-requested frozen experiment; no automatic production replacement',scenes=11,frames=377,quality_macro=macro,delta_vs_previous=delta,camera_feature_macro=fm,audits=audits,resources=resources,quality_protocol=protocol,speed=speed,speedup_vs_dense=None,speedup_unavailable='No contemporaneous paired warmed Dense timing; unchanged controls not rerun',units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'))
 save(OUT/'summary.json',summary)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=['# A＋全特殊token＋20%簇：浅层10%、中深层25%保留率','','唯一参数变化：前6个全局层的Key保留率保持10%，第7–18层从22%改为25%，第19–24层从10%改为25%；预算均值从16%增至21.25%。此处百分比为Key保留预算，不是删除比例；特殊Key计入总Key预算，特殊Query始终完整注意力。帧内注意力不做该稀疏化，保持稠密；每帧camera＋4个register双向保护、History Loss A、20%簇数、Query group256、内核及其余参数不变。','','ETH3D沿用既有filtered11协议：11/11场景、377帧、每场景最多128帧uniform seed0；terrace/meadow按原协议排除。相同checkpoint、完整精确帧表、预处理和评测设置；没有新过滤。只使用GPU4/5/6动态队列。旧方法质量和Dense逐层特征复用完整既有数据，没有重跑未变化对照。','','## 六项精度','','AUC/F1为百分数，ACC/Comp/CD为米；CD=(ACC+Comp)/2，非平方距离；11场景等权宏平均。','','|方法|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|','|---|---:|---:|---:|---:|---:|---:|']
 for m,v in macro.items():lines.append('| '+m+' | '+cells(v)+' |')
 lines+=['',f"相对旧方案：AUC@3 {delta['auc03']*100:+.4f}个百分点，AUC@30 {delta['auc30']*100:+.4f}个百分点，F1 {delta['f1']*100:+.4f}个百分点，CD {delta['cd']:+.6f}米。",'',f"Pose＋recon_unposed；F1阈值{protocol['f1_threshold_m']}m，下采样{protocol['downsample_m']}m，voxel{protocol['voxel_length_m']}m，TSDF trunc{protocol['sdf_trunc_m']}m，mesh采样{protocol['mesh_sample_points']}点；原对齐/融合/掩码不变。方法本身不使用SAF3R Top-K/profile。未使用7Scenes测试真值调参。",'','## 逐场景六指标','','|方法|场景|帧数|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['method']} | {r['scene']} | {r['frames']} | "+cells(r)+' |')
 lines+=['','## Camera token逐层相似度','','下表均相对保存的Dense特征，层号从1开始。cos=<x,d>/(||x||||d||)，relativeL2=||x-d||/||d||；每场景展平所有camera token后计算，再做场景等权均值。Dense参考cos=1、relativeL2=0。注意力输出为实际投影后特征，Block输出包含残差与MLP；不是注意力概率分布。另有投影前、输入、归一化、注意力残差后，以及相机头的完整逐层CSV。','','|层|帧内Block cos|帧内Block relL2|全局attention cos|全局attention relL2|全局Block cos|全局Block relL2|','|---|---:|---:|---:|---:|---:|---:|']
 for i in range(24):
  vals=[feature(f'{kind}_{i:02d}_{stage}',k) for kind,stage in [('frame','block_output'),('global','attention_post_Wo'),('global','block_output')] for k in ['cosine','relative_l2']];lines.append('| '+str(i+1)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
 lines+=['',f"相机头输入：新方案cos={feature('camera_head_input','cosine'):.6f}、relativeL2={feature('camera_head_input','relative_l2'):.6f}；旧方案cos={feature('camera_head_input','cosine','old'):.6f}、relativeL2={feature('camera_head_input','relative_l2','old'):.6f}。整体特征相似度不能代替几何质量。本方案作为独立实验保留；浅层5%与10%的直接对照另见shallow_ablation.md。",'','## 相机头输入逐场景','','|场景|cosine|relative L2|','|---|---:|---:|']
 for scene in plan['scenes']:
  r=next(r for r in fr if r['scene']==scene and r['stage']=='camera_head_input' and r['token']=='camera');lines.append(f"| {scene} | {r['cosine']:.6f} | {r['relative_l2']:.6f} |")
 lines+=['| 场景等权均值 | '+f"{feature('camera_head_input','cosine'):.6f} | {feature('camera_head_input','relative_l2'):.6f} |",'','## 运行与完整性','','质量使用无hook的正常完整前向预测。随后额外运行只读camera特征hook诊断，逐场景与质量预测的数值重复性差异保存在normal_vs_diagnostic.csv。完整概率重建未在本轮执行。诊断自身六项质量为N/A（仅特征观测）；完整六项质量来自正常预测的独立pose/recon_unposed评测。特征相似度不是位姿精度。','','ETH3D时间为0暖机、1次正常冷前向，包含JIT/缓存初始化，不能当暖机速度；GPU显存为Torch峰值allocated。诊断耗时另存resources.csv，不混用。实际密度记录包含稀疏observer及特殊Query完整SDPA覆盖的重复计算，因此可略高于设定Key预算；21.25%预算并非实际总计算量严格21.25%。','','|场景|GPU|正常冷前向s|峰值GiB|实际平均唯一Key密度|','|---|---:|---:|---:|---:|']
 for r in resources:lines.append(f"| {r['scene']} | {r['gpu']} | {r['cold_full_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['actual_unique_density']:.6f} |")
 if speed:
  lines+=['','## 7Scenes速度（仅修改后方法）','',f"heads128、GPU6、2暖机＋5重复：完整前向中位数 **{speed['seconds']:.6f}s**，峰值allocated **{speed['peak_allocated_gib']:.4f}GiB**。包含每次reset、聚类、路由、历史、重排、全特殊Query补算。无诊断hook；不含磁盘模型/图像加载。计时自身六项精度N/A，原因是RGB计时任务，无pose/recon评测。",'', '相对Dense同批加速比为N/A：依据AGENTS.md只运行修改后的方法，未重跑未变化的Dense/SAF3R对照，历史计时不能作为同期速度比较。Dense=1.00x仅是参考定义，不能据此宣称已满足不降速或SAF3R+10%目标。']
 else:lines+=['','相对Dense暖机完整前向加速比N/A：本轮没有同批Dense计时；依据AGENTS.md不重跑未变化控制。历史噪声时间不作为同批比较。']
 lines+=['','原始质量和特征指标已压缩保存raw_metrics/*.json.gz；输入、参数、源码SHA、GPU隔离监控均保留。未将本轮结果与论文公开基准混用。']
 shallow_report(summary)
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');print('RETENTION_REPORT_COMPLETE',json.dumps(dict(macro=macro,delta=delta,camera_head_input={k:feature('camera_head_input',k) for k in ['cosine','relative_l2']},speed=speed)),flush=True)


def shallow_report(new):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 prev=ROOT/'outputs/rep_history_A_cluster20_early05_late25_20261106';old=read(prev/'summary.json');cfg=read(Path(read(OUT/'frozen_plan.json')['config']));oldcfg=read(Path(read(prev/'frozen_plan.json')['config']))
 assert cfg==oldcfg|dict(layer_retentions=[.10]*6+[.25]*18)
 a=read(OUT/'frozen_plan.json')['source_sha256'];b=read(prev/'frozen_plan.json')['source_sha256'];assert all(a[k]==v for k,v in b.items() if k.startswith(('saf3r/','evaluation/')))
 scenes=read(OUT/'frozen_plan.json')['scenes'];rows=[];changes=[]
 for scene in scenes:
  d=read(OUT/'runs'/scene/'result.json');p=read(prev/'runs'/scene/'result.json');assert d['input']==p['input'] and d['quality_protocol']==p['quality_protocol']
  for m,r in [('early05',p),('early10',d)]:rows.append(dict(scene=scene,method=m,**r['quality']['values']))
  changes.append(dict(scene=scene,**{k:d['quality']['values'][k]-p['quality']['values'][k] for k in FIELDS}))
 macro={'early05':old['quality_macro']['new_05_25_25'],'early10':new['quality_macro']['new_10_25_25']};delta={k:macro['early10'][k]-macro['early05'][k] for k in FIELDS};csvsave(OUT/'shallow_ablation_per_scene.csv',rows);csvsave(OUT/'shallow_ablation_scene_delta.csv',changes)
 camera_rows=[]
 def camera(m,kind,i,metric):
  data=old['camera_feature_macro'] if m=='early05' else new['camera_feature_macro'];return next(r[metric] for r in data if r['stage']==f'{kind}_{i:02d}_block_output' and r['token']=='camera')
 for kind in ['frame','global']:
  for i in range(24):camera_rows.append(dict(kind=kind,layer=i+1,**{m+'_'+k:camera(m,kind,i,k) for m in ['early05','early10'] for k in ['cosine','relative_l2']}))
 csvsave(OUT/'shallow_ablation_camera_layers.csv',camera_rows)
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for ri,kind in enumerate(['frame','global']):
  for ci,k in enumerate(['cosine','relative_l2']):
   for m,label in [('early05','5/25/25'),('early10','10/25/25')]:axs[ri,ci].plot(range(1,25),[camera(m,kind,i,k) for i in range(24)],label=label)
   axs[ri,ci].set_title(f'{kind} camera block output {k} vs Dense');axs[ri,ci].set_xlabel('Layer (1-based)');axs[ri,ci].grid(alpha=.2);axs[ri,ci].legend()
 fig.savefig(OUT/'shallow_ablation_camera.png',dpi=170);plt.close(fig)
 save(OUT/'shallow_ablation_summary.json',dict(complete=True,only_change='first6 retention0.05->0.10;remaining18 retention0.25;graph cache8,cluster0.20,A,all-specials and native sources unchanged',quality_macro=macro,delta_early10_minus_early05=delta,scenes=11,frames=377,per_scene=rows,camera_layers=camera_rows,speed=new['speed'],dense_speedup=None,unavailable_reason='No contemporaneous paired warmed Dense; no unchanged controls rerun'))
 lines=['# 浅层Key保留率5%→10%的单因素实验','','本次仅前6个全局层从5%保留提高至10%保留（删除比例95%→90%），后18层均保持25%保留。20%簇数、全特殊token修正、Loss A、Graph缓存容量8、其余参数和所有原生计算/评测源码均逐项核对不变。均值预算20%→21.25%。旧方法结果完整复用。','','ETH3D既有filtered11、377帧，max128 uniform seed0；terrace/meadow依原协议排除，非13场景。相同权重、精确帧表、预处理、pose/recon_unposed协议及重建阈值。AUC/F1百分数；ACC/Comp/CD米；CD=(ACC+Comp)/2非平方；场景等权平均。','','|方案|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|','|---|---:|---:|---:|---:|---:|---:|']
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 for m,v in macro.items():lines.append('| '+m+' | '+cells(v)+' |')
 lines+=['',f"10%减5%：AUC@3 {delta['auc03']*100:+.4f}pp，AUC@30 {delta['auc30']*100:+.4f}pp，F1 {delta['f1']*100:+.4f}pp，CD {delta['cd']:+.6f}m。",'','浅层是两次配置之间唯一干预变量；后续层输入、聚类、状态选择、历史及输出仍会随浅层变化而改变。若质量恢复，说明增加浅层预算在这组固定条件下有帮助，不能据此证明全部退化仅由浅层造成；未做重复统计，微小差异不宣称显著。','','## camera逐层输出相对Dense','','Block输出含残差与MLP；各场景先展平所有camera token计算cosine/relativeL2，再等权平均。Dense参考cos1、relativeL2为0。详细frame/global注意力投影前后、输入、归一化、残差及相机头各迭代在camera_per_scene.csv和camera_macro.csv。','','|全局层|5% cos|5% relL2|10% cos|10% relL2|','|---|---:|---:|---:|---:|']
 for r in camera_rows:
  if r['kind']=='global':lines.append('| '+str(r['layer'])+' | '+' | '.join(f"{r[m+'_'+k]:.6f}" for m in ['early05','early10'] for k in ['cosine','relative_l2'])+' |')
 lines+=['','## 逐场景六指标','','|场景|方案|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append('| '+r['scene']+' | '+r['method']+' | '+cells(r)+' |')
 lines+=['','正常质量预测和额外camera诊断分开保存；重复前向数值差异见normal_vs_diagnostic.csv。诊断自身六指标为N/A，原因是只观测特征；正常质量运行完整评分六指标。ETH3D冷前向不能作为速度比较。速度及同批Dense加速比缺失原因见report.md；未重跑未变化的控制，历史时间不作配对比较。']
 improved=all(delta[k]>0 for k in ['auc03','auc30','f1']) and all(delta[k]<0 for k in ['acc','comp','cd'])
 original=new['quality_macro']['previous_10_22_10']
 lines+=['',f"本次相对5%浅层版本六项宏平均均改善：{improved}。这支持浅层预算过低是退化的重要因素。相对最初10/22/10版本，当前AUC@3 {(macro['early10']['auc03']-original['auc03'])*100:+.4f}pp、AUC@30 {(macro['early10']['auc30']-original['auc30'])*100:+.4f}pp，但F1仍 {(macro['early10']['f1']-original['f1'])*100:+.4f}pp、CD {(macro['early10']['cd']-original['cd']):+.6f}m，不能证明所有退化只来自浅层。"]
 (OUT/'shallow_ablation.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
