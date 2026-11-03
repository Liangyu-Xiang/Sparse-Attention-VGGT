import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_coarse_observation import m5c,patch

def diff(x,y):return dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-30)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True);torch.manual_seed(0);checks=[]
    for g,h,r,s in [(3,3,7,12),(64,16,512,640)]:
        base=torch.randn(h,g,4,r,device='cuda').transpose(0,1);special=torch.randn(h,g,4,s,device='cuda').transpose(0,1);sizes=torch.randint(1,267,(r,),device='cuda').float();logs=sizes.log()
        z=torch.logaddexp((base+logs).logsumexp(-1),special.logsumexp(-1))[...,None];expected=(base-z).exp()*sizes
        value,lz=m5c(base,special,sizes,logs);d=diff(expected,value);ld=diff(z,lz);assert d['relative_l2']<5e-6 and ld['max_abs']<1e-5
        dot=torch.randn(h,g,4,r,device='cuda',dtype=torch.bfloat16);previous=torch.rand(g,r,device='cuda').half();previous[:,:2]=0;real=(previous.float()*.7).half()
        prob=(dot.float()*.125+logs).softmax(-1).mean((0,2));beta=torch.where(previous.float()>1e-12,(prob/previous.float().clamp_min(1e-12)).clamp_max(1),0);demand=(prob-beta*real.float()).clamp_min(0)
        pp,dd=patch(dot,logs,previous,real,.125);pd=diff(prob,pp);de=diff(demand,dd);assert pd['relative_l2']<5e-6 and de['relative_l2']<5e-6
        checks.append(dict(groups=g,heads=h,m5c=d,logz=ld,patch_prob=pd,demand=de))
    result=dict(complete=True,checks=checks,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Coarse observation kernel numerical diagnostic',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
