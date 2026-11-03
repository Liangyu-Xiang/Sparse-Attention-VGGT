"""Same16 dual rounds, tie order and fresh inputs; graph replay only."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_compact import budget as old
from saf3r.triton.reuse_budget_graph import budget as new

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(0);rows=[]
    for g,f in [(3,257),(64,26009)]:
        costs=torch.randint(1,8,(g,f,4),device='cuda').float();costs[:,:,0]=0;costs[:,:,1]=1;cap=f
        for seed in range(2):
            e=torch.rand(g,f,4,device='cuda');e[:,:,2].masked_fill_(torch.rand(g,f,device='cuda')>.8,float('inf'));upper=e[:,:,0].amax(1)+1e-12
            a,c=old(e,costs,upper,cap,16);b,d=new(e,costs,upper,cap,16)
            assert torch.equal(a,b) and torch.equal(c,d)
            times={}
            for name,fn in [('old',old),('graph',new)]:
                values=[]
                for i in range(7):
                    x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record();fn(e,costs,upper,cap,16);y.record();y.synchronize()
                    if i>=2:values.append(x.elapsed_time(y))
                times[name]=statistics.median(values)
            rows.append(dict(groups=g,clusters=f,fresh_input_seed=seed,selected_bitwise_equal=True,count_bitwise_equal=True,median_ms=times))
    result=dict(complete=True,rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Solver execution diagnostic only',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
