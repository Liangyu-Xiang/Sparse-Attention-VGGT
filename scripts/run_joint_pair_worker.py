"""One same-UUID Dense timing/new-method quality pair, without changing method."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gpu_recovery_guard import query_inventory,bounded_command
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record

def execute(args,logpath,uuid):
    foreign=set();waiting=None;last_check=0
    forward_file=Path(args[args.index('--output-dir')+1])/'forward.json'
    with logpath.open('w') as log:
        p=subprocess.Popen([sys.executable]+args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            if time.monotonic()-last_check>=15:
                if not forward_file.exists():
                    inventory=bounded_command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
                    for line in inventory.splitlines():
                        fields=[t.strip() for t in line.split(',')]
                        if len(fields)==2 and fields[0]==uuid and fields[1].isdigit() and int(fields[1])!=p.pid:foreign.add(int(fields[1]))
                last_check=time.monotonic()
                try:
                    state=next(x for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('State:')).split()[1]
                    wchan=Path(f'/proc/{p.pid}/wchan').read_text().strip()
                except (OSError,StopIteration):state='unknown';wchan='unknown'
                if state=='D':waiting=waiting or time.monotonic()
                else:waiting=None
                if waiting and time.monotonic()-waiting>(900 if wchan=='wait_on_page_bit_common' else 120):
                    raise RuntimeError('Child kernel wait: '+wchan+'; no reset or unrelated termination attempted')
            time.sleep(5)
    if p.returncode:raise RuntimeError(f'Child failed ({p.returncode}); see {logpath}')
    return sorted(foreign)

def main(a):
    out=a.output_dir.resolve();dense=out.parents[1]/'paired_dense128'/a.scene
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];logs=out.parents[1]/'logs'
    checkpoint='/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt'
    expected=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128'/a.scene/'dense/input.json'
    common=['--scene',a.scene,'--frames','128','--warmup','1','--repeats','3','--checkpoint',checkpoint,'--expected-input',str(expected)]
    first=execute(['scripts/evaluate_all_edge_history.py','--method','dense','--diagnostic','--output-dir',str(dense)]+common,
        logs/f'dense_{a.scene}.log',uuid)
    while query_inventory([uuid])[0]['memory_used_mib']>256:time.sleep(10)
    second=execute(['scripts/evaluate_all_edge_history.py','--method','reuse_explore','--output-dir',str(out),
        '--baseline',str(dense/'diagnostic.json')]+common,logs/f'joint_{a.scene}.log',uuid)
    d=json.loads((dense/'diagnostic.json').read_text());s=json.loads((out/'result.json').read_text())
    assert d['input']==s['input'] and d['hardware']==s['hardware'] and d['hardware']['device_uuid']==uuid
    uncontended=not(first or second)
    if not uncontended:
        s['quality']['speedup_vs_vggt']=None
        s['quality']['speedup_unavailable']='Foreign GPU processes observed in paired run; exclude as uncontended speed comparison'
        save(out/'result.json',s);write_quality(out,s['quality'])
    save(out/'paired_timing.json',dict(device_uuid=uuid,paired=True,uncontended=uncontended,
        dense_seconds=d['timing']['median_seconds'],method_seconds=s['timing']['median_seconds'],
        speedup=d['timing']['median_seconds']/s['timing']['median_seconds'] if uncontended else None,
        foreign_gpu_processes={'dense':first,'joint':second},dense_quality='N/A: timing-only; historical matched quality retained separately'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    try:main(a)
    except Exception as e:
        a.output_dir.mkdir(parents=True,exist_ok=True)
        if not (a.output_dir/'result.json').exists():write_quality(a.output_dir,quality_record({},str(e)))
        save(a.output_dir/'pair_failure.json',dict(reason=str(e)));raise
