"""Reserve a GPU across import/model-loading gaps; monitor compute overlap."""
import argparse,ctypes,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import psutil


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--single-process',action='store_true',help='Use only a file lock and nvidia-smi monitoring; create no auxiliary CUDA context');p.add_argument('--strict-contexts',action='store_true');p.add_argument('--gpu',type=int,required=True);p.add_argument('--monitor',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:raise ValueError('Missing command')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    bus=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
    def phase(proc,args):
        if '--output-dir' not in args:return None
        q=Path(args[args.index('--output-dir')+1])/'status.json'
        try:return json.loads(q.read_text()).get('stage')
        except (FileNotFoundError,json.JSONDecodeError):return None
    ignored={};seen_cpu=set()
    def active():
        result=set();raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        for line in raw.splitlines():
            cols=line.split(',')
            if cols[0].strip()!=uuid:continue
            pid=int(cols[1]);
            if pid==os.getpid():continue
            try:
                proc=psutil.Process(pid)
                if proc.status()==psutil.STATUS_ZOMBIE:continue
                if a.strict_contexts:
                    result.add(pid);continue
                stage=phase(proc,proc.cmdline())
                if stage in ('quality','complete','quality_incomplete','diagnostic_complete','timing_complete'):
                    seen_cpu.add(pid);ignored[pid]=stage;continue
                if pid in seen_cpu:continue
                result.add(pid)
            except psutil.NoSuchProcess:continue
        return result
    def loading():
        for proc in psutil.process_iter(['pid','cmdline']):
            if proc.pid==os.getpid():continue
            args=proc.info['cmdline'] or []
            if '--output-dir' not in args:continue
            try:
                if proc.environ().get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):continue
                if phase(proc,args) not in ('quality','complete','quality_incomplete','diagnostic_complete','timing_complete'):return True
            except psutil.Error:continue
        return False
    cuda=None;context=ctypes.c_void_p();device=ctypes.c_int()
    if not a.single_process:
        cuda=ctypes.CDLL('libcuda.so.1')
        assert cuda.cuInit(0)==0 and cuda.cuDeviceGetByPCIBusId(ctypes.byref(device),bus.encode())==0
    with open(f'/tmp/vggt_gpu_{a.gpu}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            while active() or loading():time.sleep(.5)
            if cuda is not None:assert cuda.cuCtxCreate_v2(ctypes.byref(context),0,device)==0
            time.sleep(.5)
            if not active() and not loading():break
            if cuda is not None:cuda.cuCtxDestroy_v2(context)
            context=ctypes.c_void_p()
        env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':str(a.gpu),'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
        a.log.parent.mkdir(parents=True,exist_ok=True);foreign=set();aborted=False
        try:
            with a.log.open('w') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    try:own={child.pid}|{p.pid for p in psutil.Process(child.pid).children(recursive=True)}
                    except psutil.NoSuchProcess:own={child.pid}
                    others=active()-own
                    if others:
                        if phase(None,command) in ('quality','complete'):
                            ignored.update({pid:'Joined after own timed GPU phases completed' for pid in others})
                            time.sleep(.5);continue
                        foreign|=others
                        try:
                            for descendant in reversed(psutil.Process(child.pid).children(recursive=True)):
                                try:descendant.terminate()
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        child.terminate();aborted=True;break
                    time.sleep(.5)
                code=child.wait(timeout=30)
            write(a.monitor,dict(gpu=a.gpu,gpu_uuid=uuid,child_pid=child.pid,reservation_pid=os.getpid(),foreign_gpu_pids=sorted(foreign),
                same_gpu_overlap_detected=bool(foreign),strict_contexts=a.strict_contexts,single_process=a.single_process,poll_seconds=.5,returncode=code,aborted=aborted,
                ignored_cpu_quality_contexts=ignored,reservation_runs_kernels=False,reservation_memory_scope='No auxiliary CUDA context' if a.single_process else 'Auxiliary context excluded from algorithm-process Torch counters'))
        finally:
            if cuda is not None:cuda.cuCtxDestroy_v2(context)
    raise SystemExit(code)


if __name__=='__main__':main()
