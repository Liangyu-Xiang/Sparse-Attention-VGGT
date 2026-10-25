"""Bounded read-only checks and UUID quarantine; never reset GPUs or drivers."""
import csv
import io
from pathlib import Path
import subprocess
import tempfile
import time

FAULT_UUID='GPU-484e9ea2-131d-bf08-ea41-710743966793'


def active_experiment_workers(root):
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if 'scripts/evaluate_reuse_exploration.py' in cmd and str(root) in cmd:
            found.append(int(path.name))
    return found


def bounded_command(command,timeout=10):
    # subprocess.run(timeout=...) can still wait indefinitely after killing a
    # D-state process. Never use wait()/communicate() after the deadline here.
    with tempfile.TemporaryFile(mode='w+') as out:
        p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+timeout
        while p.poll() is None and time.monotonic()<deadline:time.sleep(.05)
        if p.poll() is None:
            p.kill()
            raise RuntimeError('GPU health query timed out; driver recovery required. No CUDA worker launched.')
        out.seek(0);text=out.read()
        if p.returncode:raise RuntimeError('GPU health query failed: '+text[-1000:])
        return text


def parse_inventory(text):
    devices=[]
    for row in csv.reader(io.StringIO(text)):
        if not row:continue
        if len(row)!=4:raise ValueError('Unexpected GPU inventory row')
        index,uuid,bus,used=[v.strip() for v in row]
        if not uuid.startswith('GPU-'):raise ValueError('Invalid GPU UUID')
        devices.append(dict(gpu=int(index),uuid=uuid,bus=bus,memory_used_mib=int(used)))
    if not devices or len({d['uuid'] for d in devices})!=len(devices):raise ValueError('Invalid/empty GPU inventory')
    return devices


def idle_healthy_devices(devices,quarantine=()):
    excluded={FAULT_UUID.lower(),*(u.lower() for u in quarantine)}
    healthy=[d for d in devices if d['uuid'].lower() not in excluded and d['memory_used_mib']<=256]
    if not healthy:raise RuntimeError('No idle non-quarantined GPU available; existing workloads are not stopped')
    return sorted(healthy,key=lambda d:d['gpu'])


def remap_jobs(jobs,devices):
    by={d['gpu']:d for d in devices}
    load={gpu:sum(j['gpu']==gpu for j in jobs) for gpu in by}
    output=[]
    for j in jobs:
        gpu=j['gpu']
        if gpu not in by:
            gpu=min(load,key=lambda g:(load[g],g));load[gpu]+=1
        output.append(dict(j,original_gpu=j['gpu'],gpu=gpu,device_uuid=by[gpu]['uuid']))
    return output


def restrict_devices(devices,indices=None):
    if indices is None:return devices
    wanted=set(indices)
    if len(wanted)!=len(indices) or not wanted:raise ValueError('Select distinct GPU indices')
    selected=[d for d in devices if d['gpu'] in wanted]
    if {d['gpu'] for d in selected}!=wanted:
        raise RuntimeError('Requested GPU is busy, missing or quarantined; no substitute GPU will be used')
    return selected


def query_inventory(uuids=None):
    command=['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits']
    if uuids:command+=['--id='+','.join(uuids)]
    return parse_inventory(bounded_command(command))
