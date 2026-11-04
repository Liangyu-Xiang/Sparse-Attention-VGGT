"""CPU-only GPU lock/monitor: exactly one CUDA application during timing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--monitor', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == '--' else a.command
    if not command:
        raise ValueError('Missing command')
    uuid = subprocess.check_output([
        'nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).strip()

    def applications():
        raw = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True)
        found = set()
        for line in raw.splitlines():
            cols = line.split(',')
            if len(cols) != 2 or cols[0].strip() != uuid:
                continue
            pid = int(cols[1])
            try:
                if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                    found.add(pid)
            except psutil.NoSuchProcess:
                pass
        return found

    # Same lock namespace as the prior lease wrapper, without importing Torch
    # or libcuda or creating any auxiliary CUDA context.
    with open(f'/tmp/vggt_gpu_{a.gpu}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while applications():
            time.sleep(.5)
        time.sleep(.5)
        while applications():
            time.sleep(.5)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=str(a.gpu), OPENBLAS_NUM_THREADS='2',
                   OMP_NUM_THREADS='2')
        a.log.parent.mkdir(parents=True, exist_ok=True)
        foreign = set()
        multiple = set()
        observed = set()
        aborted = False
        max_applications = 0
        polls = 0
        with a.log.open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            while child.poll() is None:
                active = applications()
                polls += 1
                observed |= active
                max_applications = max(max_applications, len(active))
                allowed={child.pid}
                try:allowed|={p.pid for p in psutil.Process(child.pid).children(recursive=True)}
                except psutil.NoSuchProcess:pass
                foreign |= active - allowed
                if len(active) > 1:
                    multiple |= active
                if foreign or multiple:
                    aborted = True
                    try:
                        for c in reversed(psutil.Process(child.pid).children(recursive=True)):
                            c.terminate()
                    except psutil.NoSuchProcess:
                        pass
                    child.terminate()
                    break
                time.sleep(.5)
            code = child.wait(timeout=30)
        # No new process on this GPU until the previous application's context
        # is gone, including its CPU-only quality-evaluation tail.
        while child.pid in applications():
            time.sleep(.5)
        write(a.monitor, dict(
            gpu=a.gpu, gpu_uuid=uuid, child_pid=child.pid,
            monitor_pid=os.getpid(), auxiliary_cuda_context=False,
            foreign_gpu_pids=sorted(foreign), observed_gpu_pids=sorted(observed),
            simultaneous_application_pids=sorted(multiple),
            max_simultaneous_gpu_applications=max_applications,
            same_gpu_overlap_detected=bool(foreign or multiple),
            poll_seconds=.5, polls=polls, returncode=code, aborted=aborted,
            waits_for_full_process_exit=True,
            timing_protocol='One CUDA application per physical GPU; CPU-only lock/monitor'))
    raise SystemExit(code if not aborted else (code or 1))


if __name__ == '__main__':
    main()
