"""Evaluate the remote history fix in an isolated checkout on authorized GPUs."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

ROOT = Path(__file__).resolve().parents[1]
SCENES = {'7scenes': ['heads', 'chess', 'stairs', 'fire', 'office', 'pumpkin', 'redkitchen'],
          'eth3d': ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrains']}
VARIANTS = ['single_probe_repaired', 'full_repaired']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for directory in ['saf3r', 'scripts', 'configs', 'evaluation', 'tests']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml', '.toml'}:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(out / 'source_sha256.json', hashes)
    with tarfile.open(out / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for name in hashes:
            archive.add(ROOT / name, arcname=name, recursive=False)
    manifest = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    requested_branch='fix/gpt6-single-probe-history-20261103', variants=VARIANTS,
                    configs={v:json.loads((ROOT / f'configs/evaluation/reuse_light_{v}.json').read_text()) for v in VARIANTS},
                    scenes=SCENES, gpus=[4,5,6,7], quality_tasks=['pose','recon_unposed'], controls_executed=0,
                    quality_retuning=False, quality_timing='One warmup + one measured full forward per scene',
                    benchmark='Fixed heads128, one warmup + three measured forwards, then separate exclusive profile',
                    dense_speedup=None, dense_reason='Unchanged Dense not rerun; no contemporaneous paired timing',
                    implementation_changes='Remote branch model code unchanged; evaluator additionally archives repair source',
                    single_cuda_process=True, auxiliary_cuda_context=False)
    save(out / 'manifest.json', manifest)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    wrapper = ROOT / 'scripts/run_gpu_exclusive.py'

    def launch(gpu, label, command):
        code = subprocess.call([sys.executable, str(wrapper), '--gpu', str(gpu),
                                '--monitor', str(out/'monitors'/f'{label}.json'),
                                '--log', str(out/'logs'/f'{label}.log'), '--', *command],
                               cwd=ROOT, env=environment)
        print(label, code, flush=True)
        return code

    with (out/'cpu_tests.log').open('w') as log:
        cpu_code = subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                   '-p', 'test_lightweight_budget_repair.py'], cwd=ROOT,
                                  env=environment, stdout=log, stderr=subprocess.STDOUT)
    save(out/'cpu_tests.json', dict(returncode=cpu_code, quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},
                                  quality_reason='CPU unit tests only; no model quality evaluation', dense_speedup=None))
    assert cpu_code == 0, 'CPU validation failed'
    code = launch(7, 'numerical_validation', [sys.executable, str(ROOT/'scripts/validate_lightweight_reuse.py'),
                                            '--output-dir', str(out/'numerical_validation')])
    assert code == 0, 'GPU numerical validation failed'
    queue = [(variant, dataset, scene) for variant in VARIANTS for dataset, scenes in SCENES.items() for scene in scenes]
    lock = threading.Lock()
    done = []

    def worker(gpu):
        if gpu == 7:
            for variant in VARIANTS:
                label = f'timing_{variant}'
                code = launch(gpu, label, [sys.executable, str(ROOT/'scripts/benchmark_light_reuse.py'),
                    '--config', str(ROOT/f'configs/evaluation/reuse_light_{variant}.json'),
                    '--input', str(ROOT/'outputs/fastest_single_probe_snapshot_20261103/timing/input.json'),
                    '--output-dir', str(out/'timing'/variant), '--profile'])
                save(out/f'{label}_status.json', dict(returncode=code))
        while True:
            with lock:
                if not queue:
                    return
                variant, dataset, scene = queue.pop(0)
            label = f'{variant}_{dataset}_{scene}'
            command = [sys.executable, str(ROOT/'scripts/evaluate_zdrop_history_states.py'),
                '--dataset', dataset, '--scene', scene, '--method', 'light_reuse', '--frames', '128',
                '--warmup', '1', '--repeats', '1', '--config', str(ROOT/f'configs/evaluation/reuse_light_{variant}.json'),
                '--output-dir', str(out/'quality'/variant/dataset/scene)]
            code = launch(gpu, label, command)
            with lock:
                done.append(dict(variant=variant, dataset=dataset, scene=scene, gpu=gpu, returncode=code))
                save(out/'progress.json', done)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker, [4,5,6,7]))
    unchanged = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    save(out/'completion.json', dict(runs=done, all_processes_succeeded=len(done)==36 and all(x['returncode']==0 for x in done),
                                    source_unchanged=unchanged))
    assert unchanged, 'Frozen source changed during evaluation'


if __name__ == '__main__':
    main()
