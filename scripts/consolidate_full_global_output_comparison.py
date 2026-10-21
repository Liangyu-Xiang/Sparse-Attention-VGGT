"""Build an auditable report view, preserving failed attempts and retry evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_full_global_output_comparison import METHODS,save
from scripts.run_full_global_output_comparison import report


def consolidate(original,retry,destination):
    original=original.resolve();retry=retry.resolve();destination=destination.resolve()
    if destination.exists():raise ValueError('Fresh report destination required')
    manifest=json.loads((original/'manifest.json').read_text())
    selections={}
    for dataset,scenes in manifest['scenes'].items():
        for scene in scenes:
            old=original/dataset/scene
            replacement=retry/dataset/scene
            chosen=replacement if replacement.exists() else old
            summary=json.loads((chosen/'comparison.json').read_text())
            if not summary['outputs_complete'] or not summary['quality_complete']:
                raise ValueError(f'Incomplete paired outputs/quality: {chosen}')
            hashes=json.loads((chosen/'source_sha256.json').read_text())
            if any(manifest['source_sha256'][k]!=v for k,v in hashes.items()):
                raise ValueError(f'Source changed: {chosen}')
            for method in METHODS:
                current=json.loads((chosen/method/'input.json').read_text())
                old_input=old/method/'input.json'
                # A failed initialization can precede writing sparse input.json.
                # Its completed Dense input is the canonical paired frame list.
                if not old_input.exists():old_input=old/'dense/input.json'
                previous=json.loads(old_input.read_text())
                if current!=previous:raise ValueError(f'Input mismatch: {dataset}/{scene}/{method}')
                layers=summary['methods'][method]['layers']
                if len(layers)!=24 or any(len(l['heads'])!=16 for l in layers):
                    raise ValueError(f'Incomplete captures: {chosen}/{method}')
            selections[f'{dataset}/{scene}']=dict(source=str(chosen),repeated=chosen!=old,
                original_attempt=str(old),original_status=json.loads((old/'status.json').read_text()))
    destination.mkdir(parents=True)
    save(destination/'manifest.json',manifest)
    save(destination/'provenance.json',dict(original_campaign=str(original),retry_root=str(retry),
        explanation='Fresh-process retries with unchanged source, method and exact inputs. Original failures retained.',
        scenes=selections))
    for name,entry in selections.items():
        link=destination/name;link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(entry['source'],target_is_directory=True)
    report(destination)
    datasets=json.loads((destination/'dataset_summary.json').read_text())['datasets']
    head_rows=[dict(dataset=dataset,**row) for dataset,d in datasets.items() for row in d['head_macro']]
    with (destination/'head_macro.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(head_rows[0]))
        writer.writeheader();writer.writerows(head_rows)
    segments=[]
    for dataset,d in datasets.items():
        for method in ('saf3r','history80'):
            for comparison in ('rollout_vs_dense','same_input_vs_dense'):
                for first,last in ((0,9),(10,16),(17,23)):
                    rows=[r for r in d['layer_macro'] if r['method']==method and
                        r['position']=='module_output' and r['scope']=='all_queries' and
                        r['comparison']==comparison and first<=r['layer']<=last]
                    segments.append(dict(dataset=dataset,method=method,comparison=comparison,
                        first_layer=first,last_layer=last,
                        mean_layer_relative_l2=sum(r['relative_l2_frobenius'] for r in rows)/len(rows),
                        mean_layer_query_cosine=sum(r['mean_per_query_cosine'] for r in rows)/len(rows)))
    save(destination/'segment_summary.json',dict(definition='Equal-layer mean of equal-scene metrics; not a pooled layer error',rows=segments))
    with (destination/'dataset_report.md').open('a') as stream:
        stream.write('\n## Retry provenance\n\nOriginal attempts and failures are preserved. '
            'This consolidated view selects complete unchanged-protocol repeats where needed.\n\n')
        for name,entry in selections.items():
            if entry['repeated']:stream.write(f'- {name}: {entry["source"]}; original status {entry["original_status"]["stage"]}.\n')
    save(destination/'status.json',dict(outputs_complete=True,quality_complete=True,
        method_runs=3*len(selections),retry_scenes=[n for n,e in selections.items() if e['repeated']]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--retry',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();consolidate(args.original,args.retry,args.output_dir)
