"""Offline sensing sweep; never confuse dense-trajectory diagnostics with eval."""
import argparse
import csv
import json
from pathlib import Path
import torch
from saf3r.patch.layer_shared_av import branch_statistics, sample_queries


def main(args):
    torch.set_num_threads(4)
    rows = torch.load(args.logits, weights_only=True, map_location='cpu')
    out = []
    # Temperature is an explicit optional extension: the literal chat reference
    # is multiplier=1. A single global multiplier, never per-layer fitting.
    for multiplier in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for fraction in [1., .5, .25, .1]:
            for tau in [None, .8, .9, .95]:
                stats = []
                for layer, logits in enumerate(rows):
                    sample = logits[sample_queries(len(logits), fraction, 'cpu')]
                    r = branch_statistics((sample*multiplier).softmax(-1), tau)
                    stats.append(dict(layer=layer, **r))
                early = sum(r['alpha'] for r in stats[3:10])/7
                middle = sum(r['alpha'] for r in stats[10:17])/7
                late = sum(r['alpha'] for r in stats[17:24])/7
                # Equal weight to the three ranges, excluding undefined early
                # 0..2. No pose labels are used to tune the branch profile.
                error = (early+(1-middle)+late)/3
                out.append(dict(multiplier=multiplier, sampling=.01*fraction, dominance=tau,
                    early_attention=early, middle_attention=middle, late_attention=late,
                    target_mae=error, pure_layers=sum(r['alpha'] in (0., 1.) for r in stats), layers=stats))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'sweep.json').write_text(json.dumps(out, indent=2)+'\n')
    flat = [{k: v for k, v in r.items() if k != 'layers'} for r in out]
    with (args.output_dir/'sweep.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    best = min(out, key=lambda r: (r['target_mae'], r['multiplier'], -r['sampling']))
    print(json.dumps(best, indent=2))
    for multiplier in [1, 4, 16, 64, 256]:
        r = next(r for r in flat if r['multiplier'] == multiplier and r['sampling'] == .01 and r['dominance'] == .9)
        print(r)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logits', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    main(p.parse_args())
