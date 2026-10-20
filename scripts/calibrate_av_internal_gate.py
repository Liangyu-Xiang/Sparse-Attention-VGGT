"""Unsupervised two-mode calibration on ETH3D internal features; no layer labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def fit(root,output):
    paths=sorted(root.glob('eth3d/*/result.json'))
    if len(paths)<2:raise ValueError('At least two ETH3D sequences required')
    samples=[]
    for path in paths:
        d=json.loads(path.read_text())
        assert d['dataset']=='eth3d' and not d['protocol']['ground_truth_used']
        for r in d['layers'][1:]:
            samples.append(math.log(max(1e-12,math.sqrt(max(0,r['head_parent_kl'])*max(0,r['query_group_js'])))))
    x=np.array(samples);centers=np.array([x.min(),x.max()])
    for _ in range(100):
        labels=np.abs(x[:,None]-centers).argmin(1)
        updated=np.array([x[labels==i].mean() for i in range(2)])
        if np.allclose(updated,centers,atol=1e-10):break
        centers=updated
    centers=np.sort(centers)
    if centers[1]-centers[0]<1e-6:raise ValueError('No separated internal signal modes')
    d=dict(version=1,feature='log_sqrt_head_kl_times_group_js',centers=centers.tolist(),
        midpoint=float(centers.mean()),temperature=float((centers[1]-centers[0])/(2*math.log(9))),
        min_alpha=.2,max_alpha=.8,alpha_step=.05,groups=64,
        calibration=dict(dataset='eth3d',source_root=str(root),ground_truth_used=False,layer_labels_used=False,
            method='1D two-means on nonseed-layer signals, min/max initialization; centers map to posterior .1/.9',
            inputs=[dict(result=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         frames=json.loads(p.read_text())['input']['shape'][0 if len(json.loads(p.read_text())['input']['shape'])==4 else 1]) for p in paths]))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();fit(a.root,a.output)
