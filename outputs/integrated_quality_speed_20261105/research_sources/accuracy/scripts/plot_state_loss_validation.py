"""Standalone scientific figures for the measured 24-layer diagnostics."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with (a.output/'layer_similarities.csv').open() as f:similarities=list(csv.DictReader(f))
 with (a.output/'layer_states.csv').open() as f:states=list(csv.DictReader(f))
 names=[('原方案','Original Shared Value'),('上一修复版','Previous repair'),('A_history_self_residual_rep','A: self History + residual Rep'),('B_history_group','B: History represents cluster')]
 fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
 for col,comparison in enumerate(['local_same_QKV','independent_Dense_trajectory']):
  for name,label in names:
   rows=sorted([r for r in similarities if r['scene']=='kicker' and r['method']==name and r['comparison']==comparison and r['subset']=='probe_non_camera'],key=lambda r:int(r['layer']))
   if not rows:continue
   x=[int(r['layer']) for r in rows]
   axes[0,col].plot(x,[100*float(r['relative_l2']) for r in rows],label=label)
   axes[1,col].plot(x,[float(r['cosine']) for r in rows],label=label)
  axes[0,col].set_title('Same sparse QKV / local error' if col==0 else 'Independent Dense trajectory / accumulated error')
  for row in [0,1]:
   axes[row,col].axvspan(9.5,16.5,color='grey',alpha=.12);axes[row,col].grid(alpha=.2);axes[row,col].set_xlabel('Global attention layer (zero based)')
  axes[0,col].set_ylabel('Relative L2 (%)');axes[1,col].set_ylabel('Cosine similarity')
 axes[0,0].legend(fontsize=8)
 for col,(name,label) in enumerate(names[2:]):
  rows=sorted([r for r in states if r['scene']=='kicker' and r['method']==name],key=lambda r:int(r['layer']))
  x=[int(r['layer']) for r in rows]
  axes[2,col].stackplot(x,*[[100*float(r[f'state_{s}']) for r in rows] for s in [0,1,5,10]],labels=['Drop','Rep','History','Exact'],colors=['#c9c9c9','#4393c3','#f4a582','#92c5a3'])
  axes[2,col].plot(x,[100*float(r['history_key_reuse']) if r['history_key_reuse'] else 0 for r in rows],color='black',linestyle='--',label='Actual old-Key reuse (%)')
  axes[2,col].set_title(label);axes[2,col].set_ylim(0,100);axes[2,col].set_xlabel('Global attention layer (zero based)');axes[2,col].set_ylabel('Cluster state / old-Key reuse (%)')
 axes[2,1].legend(fontsize=8,loc='lower right')
 fig.suptitle('Kicker, 31 matched frames: 31 camera + 60 regular queries\nUpper panels: regular queries; 20% total Key budget includes Specials',fontsize=13)
 for suffix in ['png','svg']:fig.savefig(a.output/f'layer_comparison.{suffix}',dpi=180)
 svg=a.output/'layer_comparison.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)
if __name__=='__main__':main()
