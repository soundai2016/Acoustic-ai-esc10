#!/usr/bin/env python3
"""Analyze held-out errors without changing training or model selection."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from esc10.audio import load_wave, log_mel
from esc10.data import read_records


def main():
    p=argparse.ArgumentParser(description='Post-hoc analysis of frozen held-out ESC-10 predictions')
    p.add_argument('--results',required=True,help='Training output containing predictions_all.csv')
    p.add_argument('--data',help='Optional official ESC-50 root, for visualizing real WAV error examples')
    p.add_argument('--output',default='error_analysis')
    p.add_argument('--max-examples',type=int,default=8)
    args=p.parse_args()
    root=Path(args.results);pred=root/'predictions_all.csv'
    if not pred.is_file():raise FileNotFoundError(pred)
    import json
    summary_path=root/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('dataset_kind')!='ESC-10_REAL':
            raise ValueError('This script requires REAL ESC-10 predictions; synthetic smoke results are not a research error analysis')
    else:
        raise ValueError(f'Missing summary.json needed to verify real ESC-10 provenance: {summary_path}')
    with pred.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    required={'filename','fold','truth','prediction','correct','confidence'}
    if not rows or not required.issubset(rows[0]):raise ValueError('Invalid predictions_all.csv format')
    errors=[r for r in rows if r['truth']!=r['prediction']]
    errors.sort(key=lambda r:float(r['confidence']),reverse=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'errors_ranked.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(errors)
    pairs=Counter((r['truth'],r['prediction']) for r in errors)
    with (out/'confusion_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['truth','predicted','count'])
        for (t,p),count in pairs.most_common():w.writerow([t,p,count])
    print(f'Total predictions: {len(rows)}; misclassified: {len(errors)}; output: {out}')
    if args.data and errors and args.max_examples>0:
        records,_=read_records(args.data)
        lookup={r.path.name:r.path for r in records}
        selected=errors[:args.max_examples]
        ncols=2;nrows=(len(selected)+1)//2
        fig,axes=plt.subplots(nrows,ncols,figsize=(14,max(3.4,3.3*nrows)),squeeze=False)
        for ax,r in zip(axes.flat,selected):
            path=lookup.get(r['filename'])
            if not path:raise FileNotFoundError(f'ESC-10 WAV missing: {r["filename"]}')
            mel=log_mel(load_wave(path))
            ax.imshow(mel,origin='lower',aspect='auto',cmap='magma')
            ax.set(title=f'{r["truth"]} -> {r["prediction"]} (score={float(r["confidence"]):.2f})',
                   xlabel='Time bin (resampled to 128)',ylabel='Mel band (40)')
        for ax in list(axes.flat)[len(selected):]:ax.axis('off')
        fig.suptitle('REAL ESC-10 held-out errors — investigate source / propagation / sensor / background',fontsize=14)
        fig.tight_layout(rect=[0,.01,1,.96]);fig.savefig(out/'failure_examples.png',dpi=220);fig.savefig(out/'failure_examples.svg');plt.close(fig)
        print('Created spectrogram panels for real errors; these are visual observations, not established physical causes.')


if __name__=='__main__':main()
