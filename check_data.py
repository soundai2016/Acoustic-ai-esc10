#!/usr/bin/env python3
"""Check ESC-10 official CSV, source leakage and audio integrity before training."""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import soundfile as sf
from esc10.data import read_records
from esc10.core import write_json


def main():
    p=argparse.ArgumentParser(description='Validate official ESC-10 metadata and WAV files without changing them')
    p.add_argument('--data',default='data/ESC-50')
    p.add_argument('--output',default='results/data_report.json')
    p.add_argument('--decode-all',action='store_true',help='Decode and verify finite WAV samples')
    args=p.parse_args()
    records, meta=read_records(args.data)
    duration=[]; sr_count=Counter(); channel_count=Counter(); invalid=[]
    for record in records:
        try:
            info=sf.info(str(record.path))
            if info.frames<=0 or info.samplerate<=0: raise ValueError('Empty/invalid WAV')
            duration.append(info.duration)
            sr_count[str(info.samplerate)]+=1
            channel_count[str(info.channels)]+=1
            if args.decode_all:
                import numpy as np
                y,_=sf.read(str(record.path),dtype='float32',always_2d=True)
                if not np.isfinite(y).all():raise ValueError('Nonfinite WAV samples')
        except (RuntimeError,ValueError,OSError) as exc:
            invalid.append({'file':record.path.name,'reason':str(exc)})
    report={'dataset':'ESC-10 subset of ESC-50','num_files':len(records),'categories':dict(Counter(r.category for r in records)),
            'fold_counts':dict(sorted(Counter(str(r.fold) for r in records).items())),
            'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
            'sample_rates':dict(sr_count),'channel_counts':dict(channel_count),
            'duration_min_s':min(duration) if duration else None,
            'duration_max_s':max(duration) if duration else None,
            'decoded_audio':args.decode_all,'invalid_files':invalid,
            'caution':'Checks physical recording structure and official fold metadata; does not establish that trained model generalizes to new devices.'}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);write_json(dest,report)
    print(f'Checked {len(records)} CSV records; invalid WAVs: {len(invalid)}; report: {dest}')
    if invalid:raise SystemExit('Audio integrity check failed; inspect invalid_files in report')


if __name__=='__main__':main()
