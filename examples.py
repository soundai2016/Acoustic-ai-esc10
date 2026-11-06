#!/usr/bin/env python3
"""Prepare held-out ESC-10 clips and check an existing checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from esc10.core import choose_device, classification_metrics, load_checkpoint, write_json
from esc10.data import read_records
from esc10.figures import save_confusion
from predict import predict_files

DEFAULT_CLASSES = ('dog', 'rain', 'sea_waves', 'clock_tick', 'chainsaw', 'helicopter')
FIELDNAMES = ['filename', 'category', 'fold', 'src_file', 'sha256', 'license', 'attribution']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_credits(root: Path) -> dict[str, str]:
    license_file = root / 'LICENSE'
    if not license_file.is_file():
        return {}
    lines = license_file.read_text(encoding='utf-8', errors='replace').splitlines()
    credits = {}
    for line in lines:
        match = re.match(r'^\s*[-*]\s*\[([^\]]+\.ogg)\]:\s*(.+)$', line)
        if match:
            credits[Path(match[1]).stem] = match[2].strip()
    return credits


def prepare(root: Path, output: Path, fold: int, classes: list[str]) -> list[dict]:
    records, metadata = read_records(root)
    esc_root = metadata.parent.parent
    credits = read_credits(esc_root)
    available = {record.category for record in records}
    missing = sorted(set(classes) - available)
    if missing:
        raise ValueError(f'Unknown ESC-10 classes: {missing}. Available: {sorted(available)}')
    if len(classes) != len(set(classes)):
        raise ValueError('Duplicate class names')

    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        candidates = sorted(
            (r for r in records if r.fold == fold and r.category == cls),
            key=lambda r: r.path.name,
        )
        if not candidates:
            raise ValueError(f'No clip for fold={fold}, class={cls}')
        source = candidates[0]
        dest = output / source.path.name
        if dest.exists():
            if sha256(dest) != sha256(source.path):
                raise FileExistsError(f'Existing file does not match the official source: {dest}')
        else:
            shutil.copy2(source.path, dest)
        info = sf.info(str(dest))
        if not 4.9 <= info.duration <= 5.1:
            raise ValueError(f'Unexpected audio duration: {dest} ({info.duration:.3f}s)')
        credit_key = source.path.stem.rsplit('-', 1)[0]
        credit = credits.get(credit_key, 'See ESC-50 LICENSE for original Freesound attribution')
        rows.append({
            'filename': dest.name,
            'category': source.category,
            'fold': source.fold,
            'src_file': source.src_file,
            'sha256': sha256(dest),
            'license': 'CC BY 3.0 (ESC-10 subset)',
            'attribution': credit,
        })

    manifest = output / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Prepared {len(rows)} official ESC-10 WAV files from fold {fold}: {output}')
    print(f'Manifest: {manifest}')
    return rows


def verify(manifest: Path, checkpoint: Path, output: Path, device: str) -> dict:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not set(FIELDNAMES).issubset(rows[0]):
        raise ValueError('Invalid example manifest')

    dev = choose_device(device)
    _, names, _, ckpt = load_checkpoint(checkpoint, dev)
    if ckpt['dataset_kind'] != 'ESC-10_REAL':
        raise ValueError('Official sample checks require a real ESC-10 checkpoint')
    held_out = int(ckpt['test_fold'])
    for row in rows:
        if int(row['fold']) != held_out:
            raise ValueError(f'Example fold {row["fold"]} differs from checkpoint test fold {held_out}')
        if row['category'] not in names:
            raise ValueError(f'Class absent from checkpoint: {row["category"]}')
        if Path(row['filename']).name != row['filename']:
            raise ValueError('Invalid manifest filename')
        audio = manifest.parent / row['filename']
        if not audio.is_file() or sha256(audio) != row['sha256']:
            raise ValueError(f'Missing or modified WAV: {audio}')

    files = [manifest.parent / row['filename'] for row in rows]
    predictions = predict_files([str(checkpoint)], files, dev, top_k=3)
    items = []
    for row, pred in zip(rows, predictions):
        items.append({
            'filename': row['filename'],
            'expected': row['category'],
            'predicted': pred['predicted_class'],
            'match': int(row['category'] == pred['predicted_class']),
            'score': pred['confidence'],
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'example_checks.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)

    targets = [names.index(item['expected']) for item in items]
    preds = [names.index(item['predicted']) for item in items]
    metrics = classification_metrics(targets, preds, names)
    result = {
        'checkpoint': str(checkpoint),
        'checkpoint_test_fold': held_out,
        'manifest': str(manifest),
        'sample_count': len(items),
        'sample_correct': sum(item['match'] for item in items),
        'sample_accuracy': metrics['accuracy'],
        'note': 'A few selected samples are a functional check, not a benchmark. Run evaluate.py on all 80 clips for fold metrics.',
    }
    write_json(output / 'example_checks.json', result)
    save_confusion(output / 'figures' / 'example_confusion', np.asarray(metrics['confusion_matrix']), names, 'Selected held-out examples')
    for item in items:
        print(f'{item["filename"]}: expected={item["expected"]}, predicted={item["predicted"]}, match={bool(item["match"])}')
    print(f'Checked {len(items)} clips; results: {output}')
    return result


def synthesize(output: Path, seed: int = 17) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sr = 8000
    t = np.arange(sr * 5, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)

    tone = 0.45 * np.sin(2 * np.pi * 440 * t)
    clicks = np.zeros_like(t)
    for onset in np.arange(0.35, 5, 0.6):
        i = int(onset * sr)
        n = min(280, len(clicks) - i)
        clicks[i:i+n] += 0.8 * np.exp(-np.arange(n) / 42.0) * np.cos(2 * np.pi * 1500 * np.arange(n) / sr)
    broadband = rng.normal(0, 0.12, len(t))
    broadband = signal.sosfilt(signal.butter(4, [300, 3300], btype='bandpass', fs=sr, output='sos'), broadband)
    broadband *= (0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t))

    for name, wave in [('synthetic_tone.wav', tone), ('synthetic_clicks.wav', clicks), ('synthetic_noise.wav', broadband)]:
        sf.write(output / name, np.clip(wave, -0.99, 0.99), sr, subtype='PCM_16')

    t_long = np.arange(sr * 12, dtype=np.float64) / sr
    amp = 0.12 + 0.32 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t_long))
    wave_long = amp * (np.sin(2 * np.pi * 260 * t_long) + 0.25 * np.sin(2 * np.pi * 600 * t_long))
    sf.write(output / 'synthetic_long.wav', np.clip(wave_long, -0.99, 0.99), sr, subtype='PCM_16')
    print(f'Wrote 4 generated test WAVs to {output}; not environmental sound recordings')


def main() -> None:
    parser = argparse.ArgumentParser(description='Manage labeled ESC-10 examples and synthetic audio checks')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p = subparsers.add_parser('prepare', help='Copy labeled official held-out clips from a local ESC-50 dataset')
    p.add_argument('--data', default='data/ESC-50')
    p.add_argument('--fold', type=int, choices=range(1, 6), default=5)
    p.add_argument('--classes', nargs='+', default=list(DEFAULT_CLASSES))
    p.add_argument('--output', default='examples/esc10')

    p = subparsers.add_parser('verify', help='Check official examples with their held-out checkpoint')
    p.add_argument('--manifest', default='examples/esc10/manifest.csv')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='results_examples')
    p.add_argument('--device', default='auto')

    p = subparsers.add_parser('generate', help='Generate functional WAV checks; no ESC-10 labels')
    p.add_argument('--output', default='examples/synthetic')
    p.add_argument('--seed', type=int, default=17)

    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(Path(args.data), Path(args.output), args.fold, args.classes)
    elif args.command == 'verify':
        verify(Path(args.manifest), Path(args.checkpoint), Path(args.output), args.device)
    else:
        synthesize(Path(args.output), args.seed)


if __name__ == '__main__':
    main()
