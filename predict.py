#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from esc10.audio import audio_to_features, windowed_features
from esc10.core import choose_device, load_checkpoint
from esc10.figures import save_batch_overview, save_inference_summary, save_window_timeline


def gather_files(paths: list[str], directory: str | None, recursive: bool) -> list[Path]:
    items = [Path(p).expanduser().resolve() for p in paths]
    if directory:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        iterator = root.rglob('*') if recursive else root.glob('*')
        items.extend(sorted(p.resolve() for p in iterator if p.suffix.lower() in {'.wav', '.flac', '.ogg'}))
    unique = []
    seen = set()
    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    if not unique:
        raise ValueError('Provide at least one --audio file or an --input-dir')
    for item in unique:
        if not item.is_file():
            raise FileNotFoundError(item)
    return unique


def predict_files(checkpoints: list[str], audio_files: list[Path], device: torch.device, top_k: int,
                  batch_size: int = 32, window_hop: float | None = None,
                  include_windows: bool = False) -> list[dict]:
    loaded = [load_checkpoint(path, device) for path in checkpoints]
    classes, config = loaded[0][1:3]
    dataset_kind = loaded[0][3]['dataset_kind']
    for _, other_classes, other_cfg, other_ckpt in loaded[1:]:
        if other_classes != classes or other_cfg != config:
            raise ValueError('Cannot ensemble checkpoints with different class orders or feature settings')
        if other_ckpt['dataset_kind'] != dataset_kind:
            raise ValueError('Cannot mix synthetic and real checkpoints')
    if window_hop is not None and not 0 < window_hop <= config.duration_seconds:
        raise ValueError('window_hop must be between 0 and 5 seconds')

    outputs = []
    for path in audio_files:
        if window_hop is None:
            features = audio_to_features(path, config)[None]
            intervals = [(0.0, config.duration_seconds)]
        else:
            features, intervals = windowed_features(path, hop_seconds=window_hop, cfg=config)
        parts = []
        for pos in range(0, len(features), batch_size):
            tensor = torch.from_numpy(features[pos:pos + batch_size, None]).to(device)
            with torch.inference_mode():
                probs = torch.stack([model(tensor).softmax(dim=1) for model, _, _, _ in loaded]).mean(dim=0)
            parts.append(probs.cpu().numpy())
        per_window = np.concatenate(parts, axis=0)
        mean_prob = per_window.mean(axis=0)
        order = np.argsort(mean_prob)[::-1][:min(top_k, len(classes))]
        item = {
            'audio': str(path),
            'predicted_class': classes[int(order[0])],
            'confidence': float(mean_prob[order[0]]),
            'num_windows': int(len(per_window)),
            'top_k': [{'class': classes[int(i)], 'probability': float(mean_prob[i])} for i in order],
            'probabilities': {name: float(value) for name, value in zip(classes, mean_prob)},
        }
        if include_windows:
            item['windows'] = [
                {
                    'start_s': round(t0, 4),
                    'end_s': round(t1, 4),
                    'predicted_class': classes[int(np.argmax(probs))],
                    'confidence': float(np.max(probs)),
                    'probabilities': {name: float(value) for name, value in zip(classes, probs)},
                }
                for (t0, t1), probs in zip(intervals, per_window)
            ]
        outputs.append(item)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Run inference with one or more ESC-10 checkpoints')
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--audio', nargs='*', default=[])
    parser.add_argument('--input-dir')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--window-hop', type=float)
    parser.add_argument('--include-windows', action='store_true')
    parser.add_argument('--output', default='predictions.json')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()

    if args.top_k < 1 or args.batch < 1:
        parser.error('top-k and batch must be positive')
    if args.window_hop is not None and not 0 < args.window_hop <= 5:
        parser.error('--window-hop must be in (0, 5]')
    if args.include_windows and args.window_hop is None:
        parser.error('--include-windows requires --window-hop')

    files = gather_files(args.audio, args.input_dir, args.recursive)
    device = choose_device(args.device)
    predictions = predict_files(args.checkpoint, files, device, args.top_k, args.batch, args.window_hop, args.include_windows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'checkpoints': args.checkpoint,
        'device': str(device),
        'n_audio': len(predictions),
        'window_hop_seconds': args.window_hop,
        'aggregation': 'mean softmax across windows and checkpoints' if args.window_hop is not None else 'mean softmax across checkpoints',
        'predictions': predictions,
    }, indent=2) + '\n', encoding='utf-8')

    csv_path = out.with_suffix('.csv')
    classes = list(predictions[0]['probabilities'])
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, ['audio', 'predicted_class', 'confidence'] + [f'prob_{name}' for name in classes])
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                'audio': row['audio'],
                'predicted_class': row['predicted_class'],
                'confidence': row['confidence'],
                **{f'prob_{name}': prob for name, prob in row['probabilities'].items()},
            })

    if not args.no_figures:
        fig_dir = out.parent / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_batch_overview(fig_dir / 'batch_overview', predictions)
        for item in predictions:
            stem = Path(item['audio']).stem
            save_inference_summary(fig_dir / f'{stem}_summary', item)
            if args.include_windows:
                save_window_timeline(fig_dir / f'{stem}_windows', item)

    for item in predictions[:20]:
        top = ', '.join(f"{row['class']}={row['probability']:.3f}" for row in item['top_k'])
        print(f"{item['audio']} -> {top}")
    if len(predictions) > 20:
        print(f'... plus {len(predictions) - 20} more files')
    print(f'Saved predictions to {out} and {csv_path}')


if __name__ == '__main__':
    main()
