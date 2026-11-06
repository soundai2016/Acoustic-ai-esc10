#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from esc10.core import (
    choose_device,
    classification_metrics,
    evaluate_model,
    load_checkpoint,
    make_loader,
    make_prediction_rows,
    write_json,
    write_predictions,
)
from esc10.data import load_dataset
from esc10.figures import save_class_metrics, save_confusion


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a frozen checkpoint on its official test fold')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/ESC-50')
    parser.add_argument('--output', default='evaluation')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.batch < 1:
        parser.error('batch must be positive')

    device = choose_device(args.device)
    model, classes, config, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint['dataset_kind'] != 'ESC-10_REAL':
        raise ValueError('evaluate.py is for real ESC-10 checkpoints only')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)

    data = load_dataset(args.data, out / 'cache', config)
    if data.classes != classes:
        raise ValueError('Checkpoint class order does not match dataset class order')

    fold = int(checkpoint['test_fold'])
    mask = data.folds == fold
    loader = make_loader(data.features, data.targets, mask, args.batch, False)
    measured = evaluate_model(model, loader, device)
    metrics = classification_metrics(measured['true'], measured['pred'], classes)

    payload = {
        **metrics,
        'test_loss': measured['loss'],
        'test_fold': fold,
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': checkpoint['best_epoch'],
        'dataset_kind': data.kind,
    }
    write_json(out / 'metrics.json', payload)
    save_confusion(fig_dir / 'confusion', np.asarray(metrics['confusion_matrix']), classes, f'Frozen checkpoint | Test fold {fold}')
    save_class_metrics(fig_dir / 'class_metrics', payload['per_class'], 'Per-class metrics')

    indices = np.flatnonzero(mask)
    names = [data.records[i].path.name for i in indices]
    rows = make_prediction_rows(measured, classes, names, fold)
    write_predictions(out / 'predictions.csv', classes, rows)
    print(f'Saved evaluation to {out}')


if __name__ == '__main__':
    main()
