#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from esc10.figures import (
    load_prediction_rows,
    save_batch_overview,
    save_class_metric_heatmap,
    save_class_metrics,
    save_confidence_histogram,
    save_confusion,
    save_confusion_normalized,
    save_entropy_by_class,
    save_error_pairs,
    save_fold_accuracy,
    save_fold_diagnostics,
    save_fold_overlay,
    save_fold_prediction_summary,
    save_inference_summary,
    save_learning_curve,
    save_multi_fold_curves,
    save_reliability_diagram,
    save_summary_dashboard,
    save_window_timeline,
)


def load_history(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            'epoch': int(row['epoch']),
            'train_loss': float(row['train_loss']),
            'val_loss': float(row['val_loss']),
            'val_accuracy': float(row['val_accuracy']),
            'lr': float(row['lr']),
        }
        for row in rows
    ]


def load_fold_reports(root: Path) -> list[dict]:
    reports = []
    for fold_dir in sorted(root.glob('fold_*')):
        metrics_path = fold_dir / 'metrics.json'
        if metrics_path.is_file():
            reports.append(json.loads(metrics_path.read_text(encoding='utf-8')))
    if not reports and (root / 'summary.json').is_file():
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
        reports = summary.get('folds', [])
    return sorted(reports, key=lambda x: x.get('test_fold', 0))


def load_histories(root: Path) -> dict[str, list[dict]]:
    histories = {}
    for fold_dir in sorted(root.glob('fold_*')):
        history_path = fold_dir / 'history.csv'
        if history_path.is_file():
            histories[fold_dir.name] = load_history(history_path)
    return histories


def load_names_and_confusion(root: Path, reports: list[dict], summary: dict | None):
    if summary is not None and 'per_class_pooled' in summary:
        names = [row['class'] for row in summary['per_class_pooled']]
    elif reports and 'class_names' in reports[0]:
        names = reports[0]['class_names']
    else:
        pred_path = root / 'predictions_all.csv'
        rows = load_prediction_rows(pred_path)
        names = [key[5:] for key in rows[0] if key.startswith('prob_')]
    if summary is not None and 'pooled_confusion_matrix' in summary:
        cm = np.asarray(summary['pooled_confusion_matrix'])
    else:
        matrices = [np.asarray(r['confusion_matrix']) for r in reports if 'confusion_matrix' in r]
        if not matrices:
            raise FileNotFoundError('No confusion matrices found in summary.json or fold metrics.json')
        cm = np.sum(matrices, axis=0)
    return names, cm


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate expanded figures from training or inference outputs')
    parser.add_argument('--results', help='Training output directory containing summary.json or fold_* directories')
    parser.add_argument('--predictions', help='Prediction JSON from predict.py')
    parser.add_argument('--output', default='figures')
    args = parser.parse_args()

    if not args.results and not args.predictions:
        parser.error('Provide --results and/or --predictions')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.results:
        root = Path(args.results)
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8')) if (root / 'summary.json').is_file() else None
        reports = load_fold_reports(root)
        histories = load_histories(root)
        if not reports:
            raise FileNotFoundError('No fold reports found under the results directory')
        names, cm = load_names_and_confusion(root, reports, summary)
        pooled_classes = summary['per_class_pooled'] if summary is not None and 'per_class_pooled' in summary else None
        if pooled_classes is None:
            pooled_classes = []
            for idx, name in enumerate(names):
                tp = int(cm[idx, idx])
                fp = int(cm[:, idx].sum() - tp)
                fn = int(cm[idx, :].sum() - tp)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                pooled_classes.append({'class': name, 'precision': precision, 'recall': recall, 'f1': f1, 'support': int(cm[idx].sum())})

        pred_path = root / 'predictions_all.csv'
        if pred_path.is_file():
            prediction_rows = load_prediction_rows(pred_path)
        else:
            prediction_rows = []
            for fold_dir in sorted(root.glob('fold_*')):
                fold_pred = fold_dir / 'predictions.csv'
                if fold_pred.is_file():
                    prediction_rows.extend(load_prediction_rows(fold_pred))

        tag = 'SYNTHETIC SMOKE | ' if (summary is not None and summary.get('dataset_kind') != 'ESC-10_REAL') else ''
        save_confusion(out / 'pooled_confusion', cm, names, f'{tag}Pooled held-out confusion')
        save_confusion_normalized(out / 'pooled_confusion_normalized', cm, names, f'{tag}Normalized pooled confusion')
        save_class_metrics(out / 'class_metrics', pooled_classes, f'{tag}Pooled per-class metrics')
        save_class_metric_heatmap(out / 'class_metric_heatmap', pooled_classes, f'{tag}Class metric heatmap')
        save_summary_dashboard(out / 'summary_dashboard', reports, pooled_classes, f'{tag}Training summary')
        save_fold_diagnostics(out / 'fold_diagnostics', reports, f'{tag}Fold diagnostics')
        if len(reports) > 1:
            save_fold_accuracy(out / 'fold_accuracy', reports, f'{tag}Accuracy by fold')
        if histories:
            save_multi_fold_curves(out / 'fold_curves', histories, f'{tag}Training curves by fold')
            save_fold_overlay(out / 'fold_overlay', histories, f'{tag}Cross-fold overlay')
            for key, history in histories.items():
                metrics_path = root / key / 'metrics.json'
                if metrics_path.is_file():
                    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                    title = f'{tag}{key} | test {metrics.get("test_fold")}, val {metrics.get("val_fold")}'
                else:
                    title = f'{tag}{key}'
                save_learning_curve(out / f'{key}_training', history, title)
        if prediction_rows:
            save_confidence_histogram(out / 'confidence_histogram', prediction_rows, f'{tag}Prediction confidence')
            save_reliability_diagram(out / 'reliability', prediction_rows, f'{tag}Reliability diagram')
            save_entropy_by_class(out / 'entropy_by_class', prediction_rows, f'{tag}Prediction entropy by class')
            save_error_pairs(out / 'error_pairs', prediction_rows, f'{tag}Most frequent confusions')
            save_fold_prediction_summary(out / 'fold_prediction_summary', prediction_rows, f'{tag}Held-out prediction summary')

    if args.predictions:
        payload = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
        entries = payload['predictions']
        save_batch_overview(out / 'batch_overview', entries)
        for item in entries:
            stem = Path(item['audio']).stem
            save_inference_summary(out / f'{stem}_summary', item)
            if 'windows' in item:
                save_window_timeline(out / f'{stem}_windows', item)


if __name__ == '__main__':
    main()
