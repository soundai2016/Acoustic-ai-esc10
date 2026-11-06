from __future__ import annotations

from pathlib import Path
import csv
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _stem(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix('') if path.suffix else path


def _finalize(fig, path: str | Path, dpi: int = 180) -> None:
    base = _stem(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def _safe_name(text: str) -> str:
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', text)
    return name.strip('._') or 'audio'


def _prediction_arrays(rows: list[dict], names: list[str] | None = None) -> dict:
    if not rows:
        raise ValueError('No prediction rows were provided')
    if names is None:
        names = [key[5:] for key in rows[0] if key.startswith('prob_')]
    confidence = np.asarray([float(r['confidence']) for r in rows], dtype=float)
    correct = np.asarray([int(r['correct']) for r in rows], dtype=int)
    truth = np.asarray([r['truth'] for r in rows], dtype=object)
    pred = np.asarray([r['prediction'] for r in rows], dtype=object)
    probs = np.asarray([[float(r[f'prob_{name}']) for name in names] for r in rows], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    fold = np.asarray([int(r['fold']) for r in rows], dtype=int)
    return {
        'names': list(names),
        'confidence': confidence,
        'correct': correct,
        'truth': truth,
        'pred': pred,
        'probs': probs,
        'entropy': entropy,
        'fold': fold,
    }


def save_learning_curve(path: str | Path, history: list[dict], title: str = 'Training curves') -> None:
    epochs = [r['epoch'] for r in history]
    train = [r['train_loss'] for r in history]
    val = [r['val_loss'] for r in history]
    val_acc = [100.0 * r['val_accuracy'] for r in history]
    lr = [r['lr'] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8))

    ax = axes[0, 0]
    ax.plot(epochs, train, label='train', linewidth=2)
    ax.plot(epochs, val, label='val', linewidth=2)
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-entropy')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(epochs, val_acc, linewidth=2)
    ax.set_title('Validation accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    gap = np.asarray(val) - np.asarray(train)
    ax.plot(epochs, gap, linewidth=2)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_title('Generalization gap')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val - train loss')
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.plot(epochs, lr, linewidth=2)
    ax.set_title('Learning rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_multi_fold_curves(path: str | Path, histories: dict[str, list[dict]], title: str = 'Training curves by fold') -> None:
    keys = sorted(histories)
    n = len(keys)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, max(3.5, 2.8 * nrows)), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        history = histories[key]
        epochs = [r['epoch'] for r in history]
        train = [r['train_loss'] for r in history]
        val = [r['val_loss'] for r in history]
        acc = [100.0 * r['val_accuracy'] for r in history]
        ax.plot(epochs, train, linewidth=1.8, label='train loss')
        ax.plot(epochs, val, linewidth=1.8, label='val loss')
        ax2 = ax.twinx()
        ax2.plot(epochs, acc, linewidth=1.6, linestyle='--', label='val acc')
        ax.set_title(key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax2.set_ylabel('Val acc (%)')
        ax.grid(alpha=0.2)
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, frameon=False, fontsize=7, loc='best')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_fold_overlay(path: str | Path, histories: dict[str, list[dict]], title: str = 'Fold overlay') -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for key in sorted(histories):
        history = histories[key]
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['train_loss'] for r in history], linewidth=1.8, label=f'{key} train')
        axes[0].plot(epochs, [r['val_loss'] for r in history], linewidth=1.8, linestyle='--', label=f'{key} val')
        axes[1].plot(epochs, [100.0 * r['val_accuracy'] for r in history], linewidth=2.0, label=key)
        axes[2].plot(epochs, [r['lr'] for r in history], linewidth=1.8, label=key)

    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-entropy')
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)

    axes[1].set_title('Validation accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percent')
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7)

    axes[2].set_title('Learning rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_confusion(path: str | Path, matrix: np.ndarray, names: list[str], title: str = 'Confusion matrix') -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Clips')
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    _finalize(fig, path)


def save_confusion_normalized(path: str | Path, matrix: np.ndarray, names: list[str], title: str = 'Normalized confusion') -> None:
    matrix = np.asarray(matrix, dtype=float)
    denom = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0) * 100.0
    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    im = ax.imshow(normalized, cmap='Blues', vmin=0.0, vmax=100.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized (%)')
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    _finalize(fig, path)


def save_fold_accuracy(path: str | Path, rows: list[dict], title: str = 'Accuracy by held-out fold') -> None:
    folds = [r['test_fold'] for r in rows]
    values = [100.0 * r['accuracy'] for r in rows]
    macro = [100.0 * r['macro_f1'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    bars = axes[0].bar(folds, values, width=0.6)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Test fold')
    axes[0].set_ylabel('Percent')
    axes[0].set_ylim(0, 100)
    axes[0].set_xticks(folds)
    for b, v in zip(bars, values):
        axes[0].text(b.get_x() + b.get_width() / 2, min(v + 1.0, 99.5), f'{v:.1f}', ha='center', fontsize=8)
    axes[0].grid(axis='y', alpha=0.2)

    axes[1].plot(folds, macro, marker='o', linewidth=2)
    axes[1].set_title('Macro-F1')
    axes[1].set_xlabel('Test fold')
    axes[1].set_ylabel('Percent')
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(folds)
    axes[1].grid(alpha=0.2)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_fold_diagnostics(path: str | Path, rows: list[dict], title: str = 'Fold diagnostics') -> None:
    folds = np.asarray([r['test_fold'] for r in rows], dtype=int)
    best_epoch = np.asarray([r.get('best_epoch', np.nan) for r in rows], dtype=float)
    epochs_run = np.asarray([r.get('epochs_run', np.nan) for r in rows], dtype=float)
    best_val = np.asarray([r.get('best_val_loss', np.nan) for r in rows], dtype=float)
    test_loss = np.asarray([r.get('test_loss', np.nan) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.1))
    axes[0, 0].plot(folds, best_epoch, marker='o', linewidth=2)
    axes[0, 0].set_title('Selected epoch')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Epoch')
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].bar(folds, epochs_run, width=0.6)
    axes[0, 1].set_title('Epochs run')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Epoch')
    axes[0, 1].grid(axis='y', alpha=0.2)

    axes[1, 0].plot(folds, best_val, marker='s', linewidth=2)
    axes[1, 0].set_title('Best validation loss')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(folds, test_loss, marker='^', linewidth=2)
    axes[1, 1].set_title('Held-out test loss')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_class_metrics(path: str | Path, rows: list[dict], title: str = 'Per-class metrics') -> None:
    names = [r['class'] for r in rows]
    precision = np.array([r['precision'] for r in rows]) * 100.0
    recall = np.array([r['recall'] for r in rows]) * 100.0
    f1 = np.array([r['f1'] for r in rows]) * 100.0
    support = np.array([r['support'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.35, 1]})
    y = np.arange(len(names))
    h = 0.22
    axes[0].barh(y - h, precision, height=h, label='precision')
    axes[0].barh(y, recall, height=h, label='recall')
    axes[0].barh(y + h, f1, height=h, label='f1')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel('Percent')
    axes[0].set_title('Precision / recall / F1')
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc='lower right')
    axes[0].grid(axis='x', alpha=0.2)

    bars = axes[1].bar(np.arange(len(names)), support, width=0.7)
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Clips')
    axes[1].set_title('Support')
    axes[1].grid(axis='y', alpha=0.2)
    for b, v in zip(bars, support):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.3, str(int(v)), ha='center', fontsize=7)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_class_metric_heatmap(path: str | Path, rows: list[dict], title: str = 'Class metric heatmap') -> None:
    names = [r['class'] for r in rows]
    values = np.array([[r['precision'], r['recall'], r['f1']] for r in rows], dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(values, cmap='viridis', vmin=0.0, vmax=100.0, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Percent')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['precision', 'recall', 'f1'])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title)
    _finalize(fig, path)


def save_summary_dashboard(path: str | Path, rows: list[dict], per_class: list[dict], title: str = 'Training summary') -> None:
    folds = [r['test_fold'] for r in rows]
    acc = np.array([r['accuracy'] for r in rows]) * 100.0
    f1 = np.array([r['macro_f1'] for r in rows]) * 100.0
    epochs = np.array([r.get('best_epoch', np.nan) for r in rows])
    class_names = [r['class'] for r in per_class]
    class_f1 = np.array([r['f1'] for r in per_class]) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4))
    axes[0, 0].bar(folds, acc, width=0.6)
    axes[0, 0].set_title('Accuracy by fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Percent')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(axis='y', alpha=0.2)

    axes[0, 1].plot(folds, f1, marker='o', linewidth=2)
    axes[0, 1].set_title('Macro-F1 by fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Percent')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].bar(np.arange(len(class_names)), class_f1, width=0.7)
    axes[1, 0].set_xticks(np.arange(len(class_names)))
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_title('Pooled class F1')
    axes[1, 0].set_ylabel('Percent')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(axis='y', alpha=0.2)

    axes[1, 1].plot(folds, epochs, marker='s', linewidth=2)
    axes[1, 1].set_title('Selected epoch')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Epoch')
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_confidence_histogram(path: str | Path, rows: list[dict], title: str = 'Prediction confidence') -> None:
    arr = _prediction_arrays(rows)
    conf = arr['confidence'] * 100.0
    correct = arr['correct'].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    bins = np.linspace(0, 100, 21)
    axes[0].hist(conf[correct], bins=bins, alpha=0.8, label='correct')
    axes[0].hist(conf[~correct], bins=bins, alpha=0.8, label='wrong')
    axes[0].set_title('Confidence distribution')
    axes[0].set_xlabel('Top-1 confidence (%)')
    axes[0].set_ylabel('Clips')
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    acc_bins = []
    centers = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 100 else conf <= hi)
        if mask.any():
            acc_bins.append(100.0 * correct[mask].mean())
            centers.append((lo + hi) / 2)
    axes[1].plot(centers, acc_bins, marker='o', linewidth=2)
    axes[1].plot([0, 100], [0, 100], linestyle='--', linewidth=1)
    axes[1].set_title('Accuracy vs confidence bin')
    axes[1].set_xlabel('Confidence bin center (%)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_reliability_diagram(path: str | Path, rows: list[dict], title: str = 'Reliability diagram', bins: int = 10) -> None:
    arr = _prediction_arrays(rows)
    conf = arr['confidence']
    acc = arr['correct'].astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_acc = np.zeros(bins, dtype=float)
    bin_conf = np.zeros(bins, dtype=float)
    bin_count = np.zeros(bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if mask.any():
            bin_acc[i] = acc[mask].mean()
            bin_conf[i] = conf[mask].mean()
            bin_count[i] = mask.sum()
    ece = float(np.sum(np.abs(bin_acc - bin_conf) * (bin_count / max(1, bin_count.sum()))))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), gridspec_kw={'width_ratios': [1.1, 1.0]})
    axes[0].bar(centers * 100.0, bin_acc * 100.0, width=(100.0 / bins) * 0.9, label='accuracy')
    axes[0].plot([0, 100], [0, 100], linestyle='--', linewidth=1.2, label='perfect')
    axes[0].plot(centers * 100.0, bin_conf * 100.0, marker='o', linewidth=1.8, label='mean confidence')
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel('Confidence (%)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'Reliability (ECE={ece:.3f})')
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(centers * 100.0, bin_count, width=(100.0 / bins) * 0.9)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_ylabel('Clips')
    axes[1].set_title('Samples per bin')
    axes[1].grid(axis='y', alpha=0.2)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_entropy_by_class(path: str | Path, rows: list[dict], title: str = 'Prediction entropy by true class') -> None:
    arr = _prediction_arrays(rows)
    names = []
    values = []
    for name in dict.fromkeys(arr['truth'].tolist()):
        mask = arr['truth'] == name
        if mask.any():
            names.append(name)
            values.append(arr['entropy'][mask])
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.boxplot(values, tick_labels=names, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel('True class')
    ax.set_ylabel('Entropy')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    ax.grid(axis='y', alpha=0.2)
    _finalize(fig, path)


def save_error_pairs(path: str | Path, rows: list[dict], title: str = 'Most frequent confusions', top_n: int = 15) -> None:
    arr = _prediction_arrays(rows)
    wrong = arr['truth'] != arr['pred']
    pairs = {}
    for t, p in zip(arr['truth'][wrong], arr['pred'][wrong]):
        pairs[(str(t), str(p))] = pairs.get((str(t), str(p)), 0) + 1
    items = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [f'{t} → {p}' for (t, p), _ in items] if items else ['none']
    values = [v for _, v in items] if items else [0]
    fig, ax = plt.subplots(figsize=(10.8, max(3.6, 0.35 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Clips')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.2)
    _finalize(fig, path)


def save_fold_prediction_summary(path: str | Path, rows: list[dict], title: str = 'Held-out prediction summary by fold') -> None:
    arr = _prediction_arrays(rows)
    folds = np.sort(np.unique(arr['fold']))
    accuracy = []
    mean_conf = []
    wrong_conf = []
    for fold in folds:
        mask = arr['fold'] == fold
        accuracy.append(100.0 * arr['correct'][mask].mean())
        mean_conf.append(100.0 * arr['confidence'][mask].mean())
        wrong = mask & (arr['correct'] == 0)
        wrong_conf.append(100.0 * arr['confidence'][wrong].mean() if wrong.any() else 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    axes[0].bar(folds, accuracy, width=0.6)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Percent')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.2)

    axes[1].plot(folds, mean_conf, marker='o', linewidth=2)
    axes[1].set_title('Mean top-1 confidence')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Percent')
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)

    axes[2].plot(folds, wrong_conf, marker='s', linewidth=2)
    axes[2].set_title('Mean wrong confidence')
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('Percent')
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.2)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def save_inference_summary(path: str | Path, entry: dict, title: str | None = None) -> None:
    top = entry['top_k']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={'width_ratios': [1.1, 1.1]})
    labels = [t['class'] for t in top][::-1]
    values = [100.0 * t['probability'] for t in top][::-1]
    axes[0].barh(labels, values)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel('Probability (%)')
    axes[0].set_title('Top predictions')
    axes[0].grid(axis='x', alpha=0.2)

    names = list(entry['probabilities'].keys())
    probs = np.array(list(entry['probabilities'].values())) * 100.0
    order = np.argsort(probs)[::-1]
    show = order[:min(10, len(order))]
    axes[1].bar(np.arange(len(show)), probs[show], width=0.7)
    axes[1].set_xticks(np.arange(len(show)))
    axes[1].set_xticklabels([names[i] for i in show], rotation=45, ha='right', fontsize=8)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Probability (%)')
    axes[1].set_title('Class distribution')
    axes[1].grid(axis='y', alpha=0.2)

    fig.suptitle(title or f'Inference summary | {Path(entry["audio"]).name}', fontsize=13)
    _finalize(fig, path)


def save_window_timeline(path: str | Path, entry: dict, title: str | None = None) -> None:
    windows = entry.get('windows')
    if not windows:
        return
    classes = list(windows[0]['probabilities'].keys())
    mat = np.array([[w['probabilities'][c] for c in classes] for w in windows], dtype=float).T * 100.0
    starts = [w['start_s'] for w in windows]
    ends = [w['end_s'] for w in windows]
    preds = [w['predicted_class'] for w in windows]
    conf = [100.0 * w['confidence'] for w in windows]
    labels = [f'{s:.1f}-{e:.1f}' for s, e in zip(starts, ends)]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.0), gridspec_kw={'height_ratios': [2.0, 1.0]})
    im = axes[0].imshow(mat, aspect='auto', cmap='magma')
    axes[0].set_yticks(np.arange(len(classes)))
    axes[0].set_yticklabels(classes, fontsize=8)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_xlabel('Window (s)')
    axes[0].set_title('Per-window class probabilities')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Probability (%)')

    axes[1].plot(np.arange(len(conf)), conf, marker='o', linewidth=2)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(preds, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Confidence (%)')
    axes[1].set_xlabel('Predicted class by window')
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)
    fig.suptitle(title or f'Window timeline | {Path(entry["audio"]).name}', fontsize=13)
    _finalize(fig, path)


def save_batch_overview(path: str | Path, entries: list[dict], title: str = 'Batch inference overview') -> None:
    names = [Path(e['audio']).name for e in entries]
    pred = [e['predicted_class'] for e in entries]
    conf = np.array([e['confidence'] for e in entries]) * 100.0
    top1 = np.argsort(conf)[::-1][:min(20, len(conf))]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={'width_ratios': [1.25, 1.0]})
    axes[0].bar(np.arange(len(top1)), conf[top1], width=0.7)
    axes[0].set_xticks(np.arange(len(top1)))
    axes[0].set_xticklabels([names[i] for i in top1], rotation=55, ha='right', fontsize=8)
    axes[0].set_ylabel('Top-1 confidence (%)')
    axes[0].set_title('Most confident files')
    axes[0].grid(axis='y', alpha=0.2)

    unique, counts = np.unique(pred, return_counts=True)
    order = np.argsort(counts)[::-1]
    axes[1].barh(np.array(unique)[order], counts[order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Files')
    axes[1].set_title('Predicted class counts')
    axes[1].grid(axis='x', alpha=0.2)
    fig.suptitle(title, fontsize=13)
    _finalize(fig, path)


def load_prediction_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))
