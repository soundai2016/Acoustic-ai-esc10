#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from esc10.audio import DEFAULT_FEATURES
from esc10.core import (
    choose_device, classification_metrics, evaluate_model, load_checkpoint,
    make_loader, make_prediction_rows, save_checkpoint, seed_everything,
    spec_augment, write_json, write_predictions, class_metrics,
)
from esc10.data import get_split, load_dataset, synthetic_smoke_dataset
from esc10.figures import (
    save_class_metric_heatmap,
    save_class_metrics,
    save_confidence_histogram,
    save_confusion,
    save_confusion_normalized,
    save_error_pairs,
    save_fold_accuracy,
    save_fold_diagnostics,
    save_fold_prediction_summary,
    save_learning_curve,
    save_multi_fold_curves,
    save_reliability_diagram,
    save_summary_dashboard,
    save_entropy_by_class,
    save_fold_overlay,
)
from esc10.model import build_model, parameter_count, AudioResNet18


def create_optimizer(model: nn.Module, args):
    if isinstance(model, AudioResNet18):
        backbone = [p for name, p in model.backbone.named_parameters() if not name.startswith('fc.')]
        head = list(model.backbone.fc.parameters())
        return torch.optim.AdamW([
            {'params': backbone, 'lr': args.backbone_lr, 'initial_lr': args.backbone_lr},
            {'params': head, 'lr': args.lr, 'initial_lr': args.lr},
        ], weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_fold(data, fold: int, val_fold: int, args, output: Path, device: torch.device):
    seed_everything(args.seed + fold, deterministic=not args.nondeterministic)
    tr, va, te = get_split(data.folds, fold, val_fold)
    loaders = (
        make_loader(data.features, data.targets, tr, args.batch, True, args.seed + fold),
        make_loader(data.features, data.targets, va, args.batch, False),
        make_loader(data.features, data.targets, te, args.batch, False),
    )
    train_loader, val_loader, test_loader = loaders
    folder = output / f'fold_{fold}'
    figures = folder / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    needs_weights = args.pretrained and not args.smoke and args.model == 'resnet18' and not (args.resume and (folder / 'last.pt').exists())
    model = build_model(args.model, len(data.classes), pretrained=needs_weights).to(device)
    optimizer = create_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    use_amp = device.type == 'cuda' and args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    settings = {
        'architecture': args.model, 'pretrained': bool(args.pretrained and not args.smoke and args.model == 'resnet18'),
        'epochs_max': args.epochs, 'head_epochs': args.head_epochs, 'min_epochs': args.min_epochs,
        'batch': args.batch, 'lr_head': args.lr, 'lr_backbone': args.backbone_lr,
        'weight_decay': args.weight_decay, 'mixup_alpha': args.mixup, 'specaugment': args.augment,
        'label_smoothing': args.label_smoothing, 'patience': args.patience,
        'seed': args.seed + fold, 'min_delta': args.min_delta, 'amp': use_amp,
        'save_every': args.save_every,
    }
    last_path = folder / 'last.pt'
    history = []
    best_loss, best_epoch, stalled = float('inf'), 0, 0
    first_epoch = 1
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=True)
        previous = dict(state['settings'])
        previous['epochs_max'] = args.epochs
        if (previous != settings or int(state['fold']) != fold or int(state['val_fold']) != val_fold
                or args.epochs < int(state['epoch'])):
            raise ValueError('Resume settings or fold IDs do not match the existing run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scheduler.T_max = args.epochs
        if int(state['settings']['epochs_max']) != args.epochs:
            fraction = min(int(state['epoch']) / args.epochs, 1.0)
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group['lr'] = args.min_lr + (base_lr - args.min_lr) * (1 + math.cos(math.pi * fraction)) / 2
            scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
        scaler.load_state_dict(state['scaler'])
        if 'random_state' in state:
            random.setstate(state['random_state'])
            np_state = state['numpy_state']
            np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
            torch.set_rng_state(state['torch_rng'])
            train_loader.generator.set_state(state['loader_rng'])
            if device.type == 'cuda' and 'cuda_rng' in state:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, best_loss, best_epoch, stalled = state['history'], state['best_loss'], state['best_epoch'], state['stalled']
        first_epoch = state['epoch'] + 1
        print(f'Fold {fold}: resumed at epoch {first_epoch}')
    elif args.resume and not last_path.exists():
        print(f'Fold {fold}: no last.pt, starting a new run')

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        if isinstance(model, AudioResNet18):
            model.freeze_backbone(args.pretrained and not args.smoke and epoch <= args.head_epochs)
            if args.pretrained and not args.smoke and epoch > args.head_epochs and args.freeze_bn:
                for layer in model.backbone.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()

        loss_total = 0.0
        count = 0
        train_correct = 0
        for xb, yb in train_loader:
            if args.augment:
                xb = spec_augment(xb, probability=0.7, time_mask=28, freq_mask=10)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            mixed = args.mixup > 0 and len(yb) > 1
            if mixed:
                lam = np.random.beta(args.mixup, args.mixup)
                perm = torch.randperm(len(yb), device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                yb_perm = yb[perm]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=use_amp):
                logits = model(xb)
                if mixed:
                    loss = lam * nn.functional.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
                    loss += (1.0-lam) * nn.functional.cross_entropy(logits, yb_perm, label_smoothing=args.label_smoothing)
                else:
                    loss = nn.functional.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach()) * len(yb)
            count += len(yb)
            if not mixed:
                train_correct += int((logits.argmax(dim=1) == yb).sum())

        validation = evaluate_model(model, val_loader, device)
        scheduler.step()
        history.append({
            'epoch': epoch, 'train_loss': loss_total / count,
            'val_loss': validation['loss'], 'val_accuracy': validation['accuracy'],
            'lr': optimizer.param_groups[-1]['lr'],
        })
        improved = validation['loss'] < best_loss - args.min_delta or best_epoch == 0
        if improved:
            best_loss, best_epoch, stalled = validation['loss'], epoch, 0
            save_checkpoint(folder / 'best.pt', model, data.classes, epoch, best_loss, val_fold, fold,
                            data.kind, settings, architecture=args.model)
        else:
            stalled += 1

        stopping = epoch >= args.min_epochs and stalled >= args.patience
        if epoch % args.save_every == 0 or epoch == args.epochs or stopping:
            np_state = np.random.get_state()
            torch.save({
                'random_state': random.getstate(),
                'numpy_state': (np_state[0], np_state[1].tolist(), *np_state[2:]),
                'torch_rng': torch.get_rng_state(),
                'loader_rng': train_loader.generator.get_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch, 'fold': fold, 'val_fold': val_fold,
                'history': history, 'best_loss': best_loss,
                'best_epoch': best_epoch, 'stalled': stalled, 'settings': settings,
            }, last_path)
        if epoch == first_epoch or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'fold={fold} epoch={epoch:03d} train_loss={loss_total/count:.4f} '
                  f'val_loss={validation["loss"]:.4f} val_acc={validation["accuracy"]:.3f} '
                  f'best={best_epoch} head_lr={optimizer.param_groups[-1]["lr"]:.2e}', flush=True)
        if stopping:
            print(f'fold={fold} early stop at epoch={epoch}; selected={best_epoch}')
            break

    if best_epoch == 0:
        raise RuntimeError('No best checkpoint saved; check the training configuration')
    best, _, _, _ = load_checkpoint(folder / 'best.pt', device)
    test = evaluate_model(best, test_loader, device)
    stats = classification_metrics(test['true'], test['pred'], data.classes)
    idx = np.flatnonzero(te)
    filenames = [data.records[i].path.name for i in idx] if data.records else [f'smoke_{i}.wav' for i in idx]
    pred_rows = make_prediction_rows(test, data.classes, filenames, fold)
    write_predictions(folder / 'predictions.csv', data.classes, pred_rows)
    with (folder / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    label = 'SYNTHETIC SMOKE | ' if args.smoke else ''
    save_learning_curve(figures / 'training', history, f'{label}Test fold {fold} | Val fold {val_fold}')
    save_confusion(figures / 'confusion', np.asarray(stats['confusion_matrix']), data.classes, f'{label}Fold {fold} held-out test')
    save_confusion_normalized(figures / 'confusion_normalized', np.asarray(stats['confusion_matrix']), data.classes, f'{label}Fold {fold} normalized confusion')
    report = {**stats, 'test_loss': test['loss'], 'test_fold': fold, 'val_fold': val_fold,
              'train_folds': [i for i in range(1, 6) if i not in {fold, val_fold}],
              'train_n': int(tr.sum()), 'val_n': int(va.sum()), 'test_n': int(te.sum()),
              'best_epoch': best_epoch, 'epochs_run': len(history), 'best_val_loss': best_loss,
              'num_parameters': parameter_count(best), 'dataset_kind': data.kind,
              'training_config': settings, 'class_names': data.classes}
    write_json(folder / 'metrics.json', report)
    return report, pred_rows


def main():
    p = argparse.ArgumentParser(description='Train a spectrogram classifier on ESC-10')
    p.add_argument('--data', default='data/ESC-50')
    p.add_argument('--output', default='results')
    p.add_argument('--model', choices=['resnet18', 'tiny'], default='resnet18')
    p.add_argument('--pretrained', dest='pretrained', action='store_true', default=True)
    p.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    p.add_argument('--five-fold', action='store_true')
    p.add_argument('--test-fold', type=int, choices=range(1,6), default=5)
    p.add_argument('--val-fold', type=int, choices=range(1,6))
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--min-epochs', type=int, default=60)
    p.add_argument('--head-epochs', type=int, default=8)
    p.add_argument('--patience', type=int, default=35)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--backbone-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--mixup', type=float, default=0.2)
    p.add_argument('--augment', action='store_true', default=True)
    p.add_argument('--no-augment', action='store_false', dest='augment')
    p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--freeze-bn', action='store_true', default=True)
    p.add_argument('--train-bn', action='store_false', dest='freeze_bn')
    p.add_argument('--device', default='auto')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no-amp', action='store_false', dest='amp')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--nondeterministic', action='store_true')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.min_epochs, args.patience, args.batch, args.threads, args.log_every, args.save_every) < 1:
        p.error('epochs, min-epochs, patience, batch, threads, log-every must be positive')
    if args.min_epochs > args.epochs or args.head_epochs < 0 or args.head_epochs >= args.epochs:
        p.error('require 0 <= head-epochs < epochs and min-epochs <= epochs')
    if args.val_fold and args.five_fold:
        p.error('Use --val-fold only for a single split')
    if args.mixup < 0 or args.weight_decay < 0 or args.lr <= 0 or args.backbone_lr <= 0 or args.grad_clip < 0:
        p.error('invalid optimizer settings')
    if not 0 <= args.label_smoothing < 1 or args.min_delta < 0 or args.min_lr < 0:
        p.error('invalid loss or scheduler settings')
    if args.model == 'tiny' and args.pretrained:
        args.pretrained = False
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    seed_everything(args.seed, deterministic=not args.nondeterministic)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = synthetic_smoke_dataset(args.seed) if args.smoke else load_dataset(args.data, None if args.no_cache else root / 'cache')
    print(f'data={data.kind} features={data.features.shape} model={args.model} pretrained={args.pretrained and not args.smoke}')
    folds = range(1,6) if args.five_fold else [args.test_fold]
    reports, predictions = [], []
    for fold in folds:
        val = (fold % 5) + 1 if args.five_fold else (args.val_fold if args.val_fold is not None else (4 if fold==5 else (fold%5)+1))
        metrics, rows = train_fold(data, fold, val, args, root, device)
        reports.append(metrics)
        predictions.extend(rows)
    if args.five_fold and len(predictions) != len(data.features):
        raise AssertionError('Every example should have one held-out prediction')
    write_predictions(root / 'predictions_all.csv', data.classes, predictions)
    cm = np.sum([np.asarray(report['confusion_matrix']) for report in reports], axis=0)
    np.save(root / 'pooled_confusion.npy', cm)
    pooled_classes = class_metrics(cm, data.classes)
    fig = root / 'figures'
    fig.mkdir(exist_ok=True)
    prefix = 'SYNTHETIC SMOKE | ' if args.smoke else ''
    save_confusion(fig / 'pooled_confusion', cm, data.classes, f'{prefix}Pooled confusion')
    save_confusion_normalized(fig / 'pooled_confusion_normalized', cm, data.classes, f'{prefix}Normalized pooled confusion')
    save_class_metrics(fig / 'class_metrics', pooled_classes, f'{prefix}Pooled class metrics')
    save_class_metric_heatmap(fig / 'class_metric_heatmap', pooled_classes, f'{prefix}Class metric heatmap')
    save_summary_dashboard(fig / 'summary_dashboard', reports, pooled_classes, f'{prefix}Training summary')
    save_fold_diagnostics(fig / 'fold_diagnostics', reports, f'{prefix}Fold diagnostics')
    save_confidence_histogram(fig / 'confidence_histogram', predictions, f'{prefix}Prediction confidence')
    save_reliability_diagram(fig / 'reliability', predictions, f'{prefix}Reliability diagram')
    save_entropy_by_class(fig / 'entropy_by_class', predictions, f'{prefix}Prediction entropy by class')
    save_error_pairs(fig / 'error_pairs', predictions, f'{prefix}Top confusions')
    save_fold_prediction_summary(fig / 'fold_prediction_summary', predictions, f'{prefix}Held-out prediction summary')
    histories = {}
    for fold in folds:
        history_path = root / f'fold_{fold}' / 'history.csv'
        if history_path.is_file():
            with history_path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            histories[f'fold_{fold}'] = [
                {'epoch': int(r['epoch']), 'train_loss': float(r['train_loss']), 'val_loss': float(r['val_loss']), 'val_accuracy': float(r['val_accuracy']), 'lr': float(r['lr'])}
                for r in rows
            ]
    if histories:
        save_multi_fold_curves(fig / 'fold_curves', histories, f'{prefix}Training curves by fold')
        save_fold_overlay(fig / 'fold_overlay', histories, f'{prefix}Fold overlay')
    if args.five_fold:
        save_fold_accuracy(fig / 'fold_accuracy', reports, f'{prefix}Accuracy by fold')
    summary = {
        'dataset_kind': data.kind, 'architecture': args.model,
        'pretrained': bool(args.pretrained and not args.smoke and args.model == 'resnet18'),
        'data_source': 'https://github.com/karolpiczak/ESC-50' if not args.smoke else 'synthetic smoke test',
        'split_protocol': '3 train / 1 val / 1 test; test=f, val=(f % 5) + 1 in fivefold',
        'feature_config': DEFAULT_FEATURES.to_dict(),
        'accuracy_mean': float(np.mean([r['accuracy'] for r in reports])),
        'accuracy_std_ddof1': float(np.std([r['accuracy'] for r in reports], ddof=1)) if len(reports)>1 else None,
        'macro_f1_mean': float(np.mean([r['macro_f1'] for r in reports])),
        'accuracy_pooled': float(np.trace(cm)/np.sum(cm)),
        'macro_f1_pooled': float(np.mean([r['f1'] for r in pooled_classes])),
        'per_class_pooled': pooled_classes, 'pooled_confusion_matrix': cm.tolist(),
        'folds': [{k: r[k] for k in ['test_fold','val_fold','train_n','val_n','test_n','accuracy','macro_f1','best_epoch']} for r in reports],
        'total_test_predictions': len(predictions),
        'torch_version': torch.__version__, 'python_version': platform.python_version(),
        'seed': args.seed, 'device': str(device),
        'elapsed_seconds': round(time.perf_counter()-started, 2),
    }
    write_json(root/'summary.json', summary)
    print(f'Saved {root/"summary.json"}; pooled acc={summary["accuracy_pooled"]:.3f}; macro-F1={summary["macro_f1_pooled"]:.3f}')
    if args.smoke:
        print('Synthetic diagnostics only. Do not report these as ESC-10 accuracy.')


if __name__ == '__main__':
    main()
