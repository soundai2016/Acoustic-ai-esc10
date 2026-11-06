"""Shared training / evaluation / inference utilities."""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .audio import DEFAULT_FEATURES, FeatureConfig
from .model import build_model


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def choose_device(request: str) -> torch.device:
    if request == 'auto':
        if torch.cuda.is_available():
            request = 'cuda'
        else:
            try:
                import torch_npu  # noqa: F401  # optional Ascend environment
                if torch.npu.is_available():
                    request = 'npu'
                else:
                    request = 'cpu'
            except (ImportError, AttributeError):
                request = 'cpu'
    if request.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if request.startswith('npu'):
        try:
            import torch_npu  # noqa: F401
            if not torch.npu.is_available():
                raise RuntimeError('Ascend NPU is not available')
        except ImportError as exc:
            raise RuntimeError('NPU requested; install compatible torch_npu and CANN first') from exc
    return torch.device(request)


def make_loader(features: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                batch_size: int, shuffle: bool, seed: int = 42) -> DataLoader:
    values = torch.from_numpy(np.ascontiguousarray(features[mask, None], dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(labels[mask], dtype=np.int64))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(values, targets), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator, pin_memory=torch.cuda.is_available())


def spec_augment(x: torch.Tensor, probability: float = 0.65,
                 time_mask: int = 16, freq_mask: int = 5) -> torch.Tensor:
    """Time/frequency masking. Apply only after per-clip normalization and only in training."""
    x = x.clone()
    _, _, freq, frames = x.shape
    for row in range(x.shape[0]):
        if random.random() > probability:
            continue
        t_width = random.randint(0, min(time_mask, frames))
        f_width = random.randint(0, min(freq_mask, freq))
        if t_width:
            start = random.randint(0, frames - t_width)
            x[row, :, :, start:start + t_width] = 0.0
        if f_width:
            start = random.randint(0, freq - f_width)
            x[row, :, start:start + f_width, :] = 0.0
    return x


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss, count = 0.0, 0
    predictions, truths, probabilities = [], [], []
    for features, y in loader:
        features, y = features.to(device), y.to(device)
        logits = model(features)
        total_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        prob = torch.softmax(logits, dim=1)
        count += len(y)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        truths.extend(y.cpu().tolist())
        probabilities.extend(prob.cpu().tolist())
    if not count:
        raise ValueError('Empty evaluation loader')
    return {'loss': total_loss / count, 'accuracy': float(np.mean(np.asarray(truths) == np.asarray(predictions))),
            'true': truths, 'pred': predictions, 'prob': probabilities}


def confusion_matrix(truths: list[int], predictions: list[int], classes: int = 10) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(truths, predictions):
        matrix[t, p] += 1
    return matrix


def class_metrics(cm: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for idx, name in enumerate(names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({'class': name, 'precision': precision, 'recall': recall,
                     'f1': f1, 'support': int(cm[idx].sum())})
    return rows


def classification_metrics(truths: list[int], predictions: list[int], names: list[str]) -> dict:
    cm = confusion_matrix(truths, predictions, len(names))
    rows = class_metrics(cm, names)
    return {'accuracy': float(np.trace(cm) / cm.sum()),
            'macro_f1': float(np.mean([row['f1'] for row in rows])),
            'per_class': rows, 'confusion_matrix': cm.tolist(), 'n': int(cm.sum())}


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_predictions(path: str | Path, names: list[str], items: list[dict]) -> None:
    fields = ['filename', 'fold', 'truth', 'prediction', 'correct', 'confidence'] + [f'prob_{x}' for x in names]
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def make_prediction_rows(results: dict, names: list[str], filenames: list[str], test_fold: int) -> list[dict]:
    rows = []
    if len(filenames) != len(results['true']):
        raise ValueError('Evaluation filename count differs from number of predictions')
    for fname, true, predicted, prob in zip(filenames, results['true'], results['pred'], results['prob']):
        row = {'filename': fname, 'fold': test_fold,
               'truth': names[true], 'prediction': names[predicted],
               'correct': int(true == predicted), 'confidence': float(max(prob))}
        row.update({f'prob_{name}': float(p) for name, p in zip(names, prob)})
        rows.append(row)
    return rows


def save_checkpoint(path: Path, model: nn.Module, classes: list[str],
                    epoch: int, val_loss: float, val_fold: int, test_fold: int,
                    dataset_kind: str, training: dict, architecture: str = 'resnet18',
                    feature_config: FeatureConfig = DEFAULT_FEATURES) -> None:
    payload = {
        'format_version': 2,
        'architecture': architecture,
        'state_dict': model.state_dict(),
        'class_names': list(classes),
        'feature_config': feature_config.to_dict(),
        'best_epoch': int(epoch),
        'validation_loss': float(val_loss),
        'test_fold': int(test_fold),
        'validation_fold': int(val_fold),
        'training_folds': [i for i in range(1, 6) if i not in {test_fold, val_fold}],
        'dataset_kind': dataset_kind,
        'training_config': training,
        'torch_version': str(torch.__version__),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device):
    path = Path(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('format_version') != 2:
        raise ValueError(f'Unsupported checkpoint format: {path}')
    classes = ckpt.get('class_names')
    if not isinstance(classes, list) or len(classes) != 10:
        raise ValueError(f'Invalid 10-class mapping: {path}')
    config = FeatureConfig(**ckpt['feature_config'])
    model = build_model(ckpt['architecture'], num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.to(device).eval(), classes, config, ckpt
