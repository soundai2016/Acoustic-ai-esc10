"""Official ESC-10 fold handling and optional feature caching."""
from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import DEFAULT_FEATURES, FeatureConfig, audio_to_features


@dataclass(frozen=True)
class Record:
    path: Path
    fold: int
    category: str
    src_file: str


@dataclass
class Dataset:
    features: np.ndarray   # standardized per clip, [N, 40, 128]
    targets: np.ndarray
    folds: np.ndarray
    classes: list[str]
    records: list[Record] | None
    kind: str


def read_records(root: str | Path) -> tuple[list[Record], Path]:
    root = Path(root).expanduser().resolve()
    direct = root / 'meta' / 'esc50.csv'
    matches = [direct] if direct.is_file() else sorted(root.rglob('esc50.csv')) if root.is_dir() else []
    if not matches:
        raise FileNotFoundError(f'Cannot find ESC-50 meta/esc50.csv in {root}; see README for dataset layout')
    if len(matches) != 1:
        raise ValueError(f'Found multiple esc50.csv files under {root}; specify an unambiguous ESC-50 root')
    metadata = matches[0]
    audio_dir = metadata.parent.parent / 'audio'
    if not audio_dir.is_dir():
        raise FileNotFoundError(f'Cannot find ESC-50 audio folder: {audio_dir}')
    records = []
    with metadata.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'esc10', 'filename', 'fold', 'category', 'src_file'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}')
        for row in reader:
            if str(row['esc10']).strip().lower() not in {'true', 'yes', '1'}:
                continue
            fold = int(row['fold'])
            source = row['src_file'].strip()
            path = audio_dir / row['filename']
            if not path.is_file():
                raise FileNotFoundError(f'Missing ESC-10 WAV: {path}')
            records.append(Record(path, fold, row['category'].strip(), source))
    if len(records) != 400:
        raise ValueError(f'ESC-10 should have exactly 400 clips, found {len(records)}')
    if len({record.path.name for record in records}) != 400:
        raise ValueError('Duplicate filenames in ESC-10 metadata')
    classes = sorted({record.category for record in records})
    if len(classes) != 10:
        raise ValueError(f'ESC-10 should have 10 classes, found {len(classes)}')
    by_source = {}
    for record in records:
        if record.fold not in range(1, 6):
            raise ValueError(f'Invalid fold for {record.path.name}: {record.fold}')
        if record.src_file:
            if record.src_file in by_source and by_source[record.src_file] != record.fold:
                raise ValueError(f'Possible source leakage: {record.src_file} is assigned across folds')
            by_source[record.src_file] = record.fold
    for fold in range(1, 6):
        for name in classes:
            count = sum(record.fold == fold and record.category == name for record in records)
            if count != 8:
                raise ValueError(f'Expected 8 clips for class {name!r} in fold {fold}, found {count}')
    return records, metadata


def _fingerprint(records: list[Record], metadata: Path, cfg: FeatureConfig) -> str:
    h = hashlib.sha256()
    h.update(metadata.read_bytes())
    h.update(json.dumps(cfg.to_dict(), sort_keys=True).encode('utf-8'))
    for record in records:
        st = record.path.stat()
        h.update(f'{record.path.name}|{st.st_size}|{st.st_mtime_ns}\n'.encode('utf-8'))
    return h.hexdigest()[:20]


def load_dataset(root: str | Path, cache_dir: str | Path | None = None,
                 cfg: FeatureConfig = DEFAULT_FEATURES) -> Dataset:
    records, metadata = read_records(root)
    classes = sorted({r.category for r in records})
    labels = {name: i for i, name in enumerate(classes)}
    targets = np.asarray([labels[r.category] for r in records], dtype=np.int64)
    folds = np.asarray([r.fold for r in records], dtype=np.int64)
    cache = None
    if cache_dir is not None:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory / f'esc10_features_{_fingerprint(records, metadata, cfg)}.npz'
    if cache is not None and cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as cached:
                features = cached['features'].copy()
                if (features.shape == (400, cfg.n_mels, cfg.n_frames)
                        and np.array_equal(cached['targets'], targets)
                        and np.array_equal(cached['folds'], folds)
                        and np.array_equal(cached['classes'], np.asarray(classes))):
                    print(f'Using cached Log-Mel features: {cache}')
                    return Dataset(features, targets, folds, classes, records, 'ESC-10_REAL')
        except (OSError, KeyError, ValueError, EOFError) as exc:
            print(f'Ignored invalid cache {cache}: {exc}')
    print(f'Extracting Log-Mel features for {len(records)} real ESC-10 clips...')
    features = np.stack([audio_to_features(r.path, cfg) for r in records]).astype(np.float32)
    if cache is not None:
        np.savez_compressed(cache, features=features, targets=targets, folds=folds, classes=np.asarray(classes))
        print(f'Saved feature cache: {cache}')
    return Dataset(features, targets, folds, classes, records, 'ESC-10_REAL')


def synthetic_smoke_dataset(seed: int = 42, samples_per_class_fold: int = 2) -> Dataset:
    """Artificial pure tones; only for code integration, never for ESC-10 metrics."""
    from .audio import log_mel, standardize
    rng = np.random.default_rng(seed)
    cfg = DEFAULT_FEATURES
    t = np.arange(round(cfg.sample_rate * cfg.duration_seconds), dtype=np.float32) / cfg.sample_rate
    x, y, folds = [], [], []
    for cls in range(10):
        for fold in range(1, 6):
            for _ in range(samples_per_class_fold):
                freq = 220.0 + 120.0 * cls
                wave = (0.5 * np.sin(2 * np.pi * freq * t)
                        + 0.10 * rng.normal(size=t.shape)).astype(np.float32)
                x.append(standardize(log_mel(wave)))
                y.append(cls)
                folds.append(fold)
    return Dataset(np.stack(x), np.asarray(y, dtype=np.int64),
                   np.asarray(folds, dtype=np.int64),
                   [f'synthetic_{i}' for i in range(10)], None,
                   'SYNTHETIC_SMOKE_NOT_ESC10')


def get_split(folds: np.ndarray, test_fold: int, val_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_fold == val_fold or test_fold not in range(1, 6) or val_fold not in range(1, 6):
        raise ValueError('Test and validation folds must be two distinct values from 1 to 5')
    train = ~np.isin(folds, [test_fold, val_fold])
    val = folds == val_fold
    test = folds == test_fold
    if not (np.any(train) and np.any(val) and np.any(test)):
        raise ValueError('Empty train, validation, or test partition')
    if np.any(train & val) or np.any(train & test) or np.any(val & test):
        raise AssertionError('Split overlap detected')
    return train, val, test
