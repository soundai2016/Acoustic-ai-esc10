import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from esc10.core import save_checkpoint
from esc10.model import TinyCNN
from examples import DEFAULT_CLASSES, prepare, sha256, synthesize, verify


def make_dataset(root: Path):
    meta = root / 'meta'
    audio = root / 'audio'
    meta.mkdir(parents=True)
    audio.mkdir()
    categories = sorted(set(DEFAULT_CLASSES) | {'crying_baby', 'rooster', 'sneezing', 'crackling_fire'})
    names = []
    rows = []
    for fold in range(1, 6):
        for c, cls in enumerate(categories):
            for ix in range(8):
                filename = f'{fold}-{c:03d}{ix:02d}-A-{c}.wav'
                path = audio / filename
                if fold == 5 and ix == 0:
                    sf.write(path, np.zeros(40000, dtype=np.float32), 8000)
                    names.append(filename)
                else:
                    path.touch()
                rows.append({'filename': filename, 'fold': fold, 'target': c,
                             'category': cls, 'esc10': 'True',
                             'src_file': f'source-{fold}-{c}-{ix}', 'take': 'A'})
    with (meta / 'esc50.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    credits = [f'- [{name.rsplit("-", 1)[0]}.ogg]: test recording by example author [CC0]' for name in names]
    (root / 'LICENSE').write_text('\n'.join(credits), encoding='utf-8')
    return categories


def test_prepare_and_verify(tmp_path):
    dataset = tmp_path / 'dataset'
    classes = make_dataset(dataset)
    output = tmp_path / 'samples'
    rows = prepare(dataset, output, 5, list(DEFAULT_CLASSES))
    assert len(rows) == 6
    assert all((output / row['filename']).is_file() for row in rows)
    assert all('example author' in row['attribution'] for row in rows)
    assert all(sha256(output / row['filename']) == row['sha256'] for row in rows)
    prepare(dataset, output, 5, list(DEFAULT_CLASSES))

    checkpoint = tmp_path / 'best.pt'
    save_checkpoint(checkpoint, TinyCNN(), classes, 1, 1.0, 4, 5, 'ESC-10_REAL', {'seed': 1}, architecture='tiny')
    result = verify(output / 'manifest.csv', checkpoint, tmp_path / 'checks', 'cpu')
    assert result['sample_count'] == 6
    assert (tmp_path / 'checks' / 'figures' / 'example_confusion.png').is_file()
    assert len(list(csv.DictReader((tmp_path / 'checks' / 'example_checks.csv').open()))) == 6

    tampered = output / rows[0]['filename']
    tampered.write_bytes(b'altered')
    with pytest.raises(ValueError, match='modified'):
        verify(output / 'manifest.csv', checkpoint, tmp_path / 'failed', 'cpu')


def test_reject_non_held_out_checkpoint(tmp_path):
    dataset = tmp_path / 'dataset'
    classes = make_dataset(dataset)
    output = tmp_path / 'samples'
    prepare(dataset, output, 5, ['dog'])
    checkpoint = tmp_path / 'best.pt'
    save_checkpoint(checkpoint, TinyCNN(), classes, 1, 1.0, 3, 4, 'ESC-10_REAL', {'seed': 1}, architecture='tiny')
    with pytest.raises(ValueError, match='differs from checkpoint test fold'):
        verify(output / 'manifest.csv', checkpoint, tmp_path / 'checks', 'cpu')


def test_synthetic_audio_generation(tmp_path):
    synthesize(tmp_path / 'signals')
    files = list((tmp_path / 'signals').glob('*.wav'))
    assert len(files) == 4
    assert sorted(round(sf.info(str(f)).duration) for f in files) == [5, 5, 5, 12]
