import numpy as np
import pytest
import soundfile as sf
import torch

from esc10.audio import DEFAULT_FEATURES, audio_to_features, load_wave, log_mel, standardize
from esc10.core import classification_metrics, load_checkpoint, save_checkpoint, spec_augment
from esc10.data import get_split
from esc10.model import TinyCNN, build_model, parameter_count


def test_model_shape_and_count():
    model = TinyCNN().eval()
    assert parameter_count(model) == 9850
    with torch.inference_mode():
        logits = model(torch.zeros(2, 1, 96, 256))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_audio_pipeline_on_generated_wav(tmp_path):
    rate = 16000
    t = np.arange(rate * 2) / rate
    wave = np.stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 500 * t)], axis=1)
    path = tmp_path / 'stereo.wav'
    sf.write(path, wave, rate)
    processed = load_wave(path)
    assert processed.shape == (80000,)
    spec = audio_to_features(path)
    assert spec.shape == (96, 256)
    assert spec.dtype == np.float32
    assert np.isfinite(spec).all()
    assert np.isclose(float(spec.mean()), 0.0, atol=1e-4)


def test_silence_and_augment():
    x = standardize(log_mel(np.zeros(80000, dtype=np.float32)))
    assert x.shape == (96, 256)
    assert np.all(x == 0)
    z = spec_augment(torch.ones(3, 1, 96, 256), probability=1.0, time_mask=16, freq_mask=5)
    assert z.shape == (3, 1, 96, 256)
    assert torch.all(z <= 1)


def test_fold_masks_and_confusion():
    folds = np.repeat(np.arange(1, 6), 80)
    train, val, test = get_split(folds, test_fold=5, val_fold=4)
    assert (train.sum(), val.sum(), test.sum()) == (240, 80, 80)
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    with pytest.raises(ValueError):
        get_split(folds, 5, 5)
    result = classification_metrics([0, 1, 1], [0, 1, 0], [str(i) for i in range(10)])
    assert np.isclose(result['accuracy'], 2 / 3)
    assert result['confusion_matrix'][1][0] == 1


def test_checkpoint_roundtrip(tmp_path):
    model = TinyCNN()
    path = tmp_path / 'best.pt'
    save_checkpoint(path, model, [f'class_{i}' for i in range(10)], 1, 2.0, 4, 5, 'ESC-10_REAL', {'seed': 42}, architecture='tiny')
    loaded, classes, config, record = load_checkpoint(path, torch.device('cpu'))
    assert classes[0] == 'class_0'
    assert config == DEFAULT_FEATURES
    assert record['test_fold'] == 5
    assert parameter_count(loaded) == 9850


def test_resnet18_shapes_and_freezing():
    model = build_model('resnet18', num_classes=10, pretrained=False)
    assert parameter_count(model) > 10_000_000
    model.freeze_backbone(True)
    assert not model.backbone.layer1[0].conv1.weight.requires_grad
    assert model.backbone.fc.weight.requires_grad
    model.freeze_backbone(False)
    assert model.backbone.layer1[0].conv1.weight.requires_grad
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 1, 96, 256))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
