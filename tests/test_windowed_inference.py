import math
import numpy as np
import soundfile as sf
import torch

from esc10.audio import audio_to_features, windowed_features
from esc10.core import save_checkpoint
from esc10.figures import save_batch_overview, save_inference_summary, save_window_timeline
from esc10.model import TinyCNN
from predict import predict_files


def test_short_audio_one_window_matches_single_clip(tmp_path):
    rate = 8000
    x = np.sin(2 * np.pi * 300 * np.arange(rate * 2) / rate).astype(np.float32)
    path = tmp_path / 'short.wav'
    sf.write(path, x, rate)
    windows, intervals = windowed_features(path, hop_seconds=2.5)
    assert windows.shape == (1, 96, 256)
    assert np.allclose(windows[0], audio_to_features(path))
    assert intervals == [(0.0, 2.0)]


def test_long_audio_last_window_reaches_end(tmp_path):
    rate = 8000
    x = np.sin(2 * np.pi * 440 * np.arange(rate * 11) / rate).astype(np.float32)
    path = tmp_path / 'long.wav'
    sf.write(path, x, rate)
    windows, intervals = windowed_features(path, hop_seconds=2.5)
    assert windows.shape[0] == 4
    assert intervals[0] == (0.0, 5.0)
    assert intervals[-1] == (6.0, 11.0)


def test_predict_multiwindow_and_figures(tmp_path):
    path = tmp_path / 'long.wav'
    rate = 8000
    wave = np.sin(2 * np.pi * 440 * np.arange(rate * 8) / rate)
    sf.write(path, wave, rate)
    model = TinyCNN()
    checkpoint = tmp_path / 'smoke.pt'
    save_checkpoint(checkpoint, model, [f'synthetic_{i}' for i in range(10)], 1, 1.2, 4, 5, 'SYNTHETIC_SMOKE_NOT_ESC10', {'seed': 7}, architecture='tiny')
    out = predict_files([str(checkpoint)], [path], torch.device('cpu'), 3, 4, window_hop=2.5, include_windows=True)
    assert len(out) == 1 and out[0]['num_windows'] == 3
    assert len(out[0]['windows']) == 3
    assert math.isclose(sum(out[0]['probabilities'].values()), 1, abs_tol=1e-6)

    save_inference_summary(tmp_path / 'summary', out[0])
    save_window_timeline(tmp_path / 'timeline', out[0])
    save_batch_overview(tmp_path / 'batch', out)
    assert (tmp_path / 'summary.png').is_file()
    assert (tmp_path / 'timeline.png').is_file()
    assert (tmp_path / 'batch.png').is_file()
