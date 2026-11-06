"""Shared audio processing for training, evaluation, and inference.

16 kHz, STFT 512/160, 96 Mel bands, 256 frames, and per-clip normalization.
This is not a calibrated dB representation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    duration_seconds: float = 5.0
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 96
    n_frames: int = 256
    mel_scale: float = 1000.0
    norm_epsilon: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_FEATURES = FeatureConfig()


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    if cfg.n_fft < 4 or cfg.n_mels < 1 or not 0 < cfg.hop_length <= cfg.n_fft:
        raise ValueError('Invalid STFT/Mel feature configuration')
    hz = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(cfg.sample_rate / 2), cfg.n_mels + 2))
    bins = np.clip(np.floor((cfg.n_fft + 1) * hz / cfg.sample_rate).astype(int), 0, cfg.n_fft // 2)
    fb = np.zeros((cfg.n_mels, cfg.n_fft // 2 + 1), dtype=np.float32)
    for idx in range(cfg.n_mels):
        left, center, right = bins[idx : idx + 3]
        if left == center:
            center = min(center + 1, cfg.n_fft // 2)
        if center == right:
            right = min(right + 1, cfg.n_fft // 2)
        if left < center:
            fb[idx, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if center < right:
            fb[idx, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    return fb


def read_mono_resampled(path: str | Path, cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    """Decode an entire recording, downmix and resample; never alter source files.

    The original classroom clips are exactly five seconds long; this function
    additionally supports independent, longer recordings for windowed inference.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Audio file not found: {path}')
    try:
        wave, input_sr = sf.read(str(path), dtype='float32', always_2d=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Could not decode audio {path}: {exc}') from exc
    if len(wave) == 0 or input_sr <= 0:
        raise ValueError(f'Empty or invalid recording: {path}')
    if not np.isfinite(wave).all():
        raise ValueError(f'NaN/Inf samples in {path}')
    mono = wave.mean(axis=1, dtype=np.float32)
    if input_sr != cfg.sample_rate:
        scale = gcd(int(cfg.sample_rate), int(input_sr))
        mono = signal.resample_poly(mono, cfg.sample_rate // scale, int(input_sr) // scale).astype(np.float32)
    return mono.astype(np.float32, copy=False)


def pad_or_crop(wave: np.ndarray, cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    count = round(cfg.sample_rate * cfg.duration_seconds)
    if len(wave) < count:
        return np.pad(wave, (0, count - len(wave))).astype(np.float32)
    return wave[:count].astype(np.float32, copy=False)


def load_wave(path: str | Path, cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    return pad_or_crop(read_mono_resampled(path, cfg), cfg)


def windowed_features(path: str | Path, hop_seconds: float = 2.5,
                      cfg: FeatureConfig = DEFAULT_FEATURES) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return one or more fixed-length model windows from an entire recording.

    Window starts cover the complete recording. The final full window is aligned
    to the file end, preventing padding artifacts on long recordings. Short
    recordings are zero-padded once. Aggregated scores are mean softmax outputs.
    """
    if not 0 < hop_seconds <= cfg.duration_seconds:
        raise ValueError('hop_seconds must be in (0, duration_seconds]')
    wave = read_mono_resampled(path, cfg)
    win = round(cfg.sample_rate * cfg.duration_seconds)
    hop = max(1, round(cfg.sample_rate * hop_seconds))
    if len(wave) <= win:
        starts = [0]
    else:
        starts = list(range(0, len(wave) - win + 1, hop))
        tail = len(wave) - win
        if starts[-1] != tail:
            starts.append(tail)
    features = []
    intervals = []
    for start in starts:
        clip = pad_or_crop(wave[start:start + win], cfg)
        features.append(standardize(log_mel(clip, cfg), cfg))
        intervals.append((start / cfg.sample_rate,
                          min((start + win) / cfg.sample_rate, len(wave) / cfg.sample_rate)))
    return np.stack(features).astype(np.float32), intervals

def log_mel(wave: np.ndarray, cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    if wave.ndim != 1 or not np.isfinite(wave).all() or len(wave) < cfg.n_fft:
        raise ValueError('Expected a finite mono waveform with at least one STFT window')
    _, _, stft = signal.stft(
        wave, fs=cfg.sample_rate, nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length, nfft=cfg.n_fft, boundary=None
    )
    power = np.abs(stft) ** 2
    mel = mel_filterbank(cfg) @ power
    log = np.log1p(cfg.mel_scale * mel)
    source = np.arange(log.shape[1])
    desired = np.linspace(0, log.shape[1] - 1, cfg.n_frames)
    return np.stack([np.interp(desired, source, row) for row in log]).astype(np.float32)


def standardize(spec: np.ndarray, cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    if spec.shape != (cfg.n_mels, cfg.n_frames):
        raise ValueError(f'Unexpected feature shape {spec.shape}; expected {(cfg.n_mels, cfg.n_frames)}')
    mean = float(spec.mean())
    std = float(spec.std())
    return ((spec - mean) / (std + cfg.norm_epsilon)).astype(np.float32)


def audio_to_features(path: str | Path, cfg: FeatureConfig = DEFAULT_FEATURES) -> np.ndarray:
    return standardize(log_mel(load_wave(path, cfg), cfg), cfg)


def features_to_batch(spec: np.ndarray) -> np.ndarray:
    """Add batch and channel dimensions."""
    return spec[np.newaxis, np.newaxis, :, :]
