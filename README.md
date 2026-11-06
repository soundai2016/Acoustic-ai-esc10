# ESC-10 Sound Classification

ESC-10 environmental sound classification with a pretrained ResNet-18 spectrogram model. Includes five-fold evaluation, checkpoint recovery, single-file and batch inference, and automatic PNG/SVG figures.

## Setup

Python 3.10+ with compatible `torch` and `torchvision` builds is required.

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git data/ESC-50
python check_data.py --data data/ESC-50 --decode-all
```

The audio and metadata remain under `data/ESC-50/`. The ESC-10 subset has 400 clips, 10 classes, and five official folds.

## Training

```bash
# Single test fold: train 1-3 / val 4 / test 5
python train.py --data data/ESC-50 --output results_single

# Full five-fold experiment
python train.py --data data/ESC-50 --output results_5fold --five-fold
```

The default run uses **ResNet-18 with ImageNet pretraining**, a 150-epoch limit, eight head-only epochs, cosine learning-rate decay, AdamW, Mixup, SpecAugment, label smoothing, and validation-loss early stopping. First use of pretrained weights requires Internet access or a populated PyTorch model cache. CPU is supported; CUDA is recommended.

```bash
# Offline training (random initialization)
python train.py --no-pretrained --data data/ESC-50 --output results_scratch

# Longer training and larger patience
python train.py --five-fold --epochs 240 --min-epochs 60 --patience 45 \
  --batch 16 --output results_long

# Resume an interrupted run from last.pt
python train.py --five-fold --epochs 240 --min-epochs 60 --patience 45 \
  --batch 16 --output results_long --resume
```

The frontend is fixed at 16 kHz, STFT 512/160, 96 Mel bands and 256 frames. Each WAV is normalized independently. Checkpoints embed the feature settings and label order; they are loaded without downloading pretrained weights.

Each run saves `fold_*/best.pt` (lowest validation loss), `fold_*/last.pt` (resume state), `fold_*/history.csv`, `predictions_all.csv`, `summary.json`, and compact figures. The five-fold protocol uses 3 training folds, 1 validation fold and 1 held-out test fold per run. It is not directly equivalent to a 4-train/1-test protocol.

## Inference

```bash
python predict.py --checkpoint results_single/fold_5/best.pt \
  --audio examples/synthetic/synthetic_tone.wav --output infer/tone.json

python predict.py --checkpoint results_single/fold_5/best.pt \
  --audio path/to/long.wav --window-hop 2.5 --include-windows \
  --output infer/long.json

python predict.py --checkpoint results_single/fold_5/best.pt \
  --input-dir path/to/wavs --recursive --output infer/batch.json
```

PNG and SVG files are saved automatically to `infer/figures/`. The four files in `examples/synthetic/` are generated diagnostics, not environmental recordings.

## Held-out checks

```bash
python examples.py prepare --data data/ESC-50 --fold 5 --output examples/esc10
python examples.py verify --manifest examples/esc10/manifest.csv \
  --checkpoint results_single/fold_5/best.pt --output results_examples

python evaluate.py --checkpoint results_single/fold_5/best.pt \
  --data data/ESC-50 --output evaluation
python analyze_errors.py --results results_5fold --data data/ESC-50
```

The six selected WAVs are quick checks; the full held-out fold is the evaluation set. Do not use test results to select training settings. ImageNet pretraining is not AudioSet pretraining and does not guarantee a gain over the original TinyCNN.

## Figures and tests

```bash
python figures.py --results results_5fold --output results_5fold/figures
python -m pytest -q
python train.py --smoke --model tiny --epochs 2 --min-epochs 1 --head-epochs 0 --output smoke
```

The `--smoke` mode uses synthetic tones and does not report ESC-10 performance. Official data and recording credits: [ESC-50](https://github.com/karolpiczak/ESC-50).

