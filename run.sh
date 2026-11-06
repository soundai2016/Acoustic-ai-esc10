#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-help}"
DATA="${DATA:-data/ESC-50}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-150}"
case "$MODE" in
  check) python check_data.py --data "$DATA" --output results/data_report.json --decode-all ;;
  single) python train.py --data "$DATA" --output results_single --epochs "$EPOCHS" --device "$DEVICE" ;;
  fivefold) python train.py --data "$DATA" --output results_5fold --five-fold --augment --epochs "$EPOCHS" --device "$DEVICE" ;;
  figures) python figures.py --results results_5fold --output results_5fold/figures ;;
  *) printf 'Usage: ./run.sh {check|single|fivefold|figures}\nOptional env: DATA=..., DEVICE=cpu|cuda|npu, EPOCHS=150\n' ;;
esac
