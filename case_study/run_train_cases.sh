#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:?Usage: bash case_study/run_train_cases.sh NEW_OUTPUT_DIR [original training arguments]}"
shift
cd "$ROOT"
python -u "$ROOT/case_study/train_and_export.py" \
  --output "$RUN_DIR" \
  --model_name "$ROOT/bert-base-uncased" \
  --cache_path "$ROOT/data/database" \
  "$@"
