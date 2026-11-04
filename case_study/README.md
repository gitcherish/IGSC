# Checkpoint saving and case-study export

These optional tools add output recording to the current five-fold training entry.
They retain the original training and evaluation logic, including its current
optimizer parameter groups. Synthetic tests do not establish paper performance.

From the repository root, supply an actual verified encoder path:

```bash
python case_study/train_and_export.py --output case_study/runs/seed209 --model_name ./saef_repro/encoder_dmp_formal --cache_path ./data/database --preflight
python -u case_study/train_and_export.py --output case_study/runs/seed209 --model_name ./saef_repro/encoder_dmp_formal --cache_path ./data/database
```

The output directory must not exist before a training run. This command uses the
current parameter defaults. It saves per-fold best.pt, run.config.json, best.json,
training/test indices, document IDs, and an export manifest. After all five folds
complete, it exports held-out iterative prediction trajectories. Selection may
report insufficient successful or failed cases instead of inventing examples.

To export existing matching checkpoints, fill manifest.example.json and
run.config.example.json with real paths and provenance, then run:

```bash
python case_study/export_cases.py export --manifest case_study/manifest.json --output case_study/outputs/export1 --device cuda:0
```

The observer verifies equality of logits and loss with and without observation.
Do not use random heads or encoder-only checkpoints as trained extraction models.
Generated outputs and weights are ignored by .gitignore. Publish selected results
separately with their configurations, fold membership, and provenance.

Optional synthetic protocol tests after installing requirements-dev.txt:

```bash
python -m pytest case_study/test_export_cases.py case_study/test_train_and_export.py
```
