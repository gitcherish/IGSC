# IGSC

Research implementation for **IGSC: An Iterative Graph-Based Event Causality Extraction Model with Structure-Aware Pretraining and Supervised Contrastive Learning**.

IGSC combines a structure-aware BERT encoder, supervised contrastive learning for event-pair classification, and iterative document-level causal graph updates. The extraction implementation builds on [iLIF](https://github.com/LchengC/iLIF), and dependency-mask pre-training follows RE-DMP. See [References and attribution](#references-and-attribution).

The workspace contains the extraction code, EventStoryLine feature caches, pre-training utilities, and optional case-study tools. The historical encoder checkpoint and complete pre-training corpus used for the manuscript results have not yet been established. These instructions describe the available code and do not certify reproduction of the reported scores.

## GitHub package contents

This folder is a separate distribution copy of the working project. It contains
the source code, two small input caches, seed-209 document splits, pre-training
utilities, optional checkpoint/export tools, and one historical reference log.
See [data documentation](data/database/README.md), [reference results](results/README.md),
and [third-party notices](THIRD_PARTY_NOTICES.md).

Weights, optimizer states, installation wheels, complete Wikipedia corpora,
paper drafts, and generated run directories are excluded. `.gitignore` protects
these outputs while explicitly keeping the two documented input pickle files.
`.gitattributes` uses LF text endings for portable Bash scripts and stable hashes.

Only the default encoder path in parameter.py was made portable; the extraction
algorithm is unchanged. The integrity backups in saef_repro/originals describe
this packaged version, not the old private workspace snapshot. Some inherited
comments and auxiliary guides remain in Chinese; an English-comment pass is still
needed if required by the target journal. No GitHub repository was created or
uploaded by this packaging step.

## Project files

- `main.py`: five-fold training, validation, and test evaluation.
- `parameter.py`: command-line arguments and defaults.
- `model/model.py`: classifier, contrastive objective, and iterative prediction loop.
- `model/CGE.py`: graph attention and intra-/inter-sentence fusion.
- `utils.py`: random seeds and evaluation functions.
- `data/process_esc_csr.py`: document feature loader and batching.
- `features_pkl_process/`: feature-generation scripts.
- `RE-DMP-main/pre-training/`: upstream dependency-mask pre-training code.
- `saef_repro/`: optional checkpoint conversion, data audits, and corpus reconstruction tools.
- `case_study/`: optional best-checkpoint saving and held-out trajectory export.

Some inherited identifiers and auxiliary documents still use `SAEF`, `bertCSRModel`, or `iLIF`. The active extraction entry imports `model/`, not `model_1/`.

## Environment and installation

The documented target is **Linux x86_64, Python 3.7, PyTorch 1.13.1 with CUDA 11.6**. This retains the PyTorch version from the existing code-side requirements and pins the previously unspecified sparse extension. The manuscript currently lists PyTorch 1.12.1; resolve this discrepancy against the actual experiment environment before publication. This installation profile is not a historical environment lock or a verified GPU run.

The original graph code allocates tensors with `.cuda()`. Full extraction training requires a working NVIDIA GPU environment, even though the argument parser exposes `--disable-cuda`. Run the following Bash examples from the repository root.

Create an isolated environment, for example with Conda:

```bash
conda create -n igsc python=3.7 pip=23.3.2 -y
conda activate igsc
```

Install PyTorch before packages whose installation may import it:

```bash
python -m pip install "torch==1.13.1+cu116" --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
python -m pip check
```

The requirements file uses the PyG `torch-1.13.0+cu116` wheel page for the PyTorch 1.13.x family and requires binary wheels for `torch-scatter` and `torch-sparse`. Do not mix these with CPU, CUDA 10.2, or PyTorch 1.12 wheels. The local `torchfile12.1/` and `kaggle_requirements.txt` describe a different environment and are not used here. Transitive dependencies are resolved by pip; capture `pip freeze` from an actual successful run to record the full environment.

Verify the installed versions and GPU access:

```bash
nvidia-smi
python -c "import torch, transformers, torch_geometric, torch_scatter, torch_sparse; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available()); print('transformers:', transformers.__version__, 'PyG:', torch_geometric.__version__, 'scatter:', torch_scatter.__version__, 'sparse:', torch_sparse.__version__); assert torch.cuda.is_available()"
```

Installation references: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [PyG installation](https://pytorch-geometric.readthedocs.io/en/2.1.0/notes/installation.html), and [matching extension wheels](https://data.pyg.org/whl/torch-1.13.0+cu116.html).

## EventStoryLine data

The manuscript evaluates on **EventStoryLine v0.9**. Obtain the corpus from its [original repository](https://github.com/tommasoc80/EventStoryLine); the inherited iLIF documentation also points to the [initial release](https://github.com/cltl/EventStoryLine/). Follow the corpus providers' terms and preserve the dataset citation.

The loader expects these files under `--cache_path`:

```text
data/database/
  EventStoryLine_bert_intra_and_inter_train_features.pkl
  EventStoryLine_bert_intra_and_inter_dev_features.pkl
```

The `train` cache is the pool split into five training/test folds. The `dev` cache is a separate validation set reused in all folds. Each feature record represents one document and contains sentence/token encodings, event trigger positions, candidate event pairs, relation labels, and sentence-location labels. Load feature pickles only from a trusted source.

Training labels are `0` for no causal relation, `1` for the first event causing the second, and `2` for the reverse direction. Sentence-location labels are `0` for intra-sentence and `1` for inter-sentence pairs. Confirm these mappings when preparing data from another source.

### Building and checking features

`features_pkl_process/process.py` currently loads an intermediate `XXX.npy` file. The raw-corpus-to-intermediate conversion is not supplied as a complete executable pipeline. It assigns topics `37` and `41` to validation and writes the two feature caches. Until the missing conversion is supplied, use the verified experiment caches and record their hashes rather than claiming that raw-data preprocessing is complete.

Inspect the caches without loading the extraction model:

```bash
python case_study/export_cases.py inspect --features data/database/EventStoryLine_bert_intra_and_inter_train_features.pkl
python case_study/export_cases.py inspect --features data/database/EventStoryLine_bert_intra_and_inter_dev_features.pkl
python saef_repro/audit.py --data-dir data/database --seed 209
```

The existing `saef_repro/data_audit.json` records 233 train/test-pool documents and 25 validation documents, with 5,478 positive pairs in total. The manuscript table lists 5,625. These inventories differ; do not silently change the caches or describe the counts as identical. Re-run the audit for any new data version and keep its hashes and document lists with the results.

## Structure-aware encoder

Supply `--model_name` with a local Hugging Face-compatible **BERT-base uncased** encoder directory containing matching model weights, `config.json`, and tokenizer files including `vocab.txt`. The classifier expects a hidden size of 768. The packaged default is `./bert-base-uncased`; pass `--model_name` explicitly to select a verified structure-aware encoder.

Model weights are not bundled. The private workspace contained mixed weight formats, which do not establish the manuscript encoder provenance. Do not present an ordinary BERT initialization or a small pre-training example as the final IGSC encoder.

For dependency-mask pre-training and checkpoint export, consult:

- [RE-DMP pre-training instructions](RE-DMP-main/pre-training/readme.md).
- [Checkpoint and data provenance notes](saef_repro/README.md).
- [Full-corpus reconstruction route](saef_repro/RUN.md).
- [Limited-disk Wikipedia subset route](saef_repro/SUBSET_RUN.md).

The two reconstruction routes use different corpus/schedule choices and must be reported separately. The RE-DMP sample Wiki files are examples, not the full pre-training corpus. Corpus construction requires parser tools and Java beyond the root Python requirements. Record the actual corpus version, parser configuration, training steps, and checkpoint hashes. A new reconstruction does not establish the provenance of the historical checkpoint.

## Five-fold training and evaluation

After preparing the caches and a verified structure-aware encoder, set `ENCODER_DIR` to its actual path. For example, if an exported encoder has been created at this location:

```bash
ENCODER_DIR=./saef_repro/encoder_dmp_formal
```

Run the extraction experiment:

```bash
python -u main.py \
  --model_name "$ENCODER_DIR" \
  --cache_path ./data/database \
  --dataset_type EventStoryLine --model_type bert --inter_or_intra intra_and_inter \
  --seed 209 --k_fold 5 --epoch 30 --batch_size 1 \
  --t_lr 2e-5 --lr 1e-4 --loss_type focal --gamma 2.0 \
  --beta_intra 0.7 --beta_inter 0.3 \
  --w 0.6 --rate 0.0 --threshold 2 \
  --max_iteration 9 --min_iteration 2 --num_heads 4 --mlp_drop 0.4 \
  --gradient_accumulation_steps 1 --valid_epoch 1 --test_epoch 1
```

This command reflects the current defaults and the configuration printed in the existing October 9, 2026 log. It is not proof that a newly supplied encoder will reproduce that log. Batch size 1 is required by the current document collator. The code hard-codes contrastive weight `0.1` and temperature `0.5`; these are not command-line options.

The loop uses `KFold(n_splits=5, shuffle=True, random_state=209)` over document feature records. This is a document-level split, not a topic-isolated split. Keep cache ordering unchanged when reconstructing folds from the seed. Validation F1 selects the reported test epoch; ties select the later epoch. The random seed is initialized once before the five-fold loop.

Logs are written under `logout/seed_209/` with a timestamp. `main.py` reports per-fold and fold-mean precision, recall, and F1 for all, intra-sentence, and inter-sentence pairs. Scores follow the transformation and counting in `utils.py`: reversed predictions on a gold causal pair count as missed positives, while causal predictions on a gold noncausal pair count as false positives. Do not assume equivalence with another implementation's directional micro-F1. Mean F1 is the mean of the five fold F1 values, not pooled F1 or F1 calculated from mean precision and recall.

### Saving complete models and exporting cases

The original `main.py` does not save checkpoints or provide a standalone checkpoint-evaluation command. To save best-validation models, exact fold indices, configuration, and held-out iterative trajectories, use the optional entry:

```bash
python -u case_study/train_and_export.py \
  --output case_study/runs/igsc_seed209 \
  --model_name "$ENCODER_DIR" \
  --cache_path ./data/database \
  --seed 209 --epoch 30 --batch_size 1 --k_fold 5 \
  --beta_intra 0.7 --beta_inter 0.3 --w 0.6 --threshold 2
```

The output directory must not already exist. Add `--preflight` to check inputs and the environment before starting a new run. See the [training/export guide](case_study/TRAINING.md) and [trajectory export guide](case_study/README.md) for outputs and selection rules. This entry preserves the original training implementation, including its optimizer parameter groups. The current optimizer does not include `proj_head`; these instructions do not fix or validate that behavior.

## Results and release status

A complete reproducibility release still needs:

- The actual encoder download location and its training/data provenance.
- A complete raw-data preprocessing route or a documented distribution route for the exact caches.
- Resolution of the manuscript/cache pair-count and PyTorch-version differences.
- Exact fold documents, configurations, environment captures, and logs for each reported experiment.
- Figure source data, including independent runs for each parameter-sweep point. A default-configuration log does not establish a complete sweep.
- Confirmation of third-party redistribution permissions and a license for the authors' original contributions.

Distinguish existing results and examples from newly completed runs. No new GPU training or verification of manuscript scores was performed when updating this documentation. A public IGSC repository URL or archived release identifier has not yet been supplied.

## References and attribution

The extraction code and feature-processing layout derive from:

> C. Liu, W. Xiang, and B. Wang. 2024. *Identifying while Learning for Document Event Causality Identification*. ACL, pages 3815-3827. [Paper](https://aclanthology.org/2024.acl-long.210/). [iLIF code](https://github.com/LchengC/iLIF).

Dependency-mask pre-training follows:

> Y. Tian, Y. Song, and F. Xia. 2022. *Improving Relation Extraction through Syntax-induced Pre-training with Dependency Masking*. Findings of ACL, pages 1875-1886. [Paper](https://aclanthology.org/2022.findings-acl.147/). See the bundled [RE-DMP README](RE-DMP-main/README.md) for upstream information.

Dataset reference:

> T. Caselli and P. Vossen. 2017. *The Event StoryLine Corpus: A New Benchmark for Causal and Temporal Relation Extraction*. Events and Stories in the News Workshop, pages 77-86. [Paper](https://aclanthology.org/W17-2711/).

The inherited iLIF implementation also acknowledges [HAN](https://github.com/Jhy1993/HAN) and deterministic graph-operation changes contributed by [lyj963](https://github.com/lyj963). Preserve applicable upstream notices when redistributing adapted code. No blanket license is assigned here to third-party material.
