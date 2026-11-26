# IGSC

Research implementation for **IGSC: An Iterative Graph-Based Event Causality Extraction Model with Structure-Aware Pretraining and Supervised Contrastive Learning**.

IGSC combines a structure-aware BERT encoder, supervised contrastive learning for event-pair classification, and iterative document-level causal graph updates. The extraction implementation builds on [iLIF](https://github.com/LchengC/iLIF), and dependency-mask pre-training follows RE-DMP. See [References and attribution](#references-and-attribution).

## Project files

- `main.py`: five-fold training, validation, and test evaluation.
- `parameter.py`: command-line arguments and defaults.
- `model/model.py`: classifier, contrastive objective, and iterative prediction loop.
- `model/CGE.py`: graph attention and intra-/inter-sentence fusion.
- `utils.py`: random seeds and evaluation functions.
- `data/process_esc_csr.py`: document feature loader and batching.
- `features_pkl_process/`: feature-generation scripts.
- `case_study/`: optional best-checkpoint saving and held-out trajectory export.

## Environment and installation

The documented target is **Linux x86_64, Python 3.7, PyTorch 1.13.1 with CUDA 11.6**. This retains the PyTorch version from the existing code-side requirements and pins the previously unspecified sparse extension. 

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

The requirements file uses the PyG `torch-1.13.0+cu116` wheel page for the PyTorch 1.13.x family and requires binary wheels for `torch-scatter` and `torch-sparse`. 

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

## Structure-aware encoder

The IGSC project uses the **RE-DMP-pre-trained structure-aware BERT-base uncased encoder** already placed in `./bert_base_uncased/`. This is the syntax-induced encoder obtained through RE-DMP dependency-mask pre-training, not a vanilla BERT-base initialization.

Use this directory when fine-tuning IGSC:

```
ENCODER_DIR=./bert_base_uncased
```

Pass it explicitly with `--model_name "$ENCODER_DIR"` in the training command. `parameter.py` currently defaults to `./bert-base-uncased` (with hyphens), so do not rely on that default when you intend to load `./bert_base_uncased/`.

### Re-training the structure-aware encoder

If the structure-aware encoder needs to be retrained, follow the original RE-DMP paper and its official implementation:

- Paper: [Improving Relation Extraction through Syntax-induced Pre-training with Dependency Masking (Tian et al., 2022)](https://aclanthology.org/2022.findings-acl.147/).
- Official code: [synlp/RE-DMP](https://github.com/synlp/RE-DMP).
- Pre-training instructions: [`pre-training/readme.md`](https://github.com/synlp/RE-DMP/blob/main/pre-training/readme.md).

The paper's pre-training task masks and predicts dependency connections and dependency types for first-, second-, and third-order dependencies. Its reported setup uses auto-parsed English Wikipedia: sentences shorter than 10 words are filtered, yielding approximately 92 million sentences and 2.38 billion tokens. The paper describes using the Berkeley Neural Parser (trained on the Penn Treebank) and Stanford Dependency converter v3.3.0 to construct dependency data. The reported pre-training configuration uses 700K steps and a batch size of 32.

## Five-fold training and evaluation

After preparing the feature caches, use the RE-DMP-pre-trained structure-aware encoder already provided in the project directory. Set `ENCODER_DIR` to the following path:

```bash
ENCODER_DIR=./bert_base_uncased
```

Run the extraction experiment:

```bash
cd /path/to/IGSC
ENCODER_DIR=./bert_base_uncased

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

### Saving complete models and exporting cases

To save best-validation models, exact fold indices, configuration, and held-out iterative trajectories, use the optional entry:

```bash
python -u case_study/train_and_export.py \
  --output case_study/runs/igsc_seed209 \
  --model_name "$ENCODER_DIR" \
  --cache_path ./data/database \
  --seed 209 --epoch 30 --batch_size 1 --k_fold 5 \
  --beta_intra 0.7 --beta_inter 0.3 --w 0.6 --threshold 2
```

Add `--preflight` to check inputs and the environment before starting a new run. 

## References and attribution

The iterative causal graph implementation is derived from:

> C. Liu, W. Xiang, and B. Wang. 2024. *Identifying while Learning for Document Event Causality Identification*. ACL, pages 3815-3827. [Paper](https://aclanthology.org/2024.acl-long.210/). [iLIF code](https://github.com/LchengC/iLIF).

Dependency-mask pre-training follows:

> Y. Tian, Y. Song, and F. Xia. 2022. *Improving Relation Extraction through Syntax-induced Pre-training with Dependency Masking*. Findings of ACL, pages 1875-1886. [Paper](https://aclanthology.org/2022.findings-acl.147/). 

Dataset reference:

> T. Caselli and P. Vossen. 2017. *The Event StoryLine Corpus: A New Benchmark for Causal and Temporal Relation Extraction*. Events and Stories in the News Workshop, pages 77-86. [Paper](https://aclanthology.org/W17-2711/).

The inherited iLIF implementation also acknowledges [HAN](https://github.com/Jhy1993/HAN) and deterministic graph-operation changes contributed by [lyj963](https://github.com/lyj963). 

