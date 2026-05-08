# DocBank Text-Only Token Classification (DistilBERT)

## Task
This project implements a production-style homework pipeline for token classification on DocBank annotations using only text tokens.

The objective is to classify document tokens into layout/semantic classes while filtering out non-text-focused labels (figures, tables, equations, captions).

## Dataset
- Source: DocBank text annotations.
- Path used in this project: `./data/docbank/DocBank_500K_txt/`
- Input unit: one `.txt` file per page with token-level annotations.

## Why Text-Only
This homework intentionally avoids visual pipelines:
- No images
- No OCR
- No LayoutLM or other multimodal models

The model receives only token sequences and corresponding labels extracted from DocBank text files.

## Ignored Labels
Configured in `configs/config.yaml` and removed during preprocessing:
- `figure`
- `table`
- `equation`
- `caption`
- `figure_caption`
- `table_caption`

## Project Structure

```text
lesson_18/
├── data/
│   ├── docbank/
│   │   ├── DocBank_500K_txt.zip
│   │   └── DocBank_500K_txt/
│   └── processed/
├── configs/
│   └── config.yaml
├── src/
│   ├── data/
│   │   ├── docbank_reader.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/
│   │   └── distilbert_token_classifier.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── distill_student.py
│   ├── utils/
│   │   ├── seed.py
│   │   ├── logger.py
│   │   └── io.py
│   └── main.py
├── outputs/
│   ├── models/
│   ├── student_model/
│   ├── logs/
│   ├── reports/
│   └── predictions/
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── README.md
└── run.sh
```

## Model Architecture
- Backbone: `distilbert-base-uncased`
- Head: token classification head
- Main mode: freeze DistilBERT backbone and train classifier head only
- Optional mode: unfreeze last `N` transformer layers via `unfreeze_last_n_layers`

## Training Approach
- Uses Hugging Face `Trainer`
- Early stopping with `early_stopping_patience`
- Automatic mixed precision (`fp16`) when CUDA is available
- Tokenization with `is_split_into_words=True`
- Word-to-subword label alignment:
  - Special tokens get label `-100`
  - Subwords get the same label as the source word
- Dataset size is truncated/subsampled with config limits:
  - `train_samples`
  - `val_samples`
  - `test_samples`

## Metrics
Token-level metrics are computed while ignoring `-100` labels:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 (weighted)

## Configuration
Main parameters are in `configs/config.yaml`:
- data/model paths
- max sequence length
- split sizes
- training hyperparameters
- freezing strategy
- early stopping and mixed precision flags
- ignored labels

## Baseline Experiment (GPU)
Run date: 2026-05-07

Training setup:
- GPU: NVIDIA GeForce RTX 3060 (12 GB)
- `torch.cuda.is_available() = True`
- Subset size request: train=3000, val=500, test=500
- Effective split: train=3000, val=500, test=500
- Epochs: 3
- Batch size: 8
- Learning rate: 3e-4
- Backbone mode: frozen DistilBERT + trainable token head
- Mixed precision: enabled (`fp16=True`)
- Early stopping: enabled (`patience=2`)

Final metrics:
- Validation F1: 0.8015
- Validation Accuracy: 0.8523
- Test F1: 0.8346
- Test Accuracy: 0.8767
- Total training time: 55.90 seconds

GPU memory snapshot:
- Before training: 380 MB / 12288 MB
- After training: 860 MB / 12288 MB

## Knowledge Distillation
Teacher model:
- Fine-tuned DistilBERT token classifier loaded from `./outputs/models/best_model`.

Student model:
- Lightweight architecture in `src/models/cnn_student_token_classifier.py`.
- `Embedding -> 1D CNN -> Dropout -> Linear`.

Distillation loss:
- Hard loss: CrossEntropy on ground-truth labels (ignoring `-100`).
- Soft loss: KLDiv between teacher and student logits with temperature scaling.
- Combined objective:
  - `total_loss = alpha * hard_loss + (1 - alpha) * distillation_loss`
  - default `temperature=2.0`, `alpha=0.5`

Distillation outputs:
- Student weights and tokenizer: `outputs/student_model/`
- Student metrics + teacher vs student comparison: `outputs/reports/student_metrics.json`
- Student training log by epochs: `outputs/reports/student_training_history.json`

Comparison includes:
- accuracy, precision, recall, f1
- model size (MB)
- inference speed on test split (samples/sec)

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start training:

```bash
python -m src.main --config configs/config.yaml
```

or

```bash
bash run.sh
```

## Expected Outputs
After training, artifacts are saved under `outputs/`:
- `outputs/models/best_model/`: best model + tokenizer
- `outputs/logs/train.log`: training logs
- `outputs/reports/label_mapping.json`: label to id mappings
- `outputs/reports/metrics.json`: train/val/test metrics
- `outputs/reports/classification_report.json`: per-label metrics
- `outputs/reports/training_history.json`: full Trainer log history
- `outputs/reports/loss_plot.png`: training loss curve
- `outputs/reports/f1_plot.png`: validation F1 curve
- `outputs/reports/training_curves.png`: combined training curves
- `outputs/reports/confusion_matrix.png`: token-level confusion matrix
- `outputs/reports/gpu_stats.json`: CUDA/GPU info, VRAM before/after, training time
- `outputs/reports/prediction_examples.json`: token -> predicted_label -> true_label examples
- `outputs/reports/run_config.json`: config snapshot
- `outputs/predictions/test_predictions.json`: raw prediction ids

## Limitations
- Training uses only a subset of the full DocBank dataset due to hardware/time constraints.
- The approach is text-only and does not use page geometry or visual features.
- Rare labels can be challenging without larger sample counts or class balancing.
- Class imbalance (dominant `paragraph` label) still affects weighted metrics and confusion matrix interpretation.
- Some pages become invalid/empty after label filtering, so exact requested split sizes can be slightly reduced.
