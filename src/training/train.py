from __future__ import annotations

import inspect
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.training.evaluate import compute_token_metrics, flatten_valid_tokens, token_level_report
from src.utils.io import save_json


class LoggingCallback(TrainerCallback):
    """Forward trainer logs to the project logger."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("trainer_log | step=%s | %s", state.global_step, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("trainer_eval | step=%s | %s", state.global_step, metrics)

    def on_save(self, args, state, control, **kwargs):
        self.logger.info("trainer_save | step=%s | checkpoint=%s", state.global_step, args.output_dir)


def _read_nvidia_smi() -> dict[str, int] | None:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        first_line = res.stdout.strip().splitlines()[0]
        used_str, total_str = [x.strip() for x in first_line.split(",")]
        return {"used_mb": int(used_str), "total_mb": int(total_str)}
    except Exception:
        return None


def _gpu_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": str(torch.version.cuda),
    }
    smi_info = _read_nvidia_smi()
    if smi_info is not None:
        info["nvidia_smi"] = smi_info

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["torch_allocated_mb"] = float(torch.cuda.memory_allocated(0) / (1024 ** 2))
        info["torch_reserved_mb"] = float(torch.cuda.memory_reserved(0) / (1024 ** 2))

    return info


def _save_history_and_plots(log_history: list[dict[str, Any]], reports_dir: Path) -> None:
    save_json({"history": log_history}, reports_dir / "training_history.json")

    train_steps: list[float] = []
    train_losses: list[float] = []
    eval_epochs: list[float] = []
    eval_f1s: list[float] = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(float(entry.get("step", 0.0)))
            train_losses.append(float(entry["loss"]))
        if "eval_f1" in entry:
            eval_epochs.append(float(entry.get("epoch", 0.0)))
            eval_f1s.append(float(entry["eval_f1"]))

    if train_steps and train_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_steps, train_losses, marker="o")
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / "loss_plot.png", dpi=150)
        plt.close()

    if eval_epochs and eval_f1s:
        plt.figure(figsize=(8, 5))
        plt.plot(eval_epochs, eval_f1s, marker="o")
        plt.title("Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / "f1_plot.png", dpi=150)
        plt.close()

    if train_steps and train_losses and eval_epochs and eval_f1s:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_steps, train_losses, marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(eval_epochs, eval_f1s, marker="o")
        axes[1].set_title("Validation F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(reports_dir / "training_curves.png", dpi=150)
        plt.close(fig)


def _save_confusion_matrix(y_true: list[int], y_pred: list[int], id2label: dict[int, str], output_path: Path) -> None:
    if not y_true:
        return

    label_ids = sorted(set(y_true) | set(y_pred))
    labels = [id2label[x] for x in label_ids]
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Token-Level Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_prediction_examples(
    model,
    tokenizer,
    raw_test_samples: Sequence[Any],
    id2label: dict[int, str],
    max_length: int,
    output_path: Path,
    num_examples: int,
    device: torch.device,
) -> None:
    examples: list[dict[str, Any]] = []
    model.eval()

    for sample in raw_test_samples[:num_examples]:
        enc = tokenizer(
            sample.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            pred_ids = torch.argmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

        rows: list[dict[str, str]] = []
        seen_word_ids: set[int] = set()
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen_word_ids:
                continue
            if word_idx >= len(sample.tokens):
                continue

            seen_word_ids.add(word_idx)
            token = sample.tokens[word_idx]
            true_label = sample.labels[word_idx]
            pred_label = id2label.get(int(pred_ids[token_idx]), "unknown")

            rows.append(
                {
                    "token": token,
                    "predicted_label": pred_label,
                    "true_label": true_label,
                }
            )

        examples.append({"file_name": sample.file_name, "predictions": rows[:80]})

    save_json({"examples": examples}, output_path)


def train_and_evaluate(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    raw_test_samples,
    output_dir: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    logger,
    id2label: dict[int, str],
    label2id: dict[str, int],
    max_length: int,
    early_stopping_patience: int,
    use_mixed_precision: bool,
    prediction_examples: int,
) -> dict[str, Any]:
    out = Path(output_dir)
    model_dir = out / "models"
    logs_dir = out / "logs"
    reports_dir = out / "reports"
    preds_dir = out / "predictions"

    for p in [model_dir, logs_dir, reports_dir, preds_dir]:
        p.mkdir(parents=True, exist_ok=True)

    gpu_before = _gpu_snapshot()
    logger.info("torch.cuda.is_available()=%s", gpu_before.get("torch_cuda_available"))
    if gpu_before.get("gpu_name"):
        logger.info("GPU name: %s", gpu_before.get("gpu_name"))
    logger.info("VRAM before training: %s", gpu_before.get("nvidia_smi", {}))

    use_fp16 = bool(use_mixed_precision and torch.cuda.is_available())
    logger.info("Mixed precision (fp16) enabled: %s", use_fp16)

    training_kwargs = {
        "output_dir": str(model_dir / "checkpoints"),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "fp16" in ta_params:
        training_kwargs["fp16"] = use_fp16

    training_args = TrainingArguments(**training_kwargs)

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    callbacks: list[TrainerCallback] = [LoggingCallback(logger)]
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "compute_metrics": compute_token_metrics,
        "callbacks": callbacks,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    start_time = time.perf_counter()
    logger.info("Starting training...")
    train_output = trainer.train()

    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)

    logger.info("Evaluating on test set...")
    test_predictions = trainer.predict(test_dataset=test_dataset)
    total_train_seconds = float(time.perf_counter() - start_time)
    test_metrics = {
        k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in test_predictions.metrics.items()
    }

    report = token_level_report(
        predictions=test_predictions.predictions,
        labels=test_predictions.label_ids,
        id2label=id2label,
    )

    y_true, y_pred = flatten_valid_tokens(test_predictions.predictions, test_predictions.label_ids)
    _save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        id2label=id2label,
        output_path=reports_dir / "confusion_matrix.png",
    )

    logger.info("Saving best model and tokenizer...")
    trainer.save_model(str(model_dir / "best_model"))
    tokenizer.save_pretrained(str(model_dir / "best_model"))

    _save_history_and_plots(trainer.state.log_history, reports_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _save_prediction_examples(
        model=model,
        tokenizer=tokenizer,
        raw_test_samples=raw_test_samples,
        id2label=id2label,
        max_length=max_length,
        output_path=reports_dir / "prediction_examples.json",
        num_examples=prediction_examples,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_after = _gpu_snapshot()
    logger.info("VRAM after training: %s", gpu_after.get("nvidia_smi", {}))

    save_json(
        {
            "before": gpu_before,
            "after": gpu_after,
            "fp16_enabled": use_fp16,
            "total_training_seconds": total_train_seconds,
        },
        reports_dir / "gpu_stats.json",
    )

    all_metrics = {
        "train": {
            "train_loss": float(train_output.training_loss),
            "train_steps": int(train_output.global_step),
            "total_training_seconds": total_train_seconds,
        },
        "validation": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float, np.floating))},
        "test": test_metrics,
    }

    save_json(all_metrics, reports_dir / "metrics.json")
    save_json(report, reports_dir / "classification_report.json")

    # Save compact token predictions for further inspection.
    pred_ids = np.argmax(test_predictions.predictions, axis=-1).tolist()
    label_ids = test_predictions.label_ids.tolist()
    save_json(
        {
            "pred_ids": pred_ids,
            "label_ids": label_ids,
        },
        preds_dir / "test_predictions.json",
    )

    return all_metrics
