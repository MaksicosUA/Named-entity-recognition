from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from src.models.cnn_student_token_classifier import CNNStudentTokenClassifier
from src.training.evaluate import flatten_valid_tokens
from src.utils.io import save_json


@dataclass
class DistillationConfig:
    enabled: bool = False
    teacher_model_dir: str = "./outputs/models/best_model"
    student_epochs: int = 3
    student_batch_size: int = 16
    temperature: float = 2.0
    alpha: float = 0.5
    learning_rate: float = 1e-3


def _to_distill_config(cfg: dict[str, Any]) -> DistillationConfig:
    return DistillationConfig(
        enabled=bool(cfg.get("enabled", False)),
        teacher_model_dir=str(cfg.get("teacher_model_dir", "./outputs/models/best_model")),
        student_epochs=int(cfg.get("student_epochs", 3)),
        student_batch_size=int(cfg.get("student_batch_size", 16)),
        temperature=float(cfg.get("temperature", 2.0)),
        alpha=float(cfg.get("alpha", 0.5)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )


def _evaluate_logits(pred_logits: list[np.ndarray], true_labels: list[np.ndarray]) -> Dict[str, float]:
    if not pred_logits:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    preds = np.concatenate(pred_logits, axis=0)
    labels = np.concatenate(true_labels, axis=0)
    y_true, y_pred = flatten_valid_tokens(preds, labels)

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return float(total_bytes / (1024 ** 2))


def _run_eval(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[Dict[str, float], float]:
    model.eval()
    pred_logits: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []

    start = perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            pred_logits.append(logits.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
    elapsed = float(perf_counter() - start)

    return _evaluate_logits(pred_logits, true_labels), elapsed


def run_student_distillation(
    distill_cfg_raw: dict[str, Any],
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer,
    label2id: dict[str, int],
    output_dir: str,
    logger,
) -> Dict[str, Any]:
    cfg = _to_distill_config(distill_cfg_raw)
    if not cfg.enabled:
        logger.info("Distillation is disabled in config.")
        return {"enabled": False}

    out_dir = Path(output_dir)
    reports_dir = out_dir / "reports"
    student_dir = out_dir / "student_model"
    reports_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting distillation on device=%s", device)

    teacher = AutoModelForTokenClassification.from_pretrained(cfg.teacher_model_dir)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = CNNStudentTokenClassifier(
        vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        num_labels=len(label2id),
        pad_token_id=int(tokenizer.pad_token_id or 0),
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=cfg.student_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.student_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.student_batch_size, shuffle=False)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    history: list[Dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0

    temperature = cfg.temperature
    alpha = cfg.alpha

    for epoch in range(1, cfg.student_epochs + 1):
        student.train()
        running_loss = 0.0
        running_hard = 0.0
        running_soft = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask)

            valid_mask = labels != -100
            if valid_mask.sum() == 0:
                continue

            student_valid = student_logits[valid_mask]
            teacher_valid = teacher_logits[valid_mask]
            labels_valid = labels[valid_mask]

            hard = ce_loss(student_valid, labels_valid)
            soft = kl_loss(
                torch.log_softmax(student_valid / temperature, dim=-1),
                torch.softmax(teacher_valid / temperature, dim=-1),
            ) * (temperature * temperature)

            total = alpha * hard + (1.0 - alpha) * soft

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            running_loss += float(total.item())
            running_hard += float(hard.item())
            running_soft += float(soft.item())
            steps += 1

        train_loss = running_loss / max(steps, 1)
        train_hard = running_hard / max(steps, 1)
        train_soft = running_soft / max(steps, 1)

        val_metrics, _ = _run_eval(student, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_hard_loss": train_hard,
            "train_soft_loss": train_soft,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_row)
        logger.info("distill_epoch=%d | %s", epoch, epoch_row)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    if best_state is not None:
        student.load_state_dict(best_state)

    teacher_metrics, teacher_eval_time = _run_eval(teacher, test_loader, device)
    student_metrics, student_eval_time = _run_eval(student, test_loader, device)

    student_size_mb = _model_size_mb(student)
    teacher_size_mb = _model_size_mb(teacher)

    speed = {
        "teacher_eval_seconds": teacher_eval_time,
        "student_eval_seconds": student_eval_time,
        "teacher_samples_per_second": float(len(test_dataset) / max(teacher_eval_time, 1e-8)),
        "student_samples_per_second": float(len(test_dataset) / max(student_eval_time, 1e-8)),
    }

    comparison = {
        "accuracy": {
            "teacher": teacher_metrics["accuracy"],
            "student": student_metrics["accuracy"],
        },
        "precision": {
            "teacher": teacher_metrics["precision"],
            "student": student_metrics["precision"],
        },
        "recall": {
            "teacher": teacher_metrics["recall"],
            "student": student_metrics["recall"],
        },
        "f1": {
            "teacher": teacher_metrics["f1"],
            "student": student_metrics["f1"],
        },
        "model_size_mb": {
            "teacher": teacher_size_mb,
            "student": student_size_mb,
        },
        "inference": speed,
    }

    torch.save(student.state_dict(), student_dir / "student_model.pt")
    save_json(
        {
            "architecture": "Embedding -> Conv1d -> Dropout -> Linear",
            "num_labels": len(label2id),
            "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
            "distillation_config": {
                "temperature": cfg.temperature,
                "alpha": cfg.alpha,
                "student_epochs": cfg.student_epochs,
                "student_batch_size": cfg.student_batch_size,
                "learning_rate": cfg.learning_rate,
            },
        },
        student_dir / "student_config.json",
    )
    tokenizer.save_pretrained(student_dir)

    save_json(
        {
            "teacher_model_dir": cfg.teacher_model_dir,
            "student_test_metrics": student_metrics,
            "teacher_test_metrics": teacher_metrics,
            "comparison": comparison,
        },
        reports_dir / "student_metrics.json",
    )
    save_json({"history": history}, reports_dir / "student_training_history.json")

    return {
        "enabled": True,
        "student_test_metrics": student_metrics,
        "teacher_test_metrics": teacher_metrics,
        "comparison": comparison,
    }
