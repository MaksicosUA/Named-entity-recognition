from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from src.data.dataset import DocBankTokenDataset
from src.data.preprocessing import build_label_mappings, load_samples
from src.models.distilbert_token_classifier import build_distilbert_token_classifier
from src.training.distill_student import run_student_distillation
from src.training.train import train_and_evaluate
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def split_samples(samples: List[Any], train_n: int, val_n: int, test_n: int, seed: int) -> Tuple[List[Any], List[Any], List[Any]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    total_needed = train_n + val_n + test_n
    sliced = shuffled[:total_needed]

    train_end = min(train_n, len(sliced))
    val_end = min(train_end + val_n, len(sliced))
    test_end = min(val_end + test_n, len(sliced))

    train_split = sliced[:train_end]
    val_split = sliced[train_end:val_end]
    test_split = sliced[val_end:test_end]

    return train_split, val_split, test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocBank text-only token classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_yaml(args.config)

    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "reports")

    logger = get_logger("docbank_training", output_dir / "logs" / "train.log")
    logger.info("Loading config from %s", args.config)

    set_seed(int(cfg["seed"]))

    total_samples = int(cfg["train_samples"]) + int(cfg["val_samples"]) + int(cfg["test_samples"])
    input_files_limit = int(cfg.get("input_files_limit", total_samples * 2))
    input_files_limit = max(input_files_limit, total_samples)
    logger.info("Reading up to %d input files for target %d samples", input_files_limit, total_samples)

    samples = load_samples(
        data_dir=cfg["data_dir"],
        max_samples=input_files_limit,
        ignored_labels=cfg.get("ignored_labels", []),
        min_label_count=int(cfg.get("min_label_count", 0)),
        other_label=str(cfg.get("other_label", "other")),
    )

    if len(samples) < total_samples:
        raise RuntimeError(
            f"Not enough valid samples were loaded. Needed={total_samples}, loaded={len(samples)}. "
            "Increase input_files_limit in config."
        )

    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_n=int(cfg["train_samples"]),
        val_n=int(cfg["val_samples"]),
        test_n=int(cfg["test_samples"]),
        seed=int(cfg["seed"]),
    )

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(
            "One of the dataset splits is empty. Reduce requested sample counts or check data quality."
        )

    logger.info(
        "Split sizes | train=%d val=%d test=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    label2id, id2label = build_label_mappings([s.labels for s in train_samples + val_samples + test_samples])
    logger.info("Number of labels: %d", len(label2id))

    save_json(
        {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        },
        output_dir / "reports" / "label_mapping.json",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_ds = DocBankTokenDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=int(cfg["max_length"]),
    )
    val_ds = DocBankTokenDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=int(cfg["max_length"]),
    )
    test_ds = DocBankTokenDataset(
        samples=test_samples,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=int(cfg["max_length"]),
    )

    model = build_distilbert_token_classifier(
        model_name=cfg["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        unfreeze_last_n_layers=int(cfg.get("unfreeze_last_n_layers", 0)),
    )

    metrics = train_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        raw_test_samples=test_samples,
        output_dir=str(output_dir),
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        logger=logger,
        id2label=id2label,
        label2id=label2id,
        max_length=int(cfg["max_length"]),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 2)),
        use_mixed_precision=bool(cfg.get("use_mixed_precision", True)),
        prediction_examples=int(cfg.get("prediction_examples", 5)),
    )

    save_json(cfg, output_dir / "reports" / "run_config.json")
    logger.info("Training finished. Test metrics: %s", metrics.get("test", {}))

    distill_cfg = cfg.get("distillation", {})
    if bool(distill_cfg.get("enabled", False)):
        logger.info("Distillation enabled. Starting student distillation...")
        student_result = run_student_distillation(
            distill_cfg_raw=distill_cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
            label2id=label2id,
            output_dir=str(output_dir),
            logger=logger,
        )
        logger.info("Distillation finished. Student metrics: %s", student_result.get("student_test_metrics", {}))


if __name__ == "__main__":
    main()
