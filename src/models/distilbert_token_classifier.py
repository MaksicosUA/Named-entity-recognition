from __future__ import annotations

from transformers import AutoModelForTokenClassification


def build_distilbert_token_classifier(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    freeze_backbone: bool = True,
    unfreeze_last_n_layers: int = 0,
):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    if not hasattr(model, "distilbert"):
        return model

    if freeze_backbone:
        for param in model.distilbert.parameters():
            param.requires_grad = False

    if unfreeze_last_n_layers > 0:
        layers = model.distilbert.transformer.layer
        n = min(unfreeze_last_n_layers, len(layers))

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Classifier head should stay trainable.
    for param in model.classifier.parameters():
        param.requires_grad = True

    if hasattr(model, "pre_classifier"):
        for param in model.pre_classifier.parameters():
            param.requires_grad = True

    return model
