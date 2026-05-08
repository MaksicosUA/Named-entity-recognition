from __future__ import annotations

import torch
from torch import nn


class CNNStudentTokenClassifier(nn.Module):
    """Lightweight token classifier: Embedding -> 1D CNN -> Dropout -> Linear."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids)  # [batch, seq_len, emb]
        x = x.transpose(1, 2)  # [batch, emb, seq_len] for Conv1d
        x = self.conv1d(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden]
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch, seq_len, num_labels]
        return logits
