from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("rnn_encoders")

RNN_REGISTRY: dict[str, dict] = {}


def register_rnn(name: str, default_output_dim: int = 256):
    def decorator(cls):
        RNN_REGISTRY[name] = {"class": cls, "output_dim": default_output_dim}
        return cls
    return decorator


class BaseRNNEncoder(nn.Module):
    output_dim: int = 256

    def __init__(self, input_dim: int = 7, output_dim: int | None = None):
        super().__init__()
        if output_dim is not None:
            self.output_dim = output_dim

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register_rnn("Simple RNN", default_output_dim=128)
class SimpleRNNEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 128):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True, nonlinearity="tanh")
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        pooled = output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("LSTM", default_output_dim=256)
class LSTMEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256, num_layers: int = 1):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.rnn = nn.LSTM(
            input_size=input_dim, hidden_size=output_dim, num_layers=num_layers,
            batch_first=True, dropout=0.0 if num_layers == 1 else 0.2,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        pooled = output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("BiLSTM", default_output_dim=256)
class BiLSTMEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        hidden = output_dim // 2
        self.rnn = nn.LSTM(
            input_size=input_dim, hidden_size=hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        pooled = output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("GRU", default_output_dim=256)
class GRUEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.rnn = nn.GRU(
            input_size=input_dim, hidden_size=output_dim, num_layers=2,
            batch_first=True, dropout=0.2,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        pooled = output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("BiGRU", default_output_dim=256)
class BiGRUEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        hidden = output_dim // 2
        self.rnn = nn.GRU(
            input_size=input_dim, hidden_size=hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        pooled = output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("Stacked LSTM", default_output_dim=256)
class StackedLSTMEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256, num_layers: int = 3):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.rnn = nn.LSTM(
            input_size=input_dim, hidden_size=output_dim, num_layers=num_layers,
            batch_first=True, dropout=0.3,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(sequence)
        pooled = output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("Seq2Seq", default_output_dim=256)
class Seq2SeqEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.encoder_rnn = nn.LSTM(
            input_size=input_dim, hidden_size=output_dim, num_layers=2,
            batch_first=True, dropout=0.2,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder_rnn(sequence)
        context = hidden[-1]
        return self.projector(context)


@register_rnn("Encoder-Decoder LSTM", default_output_dim=256)
class EncoderDecoderLSTMEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        hidden = output_dim // 2
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.decoder = nn.LSTM(
            input_size=hidden * 2, hidden_size=output_dim, num_layers=1,
            batch_first=True,
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        enc_output, _ = self.encoder(sequence)
        dec_output, _ = self.decoder(enc_output)
        pooled = dec_output.mean(dim=1)
        return self.projector(pooled)


@register_rnn("TCN", default_output_dim=256)
class TCNEncoder(BaseRNNEncoder):
    def __init__(self, input_dim: int = 7, output_dim: int = 256):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, output_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(output_dim // 2),
            nn.Conv1d(output_dim // 2, output_dim // 2, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(output_dim // 2),
            nn.Conv1d(output_dim // 2, output_dim, kernel_size=1),
            nn.GELU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        features = self.net(sequence.transpose(1, 2))
        pooled = features.mean(dim=2)
        return self.projector(pooled)


class RNNEncoderFactory:
    @staticmethod
    def list_models() -> list[str]:
        return sorted(RNN_REGISTRY.keys())

    @staticmethod
    def get_info(model_name: str) -> dict:
        normalized = _normalize_name(model_name)
        if normalized not in RNN_REGISTRY:
            raise KeyError(f"Unknown RNN model: {model_name}. Available: {sorted(RNN_REGISTRY.keys())}")
        info = RNN_REGISTRY[normalized]
        return {"name": normalized, "output_dim": info["output_dim"]}

    @staticmethod
    def create(model_name: str, input_dim: int = 7, output_dim: int | None = None) -> BaseRNNEncoder:
        normalized = _normalize_name(model_name)
        if normalized not in RNN_REGISTRY:
            raise KeyError(f"Unknown RNN model: {model_name}. Available: {sorted(RNN_REGISTRY.keys())}")
        info = RNN_REGISTRY[normalized]
        return info["class"](input_dim=input_dim, output_dim=output_dim or info["output_dim"])


def _normalize_name(name: str) -> str:
    mapping = {
        "simple_rnn": "Simple RNN",
        "simplernn": "Simple RNN",
        "rnn": "Simple RNN",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "gru": "GRU",
        "bigru": "BiGRU",
        "stacked_lstm": "Stacked LSTM",
        "stackedlstm": "Stacked LSTM",
        "seq2seq": "Seq2Seq",
        "encoder_decoder_lstm": "Encoder-Decoder LSTM",
        "encoderdecoderlstm": "Encoder-Decoder LSTM",
        "tcn": "TCN",
        "temporal_convolutional_network": "TCN",
    }
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key, name)
