import torch
from torch import nn
import math

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V)
        return out, attn_weights


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_features: int = 14,
        cnn_channels: int = 64,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.15
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            num_features, cnn_channels,
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            cnn_channels, cnn_channels,
            kernel_size=3, padding=1
        )

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        self.ln = nn.LayerNorm(cnn_channels)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim=lstm_hidden * 2)

        self.fc1 = nn.Linear(lstm_hidden * 2, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 5)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.permute(0, 2, 1)

        x = self.ln(x)

        lstm_out, _ = self.lstm(x)

        attn_out, attn_weights = self.attention(lstm_out)

        out = attn_out.mean(dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)

        return out
