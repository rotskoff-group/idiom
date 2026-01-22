import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.output(x)
        return x
