from torch import nn
from .resnet import feature_extractor


class Embedder(nn.Module):
    def __init__(self, *, embedding_dim: int = 100):
        super().__init__()

        self.embedder = nn.Sequential(
            feature_extractor,
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(embedding_dim)
        )

    def forward(self, x):
        return self.embedder(x)
