from torch import nn
import torchvision.models as models


class Embedder(nn.Module):
    def __init__(self, *, embedding_dim: int = 100):
        super().__init__()

        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *(list(backbone.children())[:-2])
        )

        self.embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=embedding_dim),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        embeddings = self.embedder(features)
        return embeddings
