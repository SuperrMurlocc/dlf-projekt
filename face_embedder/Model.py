from .ArcFaceLoss import ArcFaceLoss
from .Embedder import Embedder
import pytorch_lightning as pl
import torch


class Model(pl.LightningModule):
    def __init__(self, embedding_dim: int = 100):
        super().__init__()
        self.embedder = Embedder(embedding_dim=embedding_dim)

        self.loss_function = ArcFaceLoss(
            num_classes=10,
            embedding_dim=embedding_dim,
            margin=0.3,
            scale=30.0
        )

    def forward(self, x):
        return self.embedder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.embedder(x)  # (None, embedding_size)
        loss = self.loss_function(embeddings, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.embedder(x)  # (None, embedding_size)
        loss = self.loss_function(embeddings, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.embedder(x)  # (None, embedding_size)
        return {'embeddings': embeddings, 'labels': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
