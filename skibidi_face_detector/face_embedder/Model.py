import pytorch_lightning as pl
import torch
import einops

from .ArcFaceLoss import ArcFaceLoss
from .Embedder import Embedder
from .. import face


class Model(pl.LightningModule):
    def __init__(self, num_classes: int, *, embedding_dim: int = 100, margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.save_hyperparameters()
        self.embedder = Embedder(embedding_dim=embedding_dim)

        self.arc_face_loss = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=margin,
            scale=scale
        )

    def forward(self, x):
        return self.embedder(x)

    def transform_batch(self, batch):
        x, y = batch['image'], batch['label']

        x = [face.align_most_confident_face(_x.cpu(), warn=False) for _x in x]
        x = torch.stack(x).to(self.device)

        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.transform_batch(batch)

        embeddings = self.embedder(x)
        loss = self.arc_face_loss(embeddings, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.transform_batch(batch)

        embeddings = self.embedder(x)
        loss = self.arc_face_loss(embeddings, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
