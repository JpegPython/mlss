import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from typing import Any, Dict, List, Tuple
from pytorch_lightning.callbacks.base import Callback
from mlss.unet import UNET, ImageBatch, Logits, SemanticBatch
import operator
import torch


class UNETModule(pl.LightningModule):

    def __init__(self, in_channels: int, num_classes: int, lr: float, early_stopping_patience: int, lr_scheduler_patience: int) -> None:
        super().__init__()
        if num_classes > 2:
            raise ValueError('Expected more than two classes (num_classes > 2)')
        if early_stopping_patience <= lr_scheduler_patience:
            raise ValueError('Invalid patience values (early_stopping_patience <= lr_scheduler_patience)')
        self.save_hyperparameters()
        # Set ome properties.
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        # Create the U-Net network.
        self.net = UNET(enc_channels=(in_channels, 64, 128, 256, 512, 1024), dec_channels=(1024, 512, 256, 128, num_classes))
        # Create the loss and metrics's functions.
        self.loss_fn = nn.CrossEntropyLoss()

    def _common_step(self, batch: Tuple[ImageBatch, SemanticBatch], task: str) -> Dict[str, Any]:
        images, targets = batch
        # Apply the semantic segmentation network.
        logits = self.net(images)
        # Compute and log the loss and the accuracy based on model output and real labels.
        loss = self.loss_fn(logits, targets)
        self.log(f'{task}/Loss/Step', loss)
        acc = torchmetrics.functional.accuracy(logits, targets)
        self.log(f'{task}/Accuracy/Step', acc)
        # Return loss and metrics.
        return {'loss': loss, 'acc': acc.detach()}

    def _common_epoch_end(self, step_outputs: List[Dict[str, Any]], task: str) -> None:
        loss = torch.stack(list(map(operator.itemgetter('loss'), step_outputs)))
        acc = torch.stack(list(map(operator.itemgetter('acc'), step_outputs)))
        # Compute and log the mean loss and the mean accuracy.
        self.log(f'{task}/Loss', loss.mean())
        self.log(f'{task}/Accuracy', acc.mean())
    
    def training_step(self, batch: Tuple[ImageBatch, SemanticBatch], batch_idx: int) -> Dict[str, Any]:
        return self._common_step(batch, 'Train')

    def training_epoch_end(self, step_outputs: List[Dict[str, Any]]) -> None:
        self._common_epoch_end(step_outputs, 'Train')

    def validation_step(self, batch: Tuple[ImageBatch, SemanticBatch], batch_idx: int) -> Dict[str, Any]:
        return self._common_step(batch, 'Val')

    def validation_epoch_end(self, step_outputs: List[Dict[str, Any]]) -> None:
        self._common_epoch_end(step_outputs, 'Val')

    def test_step(self, batch: Tuple[ImageBatch, SemanticBatch], batch_idx: int) -> Dict[str, Any]:
        return self._common_step(batch, 'Test')

    def test_epoch_end(self, step_outputs: List[Dict[str, Any]]) -> None:
        self._common_epoch_end(step_outputs, 'Test')

    def configure_callbacks(self) -> List[Callback]:
        # Apply early stopping.
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='Val/Accuracy', mode='max', patience=self.early_stopping_patience)
        # Return the list of callbacks.
        return [early_stopping]

    def configure_optimizers(self) -> Dict[str, Any]:
        # Set the optimizer.
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0001)
        # Set the learning rate scheduler.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_scheduler_patience)
        # Return the configuration.
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'Val/Loss'}},

    def forward(self, batch: ImageBatch) -> Logits:
        return self.net(batch)
