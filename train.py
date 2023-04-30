import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from data import DataModule
from model import ColaModel

mlf_logger = MLFlowLogger(experiment_name="cola", tracking_uri="file:./mlruns")


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        # default_root_dir="logs",
        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
        # accelerator='cpu',
        max_epochs=5,
        fast_dev_run=False,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()